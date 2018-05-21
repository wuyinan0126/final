import logging

import torch
from torch import nn
from torch.autograd import Variable

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tqa.reader.models import ReaderModel, CharConvModel, CharLinearModel

logger = logging.getLogger(__name__)


class ClassifierModel(ReaderModel):
    def __init__(self, args, words, chars, extra_feature_fields, states=None, model='classifier'):
        """
        :param words: word-index词典，用于根据word获得embedding向量embedding[dictionary[word]]
        :param extra_feature_fields: 特征名称词典，用于后面构建特征向量时确定向量长度
        :param states: 神经网络状态，用于加载模型或恢复检查点时初始化网络状态
        """
        super(ClassifierModel, self).__init__(args, words, chars, extra_feature_fields, states, model)

    def update(self, examples_in_batch):
        """ 前向传播batch，step the optimizer to update weights. 每个example格式：
        0. 额外特征         f: batch * max_text_length * feature_fields长度
        1. 文本单词indices  t_w: batch * max_text_length
        2. 文本单词mask     t_mask: batch * max_text_length
        3. 文本字符indices  t_c: batch * max_text_length * max_text_word_length
        -2.类别            c: LongTensor(batch) 或 list[batch * 类别个数]
        -1.文本id          ids: list[batch]
        :return (batch loss的平均值, batch大小)
        """
        # 进入train模式
        self.network.train()

        # batch中每个example[:4]为输入
        if self.use_cuda:
            inputs = [example if example is None else Variable(example.cuda(async=True))
                      for example in examples_in_batch[:4]]
            target = Variable(examples_in_batch[-2].cuda(async=True))
        else:
            inputs = [example if example is None else Variable(example)
                      for example in examples_in_batch[:4]]
            target = Variable(examples_in_batch[-2])

        # 前向传播, score: batch * num_classes
        score = self.network(*inputs)

        """ 计算loss和准确度，NLLloss：负对数似然损失函数（Negative Log Likelihood）
        input: (batch_size * C) C = number of classes
        target: (batch_size) or (batch_size * d) where each value is 0 <= targets[i] <= C-1
        output: batch loss的平均值
        """
        # loss = F.nll_loss(score, target)

        loss = self.criterion(score, target)

        # 清除梯度、反向传播计算梯度
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：防止在BP过程中产生梯度消失/爆炸，当梯度小于/大于阈值时，更新的梯度为阈值
        clip_grad_norm(self.network.parameters(), self.args.grad_clipping)
        # 更新参数
        self.optimizer.step()
        self.updates += 1

        # 这个batch训练完后重置不需微调的embeddings，需要微调的embeddings更新的参数不能重置，以用于下一个batch继续微调
        self.reset_fixed_embeddings()
        torch.cuda.empty_cache()

        return loss.data[0], examples_in_batch[0].size(0)

    def predict(self, examples_in_batch, top_n=1):
        """ example格式：
        0. 额外特征         f: batch * max_text_length * feature_fields长度
        1. 文本单词indices  t_w: batch * max_text_length
        2. 文本单词mask     t_mask: batch * max_text_length
        3. 文本字符indices  t_c: batch * max_text_length * max_text_word_length
        -2.类别            c: LongTensor(batch) 或 list[batch * 类别个数]
        -1.文本id          ids: list[batch]
        :return prediction_class:  list[batch] 或 list[batch个array[top_n]]
        :return prediction_scores:  list[batch] 或 list[batch个array[top_n]]
        """
        # 进入eval模式
        self.network.eval()

        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True), volatile=True) for e in examples_in_batch[:4]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True) for e in examples_in_batch[:4]]

        # score: batch * num_classes
        score = self.network(*inputs)
        prediction_scores = score.data.cpu()
        prediction_class = prediction_scores.max(1)[1]

        return prediction_scores, prediction_class


class RnnClassifier(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(RnnClassifier, self).__init__()

        self.args = args
        # args.cnn_param_init = 0

        # words embedding层: 1. embedding的词典大小，2. embedding向量的大小，3. 当输入中遇到padding_idx时，输出0
        self.words_embedding = nn.Embedding(args.words_size, args.embedding_size_t, padding_idx=0)
        self.chars_embedding = nn.Embedding(args.chars_size, args.embedding_size_c, padding_idx=0)

        # char cnn层
        cnn_size = sum(args.kernel_feature_sizes)
        self.char_conv = CharConvModel(
            args.embedding_size_c, args.kernel_sizes, args.kernel_feature_sizes, param_init=args.cnn_param_init
        )
        self.char_linear = CharLinearModel(cnn_size, cnn_size, args.num_layers_c, param_init=args.cnn_param_init)

        # 输入到rnn_classifier的向量长度: t_embedding_size + extra_features_size + cnn_size
        text_rnn_input_size = args.embedding_size_t + args.num_extra_features + cnn_size

        logger.info('Document features size: %d+%d+%d' %
                    (args.embedding_size_t, args.num_extra_features, cnn_size))

        # LSTM text classifier
        self.lstm = nn.LSTM(
            input_size=text_rnn_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers_t,
            dropout=args.rnn_dropout_rate,
            bidirectional=args.bidirectional
        )

        # lstm output的最后一维大小：(num_directions * hidden_size)
        num_directions = 2 if self.args.bidirectional else 1
        text_hidden_size = num_directions * args.hidden_size

        self.linear = nn.Linear(text_hidden_size, args.num_classes)

    def forward(self, f, t_w, t_mask, t_c):
        """ 输入：
        :param f        额外特征           batch * max_text_length * feature_fields长度
        :param t_w      文本单词indices    batch * max_text_length
        :param t_mask   文本单词mask       batch * max_text_length
        :param t_c:     文本字符indices    batch * max_text_length * max_text_word_length
        :return scores                    batch * max_text_length
        """
        # --------------------------------------------------------------------------
        # Embedding特征
        # d_embedding: batch * max_document_length * embedding_size
        t_w_embedding = self.words_embedding(t_w)
        # Dropout on embeddings
        if self.args.embedding_dropout_rate > 0:
            # training: switch between training and evaluation mode
            t_w_embedding = F.dropout(t_w_embedding, p=self.args.embedding_dropout_rate, training=self.training)

        # 构建文档的d_rnn_input: [d_embedding, align_feature, f]:
        # d_rnn_input: batch * max_document_length * [embedding_size + embedding_size + feature_fields长度]
        t_rnn_input = [t_w_embedding]

        # --------------------------------------------------------------------------
        # 额外特征（文档）
        if self.args.num_extra_features > 0:
            t_rnn_input.append(f)

        # --------------------------------------------------------------------------
        # char cnn特征（文档、问题）
        # t_c_embedding: (batch * max_seq_length) * max_seq_word_length * embedding_size
        t_c_embedding = self.chars_embedding(t_c.view(-1, t_c.size(2)))
        # t_conv_output: (batch * max_seq_length) * sum(kernel_feature_size)
        t_conv_output = self.char_conv(t_c_embedding)
        # t_linear_output: (batch * max_seq_length) * sum(kernel_feature_size)
        t_linear_output = self.char_linear(t_conv_output)
        # t_cnn_feature: batch * max_seq_length * sum(kernel_feature_size)
        t_cnn_feature = t_linear_output.view(t_c.size(0), t_c.size(1), -1)
        t_rnn_input.append(t_cnn_feature)

        # --------------------------------------------------------------------------
        # lstm_input: batch * max_seq_length * tatal_features_size
        lstm_input = torch.cat(t_rnn_input, 2)

        # 计算batch里每个序列的有效长度lengths: [batch_size]
        lengths = list(t_mask.data.eq(0).long().sum(1).squeeze())
        # lstm_input: max_seq_length * batch * tatal_features_size
        lstm_input = pack_padded_sequence(lstm_input.transpose(0, 1), lengths, batch_first=False)
        # lstm_output: text_length * batch * (hidden_size * num_directions)
        lstm_output, _ = self.lstm(lstm_input)
        # lstm_output: text_length * batch * (hidden_size * num_directions)
        lstm_output, _ = pad_packed_sequence(lstm_output)
        # lstm_output: batch * text_length * (hidden_size * num_directions)
        lstm_output = lstm_output.transpose(0, 1)

        lasts = []
        # 对于Many to One的LSTM，取最后一个时间步的output
        for i in range(lstm_output.size(0)):
            # t: 1 * (hidden_size * num_directions)
            lasts.append(lstm_output[i][lengths[i] - 1].view(1, -1))

        # lstm_last_output: batch * (hidden_size * num_directions)
        lstm_last_output = torch.cat(lasts, 0).contiguous()

        # 对t_output[-1]进行线性变换和softmax
        scores = self.linear(lstm_last_output)
        # scores: batch * num_classes
        scores = F.log_softmax(scores)

        return scores.contiguous()
