import logging

import numpy
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import dropout

from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch import optim

from tqa.reader import utils
from tqa.reader.layers import AlignFeatureLayer, StackedBiRnnLayer, SelfAttentionLayer, BiLinearAttentionLayer, \
    CharConvLayer, CharLinearLayer, SelfAlignLayer, ReattentionLayer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# TqaModel
# --------------------------------------------------------------------------

class ReaderModel(object):
    """ 模型类，用于初始化网络结构，初始化embeddings，保存模型和检查点，更新examples，预测examples """

    # --------------------------------------------------------------------------
    # 初始化相关
    # --------------------------------------------------------------------------

    def __init__(self, args, words, chars, extra_feature_fields, states=None, model_type='reader'):
        """
        :param words: word-index词典，用于根据word获得embedding向量embedding[dictionary[word]]
        :param extra_feature_fields: 特征名称词典，用于后面构建特征向量时确定向量长度
        :param states: 神经网络状态，用于加载模型或恢复检查点时初始化网络状态
        """
        self.args = args
        self.words = words
        self.chars = chars
        self.args.words_size = len(words)
        self.args.chars_size = len(chars)
        self.extra_feature_fields = extra_feature_fields
        self.args.num_extra_features = len(extra_feature_fields)
        self.updates = 0
        self.use_cuda = False
        self.gpu_parallel = False
        self.criterion = nn.CrossEntropyLoss()

        from tqa.classifier.models import RnnClassifier
        self.network = RnnEncoder(args) if model_type == 'reader' else RnnClassifier(args)

        # 从训练好的模型加载网络状态
        if states:
            # 如果fixed_embeddings在状态中，单独加到buffer里
            if 'fixed_embeddings' in states:
                fixed_embedding = states.pop('fixed_embeddings')
                self.network.load_state_dict(states)
                self.network.register_buffer('fixed_embeddings', fixed_embedding)
            else:
                self.network.load_state_dict(states)

    def init_optimizer(self):
        """ 初始化Optimizer """

        """ TODO 如果embedding的词不需要微调则需要加上：
        for p in self.network.embedding.parameters():
            p.requires_grad = False
        """
        parameters = [p for p in self.network.parameters() if p.requires_grad]

        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                parameters,
                lr=self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(
                parameters,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

    # --------------------------------------------------------------------------
    # word embedding相关
    # --------------------------------------------------------------------------

    def expand_dictionary(self, words, chars):
        """ 添加words, chars到TqaModel的dictionary中如果它们不存在，并且扩展embedding层 """
        to_add_words = {utils.normalize(word) for word in words if word not in self.words}

        # Add words to dictionary and expand embedding layer
        if len(to_add_words) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add_words))
            # 添加新words到dictionary中
            for word in to_add_words:
                self.words.add(word)
            self.args.words_size = len(self.words)
            logger.info('New words dictionary size: %d' % len(self.words))

            old_embedding = self.network.words_embedding.weight.data
            self.network.words_embedding = torch.nn.Embedding(
                self.args.words_size, self.args.embedding_size_w, padding_idx=0
            )
            new_embedding = self.network.words_embedding.weight.data
            # 复制旧的embedding data，并扩展新的embeddings(with random embeddings)
            new_embedding[:old_embedding.size(0)] = old_embedding

        to_add_chars = {self.chars.normalize(char) for char in chars if char not in self.chars} if chars else set()
        if len(to_add_chars) > 0:
            logger.info('Adding %d new chars to dictionary...' % len(to_add_chars))
            # 添加新chars到dictionary中
            for char in to_add_chars:
                self.chars.add(char)
            self.args.chars_size = len(self.chars)
            logger.info('New chars dictionary size: %d' % len(self.chars))

            old_embedding = self.network.chars_embedding.weight.data
            self.network.chars_embedding = torch.nn.Embedding(
                self.args.chars_size, self.args.embedding_size_c, padding_idx=0
            )
            new_embedding = self.network.chars_embedding.weight.data
            # 复制旧的embedding data，并扩展新的embeddings(with random embeddings)
            new_embedding[:old_embedding.size(0)] = old_embedding

        return to_add_words, to_add_chars

    def load_embeddings(self, words, embedded_corpus_path):
        """ 读取glove文件，获得words在glove文件中的embedding vector（如果存在的话），加载入embedding层的data中
        :param words: 单词（原始形式）列表，只保留存在于dictionary中的
        :param embedded_corpus_path: 文件格式为word embedding_vector
        """
        words = {word for word in words if word in self.words}
        logger.info('Loading pre-trained embeddings for %d words from %s' % (len(words), embedded_corpus_path))
        # embedding.weight.data: [words_size * embedding_size_w]
        embedding = self.network.words_embedding.weight.data

        # 读取glove文件，有些word可能重复，则取它们的vector的平均值
        loaded_word_counts = {}
        with open(embedded_corpus_path) as file:
            for line in file:
                splits = line.rstrip().split(' ')
                # 文件中每一行为空格分割的 一个单词+embedding向量
                assert (len(splits) == embedding.size(1) + 1)
                glove_word = utils.normalize(splits[0])
                if glove_word in words:
                    glove_vector = torch.Tensor([float(i) for i in splits[1:]])
                    if glove_word not in loaded_word_counts:
                        loaded_word_counts[glove_word] = 1
                        embedding[self.words[glove_word]].copy_(glove_vector)
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % glove_word)
                        loaded_word_counts[glove_word] = loaded_word_counts[glove_word] + 1
                        embedding[self.words[glove_word]].add_(glove_vector)

        for glove_word, count in loaded_word_counts.items():
            embedding[self.words[glove_word]].div_(count)

        logger.info('Loaded %d embeddings' % len(loaded_word_counts))

    def tune_embeddings(self, top_words):
        """ 将top_words在embedding.weight.data中的index提前，并将top_words以外的 未微调 的词向量保存在buffer里 """
        top_words = {word for word in top_words if word in self.words}

        if len(top_words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(top_words) >= len(self.words):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        embedding = self.network.words_embedding.weight.data

        # 将top_words在embedding.weight.data中的index提前，方便微调（reset的时候只需reset后面的embeddings）
        # 从dictionary.START=2开始，因为0和1为NULL和UNK
        for top_word_new_index, top_word in enumerate(top_words, self.words.START):
            # 获得被交换的单词和它的embedding vector
            swapped_word = self.words[top_word_new_index]
            swapped_word_embedding = embedding[top_word_new_index].clone()
            # 获得top_word原始的index
            top_word_old_index = self.words[top_word]
            # top_word和swapped_word交换embedding和在dictionary中相关的值
            embedding[top_word_new_index].copy_(embedding[top_word_old_index])
            embedding[top_word_old_index].copy_(swapped_word_embedding)
            self.words[top_word] = top_word_new_index
            self.words[top_word_new_index] = top_word
            self.words[swapped_word] = top_word_old_index
            self.words[top_word_old_index] = swapped_word

        # 将top_words以外的 不需要微调 的词向量保存在buffer里
        self.network.register_buffer('fixed_embeddings', embedding[top_word_new_index + 1:].clone())

    def reset_fixed_embeddings(self):
        """ 重置不需微调的embeddings，需要微调的embeddings更新的参数不能重置，以用于下一个batch继续微调 """
        if self.args.tune_top_k > 0:
            # 不需微调的embeddings的开始index
            fixed_embeddings_start = self.args.tune_top_k + self.words.START
            if self.gpu_parallel:
                embeddings = self.network.module.embedding.weight.data
                fixed_embeddings = self.network.module.fixed_embeddings
            else:
                embeddings = self.network.words_embedding.weight.data
                fixed_embeddings = self.network.fixed_embeddings
            if fixed_embeddings_start < embeddings.size(0):
                embeddings[fixed_embeddings_start:] = fixed_embeddings

    # --------------------------------------------------------------------------
    # 学习相关
    # --------------------------------------------------------------------------

    def update(self, examples_in_batch):
        """ 前向传播batch，step the optimizer to update weights. 每个example格式：
        0. 额外特征         f: batch * max_document_length * feature_fields长度
        1. 文档单词indices  d_w: batch * max_document_length
        2. 文档单词mask     d_mask: batch * max_document_length
        3. 问题单词indices  q_w: batch * max_question_length
        4. 问题单词mask     q_mask: batch * max_question_length
        5. 文档字符indices  d_c: batch * max_document_length * max_document_word_length
        6. 问题字符indices  q_c: batch * max_question_length * max_question_word_length
        -3.答案开始span     s: LongTensor(batch) 或 list[batch * 答案个数]
        -2.答案结束span     e: LongTensor(batch) 或 list[batch * 答案个数]
        -1.问题id          ids: list[batch]
        :return (batch loss的平均值, batch大小)
        """
        # 进入train模式
        self.network.train()

        # batch中每个example[:7]为输入
        if self.use_cuda:
            inputs = [example if example is None else Variable(example.cuda(async=True))
                      for example in examples_in_batch[:7]]
            target_start = Variable(examples_in_batch[-3].cuda(async=True))
            target_end = Variable(examples_in_batch[-2].cuda(async=True))
        else:
            inputs = [example if example is None else Variable(example)
                      for example in examples_in_batch[:7]]
            target_start = Variable(examples_in_batch[-3])
            target_end = Variable(examples_in_batch[-2])

        # 前向传播, score_start, score_end: batch * max_document_length
        score_start, score_end = self.network(*inputs)
        """ 计算loss和准确度，NLLloss：负对数似然损失函数（Negative Log Likelihood）
        input: (batch_size * C) C = number of classes
        target: (batch_size) or (batch_size * d) where each value is 0 <= targets[i] <= C-1
        output: batch loss的平均值
        """
        loss = F.nll_loss(score_start, target_start) + F.nll_loss(score_end, target_end)
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

    # --------------------------------------------------------------------------
    # 预测相关
    # --------------------------------------------------------------------------

    def predict(self, examples_in_batch, top_n=1):
        """ example格式：
        0. 额外特征         f: batch * max_document_length * feature_fields长度
        1. 文档单词indices  d_w: batch * max_document_length
        2. 文档单词mask     d_mask: batch * max_document_length
        3. 问题单词indices  q_w: batch * max_question_length
        4. 问题单词mask     q_mask: batch * max_question_length
        5. 文档字符indices  d_c: batch * max_document_length * max_document_word_length
        6. 问题字符indices  q_c: batch * max_question_length * max_question_word_length
        -3.答案开始span     s: LongTensor(batch) 或 list[batch * 答案个数]
        -2.答案结束span     e: LongTensor(batch) 或 list[batch * 答案个数]
        :param top_n 得分前n的答案
        :return prediction_starts:  list[batch] 或 list[batch个array[top_n]]
        :return prediction_end:     list[batch] 或 list[batch个array[top_n]]
        :return prediction_scores:  list[batch] 或 list[batch个array[top_n]]
        """
        # 进入eval模式
        self.network.eval()

        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True), volatile=True) for e in examples_in_batch[:7]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True) for e in examples_in_batch[:7]]

        # score_start, score_end: batch * max_document_length
        score_start, score_end = self.network(*inputs)
        score_start = score_start.data.cpu()
        score_end = score_end.data.cpu()

        prediction_starts = []
        prediction_ends = []
        prediction_scores = []
        # 预测答案时最长考虑跨度大小，score_start.size(1)为最大文档长度
        max_span_length = self.args.max_span_length or score_start.size(1)
        for i in range(score_start.size(0)):
            # 向量外积（叉乘），两个向量的叉积与这两个向量组成的坐标平面垂直
            # scores: max_document_length * max_document_length
            scores = torch.ger(score_start[i], score_end[i])

            """ scores.triu_(diagonal=0)返回保留主对角线的上三角矩阵，.tril_(diagonal=k)将主对角线向上第k对角线以上部分置为0, e.g.
                一个4*4的值全为1的矩阵m, m.triu_().tril_(1) = 
                1 1 1 1     1 1 1 1     1 1 0 0
                1 1 1 1 =>  0 1 1 1 =>  0 1 1 0
                1 1 1 1     0 0 1 1     0 0 1 1
                1 1 1 1     0 0 0 1     0 0 0 1
                使用上三角矩阵的目的是使得start的index大于end的index，将主对角线向上第k对角线以上部分置为0是使得span长度不大于k
            """
            scores.triu_().tril_(max_span_length - 1)

            scores = scores.numpy()
            # 变为1维
            scores_flat = scores.flatten()
            if top_n == 1:
                # 返回最大值的索引
                sorted_indices = [numpy.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                # 返回所有答案索引
                sorted_indices = numpy.argsort(-scores_flat)
            else:
                # 返回top_n的答案索引
                unsort_indices = numpy.argpartition(-scores_flat, top_n)[0:top_n]
                sorted_indices = unsort_indices[numpy.argsort(-scores_flat[unsort_indices])]

            """ unravel_index返回被flatten的矩阵中sorted_indices对应在原矩阵的位置，如：
                原矩阵m为3*3，flatten后变为size of 9，此时的indices为[0,1,2...8]，经过numpy.unravel_index([6,7,8], m.shape)
                得到：(array([2, 2, 2]), array([0, 1, 2]))，即在flatten中[6,7,8]对应的元素在原矩阵m中的位置为(2,0),(2,1),(2,2)
            """
            # start_indices, end_indices为一个array[top_n]
            start_indices, end_indices = numpy.unravel_index(sorted_indices, scores.shape)
            prediction_starts.append(start_indices)
            prediction_ends.append(end_indices)
            prediction_scores.append(scores_flat[sorted_indices])
        return prediction_starts, prediction_ends, prediction_scores

    # --------------------------------------------------------------------------
    # 保存模型和检查点相关
    # --------------------------------------------------------------------------

    def save(self, model_path):
        states = copy.copy(self.network.state_dict())
        # 不保存buffer中的fixed_embedding到文件中
        if 'fixed_embeddings' in states:
            states.pop('fixed_embeddings')
        params = {
            'states': states,
            'words': self.words,
            'chars': self.chars,
            'extra_feature_fields': self.extra_feature_fields,
            'args': self.args,
        }
        try:
            torch.save(params, model_path)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, model_checkpoint_path, epoch):
        params = {
            'states': self.network.state_dict(),
            'words': self.words,
            'chars': self.chars,
            'extra_feature_fields': self.extra_feature_fields,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, model_checkpoint_path)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    # --------------------------------------------------------------------------
    # 运行时环境相关
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def gpu_parallelize(self):
        """ 使用数据并行化将模型复制到多个gpu上，它将占用所有gpu visible with CUDA_VISIBLE_DEVICES. """
        self.gpu_parallel = True
        self.network = torch.nn.DataParallel(self.network)


# --------------------------------------------------------------------------
# RnnEncoder
# --------------------------------------------------------------------------

class RnnEncoder(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(RnnEncoder, self).__init__()

        self.args = args
        # args.cnn_param_init = 0

        # words embedding层: 1. embedding的词典大小，2. embedding向量的大小，3. 当输入中遇到padding_idx时，输出0
        self.words_embedding = nn.Embedding(args.words_size, args.embedding_size_w, padding_idx=0)
        self.chars_embedding = nn.Embedding(args.chars_size, args.embedding_size_c, padding_idx=0)

        # char cnn层
        cnn_size = sum(args.kernel_feature_sizes)
        self.char_conv = CharConvModel(
            args.embedding_size_c, args.kernel_sizes, args.kernel_feature_sizes, param_init=args.cnn_param_init
        )
        self.char_linear = CharLinearModel(cnn_size, cnn_size, args.num_layers_c, param_init=args.cnn_param_init)

        # 对齐特征层
        if args.use_align:
            self.first_round_attention_E = AlignFeatureLayer(args.embedding_size_w)
            self.first_round_attention_B = SelfAlignLayer(args.embedding_size_w)

        # re-attention
        if args.use_reattention:
            self.gamma_e = Variable(torch.FloatTensor([self.args.gamma_e]), requires_grad=True).cuda()
            self.gamma_b = Variable(torch.FloatTensor([self.args.gamma_b]), requires_grad=True).cuda()
            self.reattention = ReattentionLayer(args.embedding_size_w)

        # 输入到rnn_document_encoder的向量长度: d_embedding_size + d_embedding_size + extra_features_size + cnn_size
        document_rnn_input_size = args.embedding_size_w + args.num_extra_features
        if args.use_align:
            document_rnn_input_size += args.embedding_size_w
        document_rnn_input_size += cnn_size

        question_rnn_input_size = args.embedding_size_w
        question_rnn_input_size += cnn_size

        logger.info('Document features size: %d+%d+%d+%d' %
                    (args.embedding_size_w, args.embedding_size_w, args.num_extra_features, cnn_size))
        logger.info('Question features size: %d+%d' % (args.embedding_size_w, cnn_size))

        # RNN document encoder
        self.rnn_document_encoder = StackedBiRnnLayer(
            input_size=document_rnn_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers_d,
            dropout_rate=args.rnn_dropout_rate,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
        )

        # RNN question encoder
        self.rnn_question_encoder = StackedBiRnnLayer(
            input_size=question_rnn_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers_q,
            dropout_rate=args.rnn_dropout_rate,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
        )

        # rnn encoders的最后一维大小：(hidden_size * 2 * num_layers)
        document_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            document_hidden_size *= args.num_layers_d
            question_hidden_size *= args.num_layers_q

        # question hiddens merge层，因为每个时间步会输出一个hidden，合并question的hiddens成为一个hidden，用于表示这个问题
        self.self_attention = SelfAttentionLayer(question_hidden_size)

        # 双线性attention层用于预测答案span的start/end
        self.start_attention = BiLinearAttentionLayer(document_hidden_size, question_hidden_size)
        self.end_attention = BiLinearAttentionLayer(document_hidden_size, question_hidden_size)

    def forward(self, f, d_w, d_mask, q_w, q_mask, d_c, q_c):
        """ 输入：
        :param f        额外特征           batch * max_document_length * feature_fields长度
        :param d_w      文档单词indices    batch * max_document_length
        :param d_mask   文档单词mask       batch * max_document_length
        :param q_w      问题单词indices    batch * max_question_length
        :param q_mask   问题单词mask       batch * max_question_length
        :param d_c:     文档字符indices    batch * max_document_length * max_document_word_length
        :param q_c:     问题字符indices    batch * max_question_length * max_question_word_length
        :return scores_start              batch * max_document_length
        :return scores_end                batch * max_document_length
        """
        # --------------------------------------------------------------------------
        # Embedding特征（文档、问题）
        # d_embedding: batch * max_document_length * embedding_size
        d_w_embedding = self.words_embedding(d_w)
        # q_embedding: batch * max_question_length * embedding_size
        q_w_embedding = self.words_embedding(q_w)
        # Dropout on embeddings
        if self.args.embedding_dropout_rate > 0:
            # training: switch between training and evaluation mode
            d_w_embedding = dropout(d_w_embedding, p=self.args.embedding_dropout_rate, training=self.training)
            q_w_embedding = dropout(q_w_embedding, p=self.args.embedding_dropout_rate, training=self.training)

        # 构建文档的d_rnn_input: [d_embedding, align_feature, f]:
        # d_rnn_input: batch * max_document_length * [embedding_size + embedding_size + feature_fields长度]
        d_rnn_inputs = []
        q_rnn_inputs = []

        # --------------------------------------------------------------------------
        # 对齐特征（文档）
        # 使用attention机制对q_embedding每个词向量加权，得到问题的对齐特征
        if self.args.use_align:
            # align_feature: batch * max_document_length * embedding_size
            # e_alpha: batch * max_document_length * max_question_length
            align_feature, e_alpha = self.first_round_attention_E(d_w_embedding, q_w_embedding, q_mask)
            # b_alpha: batch * max_document_length * max_document_length
            align_feature, b_alpha = self.first_round_attention_B(align_feature, d_mask)

            if self.args.use_reattention:
                for i in range(self.args.reattention_round):
                    e_alpha, b_alpha, align_feature = self.reattention(
                        q_w_embedding, align_feature, q_mask, d_mask, e_alpha, b_alpha, self.gamma_e, self.gamma_b
                    )
            # align_feature: batch * max_document_length * embedding_size
            d_rnn_inputs.append(align_feature)

        # --------------------------------------------------------------------------
        # 额外特征（文档）
        if self.args.num_extra_features > 0:
            d_rnn_inputs.append(f)

        # --------------------------------------------------------------------------
        # char cnn特征（文档、问题）
        # d/q_c_embedding: (batch * max_seq_length) * max_seq_word_length * embedding_size
        d_c_embedding = self.chars_embedding(d_c.view(-1, d_c.size(2)))
        q_c_embedding = self.chars_embedding(q_c.view(-1, q_c.size(2)))
        # d/q_conv_output: (batch * max_seq_length) * sum(kernel_feature_size)
        d_conv_output = self.char_conv(d_c_embedding)
        q_conv_output = self.char_conv(q_c_embedding)
        # d/q_linear_output: (batch * max_seq_length) * sum(kernel_feature_size)
        d_linear_output = self.char_linear(d_conv_output)
        q_linear_output = self.char_linear(q_conv_output)
        # d/q_cnn_feature: batch * max_seq_length * sum(kernel_feature_size)
        d_cnn_feature = d_linear_output.view(d_c.size(0), d_c.size(1), -1)
        q_cnn_feature = q_linear_output.view(q_c.size(0), q_c.size(1), -1)
        d_rnn_inputs.append(d_cnn_feature)
        q_rnn_inputs.append(q_cnn_feature)

        # --------------------------------------------------------------------------

        # d_hiddens: batch * max_document_length * (hidden_size * 2 * num_layers)
        d_rnn_input = torch.cat(d_rnn_inputs, 2)
        # logger.info('Actual document features size: %d' % d_rnn_input.size(2))
        d_hiddens = self.rnn_document_encoder(d_rnn_input, d_mask)

        # 因为每个时间步会输出一个hidden，需要合并question的hiddens成为一个hidden，用于表示这个问题
        # q_hiddens: batch * max_question_length * (hidden_size * 2 * num_layers)
        q_rnn_input = torch.cat(q_rnn_inputs, 2)
        # logger.info('Actual question features size: %d' % q_rnn_input.size(2))
        q_hiddens = self.rnn_question_encoder(q_rnn_input, q_mask)

        # 对q_hiddens进行self attention merge到batch * max_question_length
        q_merge_weights = self.self_attention(q_hiddens, q_mask)
        # [batch * 1 * max_question_length].bmm[batch * seq_len * (hidden_size * 2 * num_layers)]
        # = [batch * 1 * (hidden_size * 2 * num_layers)] =>
        # question_hidden: [batch * (hidden_size * 2 * num_layers)]
        q_hidden = q_merge_weights.unsqueeze(1).bmm(q_hiddens).squeeze(1)

        scores_start = self.start_attention(d_hiddens, q_hidden, d_mask)
        scores_end = self.end_attention(d_hiddens, q_hidden, d_mask)

        return scores_start, scores_end


# --------------------------------------------------------------------------
# Char Cnn特征相关模型
# --------------------------------------------------------------------------

class CharConvModel(nn.Module):
    def __init__(self, c_embedding_size, kernel_sizes, kernel_feature_sizes, param_init=0.):
        """
        :param c_embedding_size: 字符embedding_size
        :param kernel_sizes: [1, 2, 3, 4, 5, 6, 7]
        :param kernel_feature_sizes: [50, 100, 150, 200, 200, 200, 200]
        :param param_init:
        """
        super(CharConvModel, self).__init__()

        assert len(kernel_sizes) == len(kernel_feature_sizes)
        self.kernel_sizes = kernel_sizes
        self.kernel_features = kernel_feature_sizes

        for i, (kernel_size, kernel_feature_size) in enumerate(zip(kernel_sizes, kernel_feature_sizes)):
            name = "kernel_size_%d" % i
            # TODO
            kernel_size, kernel_feature_size = (kernel_size, kernel_feature_size)
            setattr(self, name, CharConvLayer(
                c_embedding_size,
                kernel_size=(1, kernel_size),
                kernel_feature_size=kernel_feature_size,
                param_init=param_init
            ))

    def forward(self, c_embedding):
        """
        :param c_embedding: (batch * max_seq_length) * max_seq_word_length * embedding_size
        :return:            (batch * max_seq_length) * sum(kernel_feature_size)
        """
        # (batch * max_seq_length) * max_seq_word_length * 1 * embedding_size
        c_embedding = torch.unsqueeze(c_embedding, 2)
        # (batch * max_seq_length) * c_embedding_size * 1 * max_seq_word_length
        c_embedding = torch.transpose(c_embedding, 1, 3)

        outputs = []
        max_seq_word_length = c_embedding.size(3)
        for i, kernel_size in enumerate(self.kernel_sizes):
            out_width = max_seq_word_length - kernel_size + 1
            output = self.__getattr__("kernel_size_%d" % i)(c_embedding, out_width)
            # output: (batch * max_seq_length) * kernel_feature_size
            outputs.append(output)

        return torch.cat(outputs, 1)


class CharLinearModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers, param_init=0.):
        super(CharLinearModel, self).__init__()

        self.num_layers = num_layers
        for layer in range(num_layers):
            name = "num_layer_%d" % layer
            layer_input_size = input_size if layer == 0 else output_size
            setattr(self, name, CharLinearLayer(layer_input_size, output_size, param_init))

    def forward(self, x):
        output = x
        for layer in range(self.num_layers):
            output = self.__getattr__("num_layer_%d" % layer)(output)
        return output
