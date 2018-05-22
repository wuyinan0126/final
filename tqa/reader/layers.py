import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence


class CharLinearLayer(nn.Module):
    def __init__(self, input_size, output_size, param_init=0.):
        super(CharLinearLayer, self).__init__()
        self.linear_g = nn.Linear(input_size, output_size)
        self.linear_t = nn.Linear(input_size, output_size)
        self.bias = -2.0

        if param_init > 0.:
            self.linear_g.weight.data.uniform_(-param_init, param_init)
            self.linear_g.bias.data.uniform_(-param_init, param_init)
            self.linear_t.weight.data.uniform_(-param_init, param_init)
            self.linear_t.bias.data.uniform_(-param_init, param_init)

    def forward(self, input):
        """
        G = relu(x, Wg)
        T = sigmoid(x, Wt)
                                   |x, T == 0
        y = G * T + x * (1. - T) = |
                                   |G, T == 1
        """
        g = F.relu(self.linear_g(input))
        t = F.sigmoid(self.linear_t(input) + self.bias)
        output = t * g + (1. - t) * input

        return output


class CharConvLayer(nn.Module):
    """ 将一行文字看成是max_seq_length * max_seq_word_length * c_embedding_size的图片
        Conv中卷积核大小为kernel_size(正方形)，共有out_channels个卷积核
        Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(0,0))
    """

    def __init__(self, embedding_size_c, kernel_size, kernel_feature_size, param_init=0.):
        """
        :param kernel_size:         (1, one of [1, 2, 3, 4, 5, 6, 7])
        :param kernel_feature_size: one of [50, 100, 150, 200, 200, 200, 200]
        """
        super(CharConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=embedding_size_c, out_channels=kernel_feature_size, kernel_size=kernel_size)

        if param_init > 0.:
            self.conv.weight.data.uniform_(-param_init, param_init)
            self.conv.bias.data.uniform_(-param_init, param_init)

    def forward(self, input, out_width):
        """
        Input: [batch * in_channels * in_height * in_width]
        Output: [batch * out_channels * out_height * out_width]

        :param input:       (batch * max_seq_length) * c_embedding_size * 1 * max_seq_word_length
        :param out_width:   max_seq_word_length - kernel_size + 1
        :return:            (batch * max_seq_length) * kernel_feature_size
        """
        # output: (batch * max_seq_length) * kernel_feature_size * 1 * out_width
        output = F.tanh(self.conv(input))
        # output: (batch * max_seq_length) * kernel_feature_size * 1 * 1
        output = F.max_pool2d(output, kernel_size=[1, out_width], stride=[1, 1])
        # output: (batch * max_seq_length) * kernel_feature_size * 1
        output = torch.squeeze(output, dim=3)
        # return: (batch * max_seq_length) * kernel_feature_size
        return torch.squeeze(output, dim=2)


class AlignFeatureLayer(nn.Module):
    """ 将问题Y的每个词对齐到文档X的每个词上，使用attention机制对问题中每个词向量加权
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size):
        super(AlignFeatureLayer, self).__init__()
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, d_embedding, q_embedding, q_mask):
        """
        :param d_embedding:     batch * max_document_length * embedding_size
        :param q_embedding:     batch * max_question_length * embedding_size
        :param q_mask:          batch * max_question_length (1 for padding, 0 for true)
        :return: align_feature: batch * max_document_length * embedding_size
        """

        # 用线性和非线性函数project embedding vectors
        # 先转成(batch * max_document_length) * embedding_size输入liner层，得到相同维度的输出，在转回原始维度
        # 可以直接self.linear(d_embedding)
        d_projection = self.linear(d_embedding.view(-1, d_embedding.size(2))).view(d_embedding.size())
        d_projection = F.relu(d_projection)
        q_projection = self.linear(q_embedding.view(-1, q_embedding.size(2))).view(q_embedding.size())
        q_projection = F.relu(q_projection)

        # 计算scores：[batch * max_document_length * embedding_size].bmm[batch * embedding_size * max_question_length]
        # scores: [batch * max_document_length * max_question_length]
        scores = d_projection.bmm(q_projection.transpose(2, 1))

        # unsqueeze(1)在1位置加入一维，从batch * max_question_length => batch * 1 * max_question_length
        # expand()将刚加入的一维复制数据扩展成max_document_length => batch * max_document_length * max_question_length
        q_mask = q_mask.unsqueeze(1).expand(scores.size())
        # masked_fill_: 保留scores中对应mask为0的地方，为1的地方赋为-float('inf')
        scores.data.masked_fill_(q_mask.data, -float('inf'))

        # 将scores转换为(batch * max_document_length) * max_question_length，对max_question_length维使用softmax
        # 可以直接F.softmax(scores, 2)
        # alpha: [batch * max_document_length * max_question_length]
        alpha = F.softmax(scores.view(-1, q_embedding.size(1)), dim=1) \
            .view(-1, d_embedding.size(1), q_embedding.size(1))

        # 使用attention机制对q_embedding每个词向量加权
        # [batch * max_document_length * max_question_length].bmm[batch * max_question_length * embedding_size]
        # align_feature: batch * max_document_length * embedding_size
        align_feature = alpha.bmm(q_embedding)
        return align_feature, alpha


class StackedBiRnnLayer(nn.Module):
    """ 堆积双向RNN层
    需要一层一层构建，因为可能需要每一层的output，如果直接构建多层，获得的output为最后一层的，因此，
    最终获得的经过concat的output：[batch * seq_len * (hidden_size * 2 * num_layers)]
    在forword时，先对batch中的seq按照有效长度从大到小排序，封装成PackedSequence，这样能加速训练，最后需要还原原始顺序
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM, concat_layers=False):
        super(StackedBiRnnLayer, self).__init__()

        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        # ModuleList(): Holds submodules in a list
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            # 第1层输入大小为input_size，第2层后为hidden_size * 2（因为双向）
            input_size = input_size if i == 0 else hidden_size * 2
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=True))

    def forward(self, x, x_mask):
        """ Encode序列x（文档或问题）
        :param x:
            对于文档: batch * max_document_length * [embedding_size + embedding_size + feature_fields_size]
            对于问题: batch * max_question_length * embedding_size
        :param x_mask:
            对于文档: batch * max_document_length
            对于问题: batch * max_question_length
        :return output: batch * seq_len * (hidden_size * 2 * num_layers)
        """

        # 计算batch里每个序列的有效长度lengths: [batch_size]
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        # sorted_indices是针对未排序之前的序列，从大到小排序获得排好的序列
        _, sorted_indices = torch.sort(lengths, dim=0, descending=True)
        # unsort_indices是针对排好序的序列，找回未排序之前的序列
        _, unsort_indices = torch.sort(sorted_indices, dim=0)

        sorted_lengths = list(lengths[sorted_indices])
        sorted_indices = Variable(sorted_indices)
        unsort_indices = Variable(unsort_indices)

        # 对x按有效长度从大到小排序
        x = x.index_select(dim=0, index=sorted_indices)

        """ pack_padded_sequence(input, lengths):
        input: is of size T x B x * where T is the length of the longest sequence, B is the batch size
        lengths: list of sequences lengths of each batch element
        The sequences should be sorted by length in a decreasing order
        返回PackedSequence对象
        """
        rnn_input = pack_padded_sequence(x.transpose(0, 1), sorted_lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            # dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
                # 构建PackedSequence对象, batch_sizes: the batch size at each sequence step，每个时间步对batch_size
                rnn_input = PackedSequence(dropout_input, rnn_input.batch_sizes)
            # rnn output: PackedSequence
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, output in enumerate(outputs[1:], 1):
            # pad_packed_sequence(PackedSequence对象): 返回2元组：padded sequence: T x B x * 和 lengths
            # outputs[i]: [seq_len * batch * (hidden_size * 2)]
            outputs[i] = pad_packed_sequence(output)[0]

        # 连接每一层的隐含层状态或直接使用最后一层的输出
        if self.concat_layers:
            # output: [seq_len * batch * (hidden_size * 2 * num_layers)]
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # output: [batch * seq_len * (hidden_size * 2 * num_layers)]
        output = output.transpose(0, 1)
        # 恢复原始顺序
        output = output.index_select(0, unsort_indices)

        # 补0恢复原始seq_len
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(
                output.size(0), x_mask.size(1) - output.size(1), output.size(2)
            ).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)

        # 内存连续则指向同样的内存返回，不连续，内存复制
        return output.contiguous()


class SelfAttentionLayer(nn.Module):
    """ self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, question_hidden_size):
        super(SelfAttentionLayer, self).__init__()
        self.linear = nn.Linear(question_hidden_size, 1)

    def forward(self, q_hiddens, q_mask):
        """ 将(hidden_size * 2 * num_layers)线性变换到1，再softmax
        :param q_hiddens:   batch * max_question_length * (hidden_size * 2 * num_layers)
        :param q_mask:      batch * max_question_length
        :return alpha:      batch * max_question_length
        """
        # 先转成(batch * seq_len) * (hidden_size * 2 * num_layers)输入liner层，(batch * seq_len) * 1，在转成batch * seq_len
        scores = self.linear(q_hiddens.view(-1, q_hiddens.size(-1))).view(q_hiddens.size(0), q_hiddens.size(1))
        scores.data.masked_fill_(q_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha


class BiLinearAttentionLayer(nn.Module):
    """ 双线性 attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    """

    def __init__(self, document_hidden_size, question_hidden_size):
        super(BiLinearAttentionLayer, self).__init__()
        self.linear = nn.Linear(question_hidden_size, document_hidden_size)

    def forward(self, d_hiddens, q_hidden, d_mask):
        """
        :param d_hiddens:   batch * max_document_length * (hidden_size * 2 * num_layers)
        :param q_hidden:    batch * (hidden_size * 2 * num_layers)
        :param d_mask:      batch * max_document_length (1 for padding, 0 for true)
        :return alpha       batch * max_document_length
        """
        # Wy: batch * (hidden_size * 2 * num_layers)
        Wy = self.linear(q_hidden)
        # [batch * max_document_length * (hidden_size * 2 * num_layers)].bmm[batch * (hidden_size * 2 * num_layers) * 1]
        # = [batch * max_document_length * 1] =>
        # xWy: batch * max_document_length
        xWy = d_hiddens.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(d_mask.data, -float('inf'))

        if self.training:
            # 如果是训练则输出log-softmax for NLLLoss
            alpha = F.log_softmax(xWy, dim=1)
        else:
            # 如果是测试则输出0-1的概率
            alpha = F.softmax(xWy, dim=1)
        return alpha


class SelfAlignLayer(nn.Module):
    def __init__(self, input_size):
        super(SelfAlignLayer, self).__init__()
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, d_embedding, d_mask):
        """
        :param d_embedding:     batch * max_document_length * embedding_size
        :param q_embedding:     batch * max_question_length * embedding_size
        :param d_mask:          batch * max_question_length (1 for padding, 0 for true)
        :return: align_feature: batch * max_document_length * embedding_size
        """

        max_document_length = d_embedding.size(1)

        # 用线性和非线性函数project embedding vectors
        # 先转成(batch * max_document_length) * embedding_size输入liner层，得到相同维度的输出，在转回原始维度
        # 可以直接self.linear(d_embedding)
        d_projection = self.linear(d_embedding.view(-1, d_embedding.size(2))).view(d_embedding.size())
        d_projection = F.relu(d_projection)

        # 计算scores：[batch * max_document_length * embedding_size].bmm[batch * embedding_size * max_document_length]
        # scores: [batch * max_document_length * max_document_length]
        scores = d_projection.bmm(d_projection.transpose(2, 1))

        scores = scores * Variable(
            (-1 * (torch.eye(max_document_length) - 1)).unsqueeze(0).expand(scores.size()), requires_grad=False).cuda()
        print(scores.data)

        # unsqueeze(1)在1位置加入一维，从batch * max_document_length => batch * 1 * max_document_length
        # expand()将刚加入的一维复制数据扩展成max_document_length => batch * max_document_length * max_document_length
        d_mask = d_mask.unsqueeze(1).expand(scores.size())
        # masked_fill_: 保留scores中对应mask为0的地方，为1的地方赋为-float('inf')
        scores.data.masked_fill_(d_mask.data, -float('inf'))

        # 将scores转换为(batch * max_document_length) * max_document_length，对max_document_length维使用softmax
        # 可以直接F.softmax(scores, 2)
        # alpha: [batch * max_document_length * max_document_length]
        alpha = F.softmax(scores.view(-1, d_embedding.size(1)), dim=1) \
            .view(-1, d_embedding.size(1), d_embedding.size(1))

        # 使用attention机制对d_embedding每个词向量加权
        # [batch * max_document_length * max_document_length].bmm[batch * max_document_length * embedding_size]
        # align_feature: batch * max_document_length * embedding_size
        align_feature = alpha.bmm(d_embedding)
        return align_feature, alpha


class ReattentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(ReattentionLayer, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, q_hiddens, d_hiddens, q_mask, d_mask, e_alpha, b_alpha, gamma):
        """
        :param q_hiddens:   batch * max_question_length * (hidden_size * 2 * num_layers)
        :param d_hiddens:   batch * max_document_length * (hidden_size * 2 * num_layers)
        :param q_mask:      batch * max_question_length (1 for padding, 0 for true)
        :param d_mask:      batch * max_document_length (1 for padding, 0 for true)
        :param e_alpha:     batch * max_document_length * max_question_length
        :param b_alpha:     batch * max_document_length * max_document_length
        :param gamma:
        :return alpha       batch * max_document_length
        """
        max_document_length = d_hiddens.size(1)
        max_question_length = q_hiddens.size(1)

        # E_tt: batch * max_document_length * max_question_length
        E_tt = (F.softmax(e_alpha.transpose(1, 2), dim=2)
                .bmm(F.softmax(b_alpha.transpose(1, 2), dim=1))).transpose(1, 2)

        # d_projection: batch * max_document_length * (hidden_size * 2 * num_layers)
        d_projection = self.linear(d_hiddens.view(-1, d_hiddens.size(2))).view(d_hiddens.size())
        d_projection = F.relu(d_projection)
        # q_projection: batch * max_question_length * (hidden_size * 2 * num_layers)
        q_projection = self.linear(q_hiddens.view(-1, q_hiddens.size(2))).view(q_hiddens.size())
        q_projection = F.relu(q_projection)

        # scores: batch * max_document_length * max_question_length
        scores = d_projection.bmm(q_projection.transpose(2, 1))
        # q_mask: batch * max_question_length => batch * max_document_length * max_question_length
        q_mask = q_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(q_mask.data, -float('inf'))

        # E_f: batch * max_document_length * max_question_length
        E_f = F.softmax(scores.view(-1, max_question_length), dim=1) \
            .view(-1, max_document_length, max_question_length)

        # E_t: batch * max_document_length * max_question_length
        E_t = E_f + gamma * E_tt

        # H_t: [batch * max_document_length * max_question_length].bmm[batch * max_question_length * (hidden_size * 2 * num_layers)]
        # => batch * max_document_length * (hidden_size * 2 * num_layers)
        H_t = E_f.bmm(q_hiddens)

        # B_tt: batch * max_document_length * max_document_length
        B_tt = (F.softmax(b_alpha.transpose(1, 2), dim=2)
                .bmm(F.softmax(b_alpha.transpose(1, 2), dim=1))).transpose(1, 2)

        # H_t_projection: batch * max_document_length * (hidden_size * 2 * num_layers)
        H_t_projection = self.linear(H_t.view(-1, H_t.size(2))).view(H_t.size())
        H_t_projection = F.relu(H_t_projection)

        # scores: batch * max_document_length * max_document_length
        scores = H_t_projection.bmm(H_t_projection.transpose(2, 1))
        # d_mask: batch * max_document_length => batch * max_document_length * max_document_length
        d_mask = d_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(d_mask.data, -float('inf'))

        # B_f: batch * max_document_length * max_document_length
        B_f = F.softmax(scores.view(-1, H_t.size(1)), dim=1).view(-1, H_t.size(1), H_t.size(1))

        # B_t: batch * max_document_length * max_document_length
        B_t = (B_f + gamma * B_tt) * Variable(
            (-1 * (torch.eye(max_document_length) - 1)).unsqueeze(0).expand(B_tt.size()), requires_grad=False).cuda()

        # align_ht: batch * max_document_length * (hidden_size * 2 * num_layers)
        align_ht = B_t.bmm(H_t)

        return E_t, B_t, align_ht
