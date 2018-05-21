import numpy
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from tqa.reader import utils


class Dictionary(object):
    """ 用于保存word(char)2index和index2word(char) """
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    def __init__(self):
        self.token2index = {self.NULL: 0, self.UNK: 1}
        self.index2token = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.token2index)

    def __iter__(self):
        return iter(self.token2index)

    def __contains__(self, key):
        """ 重写contains方法，根据key的类型 """
        if type(key) == int:
            return key in self.index2token
        elif type(key) == str:
            return utils.normalize(key) in self.token2index

    def __getitem__(self, key):
        """ 重写getitem方法，根据key的类型返回不同value """
        if type(key) == int:
            return self.index2token.get(key, self.UNK)
        if type(key) == str:
            return self.token2index.get(utils.normalize(key), self.token2index.get(self.UNK))

    def __setitem__(self, key, value):
        """ 重写getitem方法，根据key的类型 """
        if type(key) == int and type(value) == str:
            self.index2token[key] = value
        elif type(key) == str and type(value) == int:
            self.token2index[key] = value
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        """ 将单词word加入word2index和index2word """
        token = utils.normalize(token)
        if token not in self.token2index:
            index = len(self.token2index)
            self.token2index[token] = index
            self.index2token[index] = token

    def tokens(self):
        """ 返回词典中的所有单词，除'<NULL>', '<UNK>' """
        words = [word for word in self.token2index.keys() if word not in {'<NULL>', '<UNK>'}]
        return words


class ReaderDataset(Dataset):
    """ 用于获取向量化的example """

    def __init__(self, examples, model, single_answer=False):
        self.model = model
        self.examples = examples
        self.single_answer = single_answer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # 向量化
        return utils.vectorize(self.examples[index], self.model, self.single_answer)

    def lengths(self):
        """ 返回所有examples的文档和问题长度列表，用于在Sampler中排序 """
        return [(len(example['dtext']), len(example['qtext'])) for example in self.examples]


class ReaderSampler(Sampler):
    """ 采样器，根据example中的文档长度和问题长度排序，分batch，打乱batches，返回example indices iter """

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # 根据文档长度和问题长度
        lengths = numpy.array(
            [(-length[0], -length[1], numpy.random.random()) for length in self.lengths],
            dtype=[('dtext_len', numpy.int_), ('qtext_len', numpy.int_), ('random', numpy.float_)]
        )
        # 排序获得example indices，文档长度从大到小 -> 问题长度从大到小 -> 随机（如果前两者都相同）
        indices = numpy.argsort(lengths, order=('dtext_len', 'qtext_len', 'random'))
        # 将排好序的indices分batch
        indices_batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        # 将batches打乱，每个batch里的example按上述排序排序好
        if self.shuffle:
            numpy.random.shuffle(indices_batches)
        return iter([index for indices_batch in indices_batches for index in indices_batch])

    def __len__(self):
        return len(self.lengths)
