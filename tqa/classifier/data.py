import numpy
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from tqa.classifier import utils


class ClassifierDataset(Dataset):
    """ 用于获取向量化的example """

    def __init__(self, examples, model, single_class):
        self.model = model
        self.examples = examples
        self.single_class = single_class

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # 向量化
        return utils.vectorize(self.examples[index], self.model, self.single_class)

    def lengths(self):
        """ 返回所有examples的文本长度列表，用于在Sampler中排序 """
        return [len(example['text']) for example in self.examples]


class ClassifierSampler(Sampler):
    """ 采样器，根据example中的文本长度排序，分batch，打乱batches，返回example indices iter """

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # 根据文档长度和问题长度
        lengths = numpy.array(
            [(-length, numpy.random.random()) for length in self.lengths],
            dtype=[('text_len', numpy.int_), ('random', numpy.float_)]
        )
        # 排序获得example indices，文本长度从大到小 -> 随机（如果文本长度都相同）
        indices = numpy.argsort(lengths, order=('text_len', 'random'))
        # 将排好序的indices分batch
        indices_batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        # 将batches打乱，每个batch里的example按上述排序排序好
        if self.shuffle:
            numpy.random.shuffle(indices_batches)
        return iter([index for indices_batch in indices_batches for index in indices_batch])

    def __len__(self):
        return len(self.lengths)
