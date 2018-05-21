import json
import logging

import unicodedata
from collections import Counter

import torch

from tqa.reader import utils as r_utils
from tqa.reader.data import Dictionary
from tqa.reader.utils import Timer, AverageMeter

logger = logging.getLogger(__name__)


def normalize(text):
    return unicodedata.normalize('NFD', text)


def str2bool(str):
    return str.lower() in ('yes', 'true', 't', '1', 'y', 'True')


def str2int_list(str):
    """ [1, 2, 3, ..., n] """
    return [int(s.strip()) for s in str[1: -1].split(",")]


# ------------------------------------------------------------------------------
# 数据加载相关
# ------------------------------------------------------------------------------

def load_parsed_data(file_path, uncased=False):
    """ 读取文件，每行为一个json，格式为：
    {"id": "ID",
     "text": ["qword1",..."qwordn", "?",...],
     "class": [0],
     "pos": ["NN", "NN", "NN",...],
     "ner": ["O", "O", "O",...]}
    """
    examples = []
    logger.info('Load data files = %s' % file_path)
    with open(file_path, encoding='utf-8') as file:
        for line in file:
            if line.strip() != '':
                examples.append(json.loads(line))
                # examples = [json.loads(line, encoding='utf-8') for line in file]

    if uncased:
        for json_line in examples:
            json_line['text'] = [w.lower() for w in json_line['text']]

    # 过滤没有类别的文本
    examples = [example for example in examples if len(example['class']) > 0]

    return examples


def get_embedded_words(embedding_file):
    words = set()
    with open(embedding_file) as file:
        for line in file:
            word = normalize(line.rstrip().split(' ')[0])
            words.add(word)
    return words


def load_valid_words(args, examples):
    """ 得到train_examples和dev_examples中的所有单词集合、并且单词也出现在embedded_corpus_path中 """

    def _add_words(text):
        for word in text:
            word = normalize(word)
            if valid_words and word not in valid_words:
                continue
            words.add(word)

    # 只使用embedded_corpus_path中训练好的词向量
    if args.only_use_corpus and args.embedded_corpus_path:
        logger.info('Restricting to words in %s' % args.embedded_corpus_path)
        # 将embedded_corpus_path中的词加入集合
        valid_words = r_utils.get_embedded_words(args.embedded_corpus_path)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for example in examples:
        _add_words(example['text'])
    return words


def load_chars(examples):
    """ 得到train_examples和dev_examples中的所有字符集合 """

    def _add_chars(text):
        for char_list in map(lambda word: list(word), text):
            for char in char_list:
                char = normalize(char)
                chars.add(char)

    chars = set()
    for example in examples:
        _add_chars(example['text'])
    return chars


def build_dictionary(args, examples):
    """ 返回文本中的单词和字符集合，封装在Dictionary对象中 """
    words_dictionary = Dictionary()
    chars_dictionary = Dictionary()
    for word in load_valid_words(args, examples):
        words_dictionary.add(word)
    for char in load_chars(examples):
        chars_dictionary.add(char)

    return words_dictionary, chars_dictionary


def get_top_text_words(examples, words, tune_top_k):
    """ 返回所有文本中出现次数top k的单词 """
    word_count = Counter()
    for example in examples:
        for word in example['text']:
            word = normalize(word)
            if word in words:
                word_count.update([word])
    return word_count.most_common(tune_top_k)


# ------------------------------------------------------------------------------
# 特征相关
# ------------------------------------------------------------------------------

def build_extra_feature_fields(args, examples):
    """ 建立特征名称索引，如：
    {pos_NN:0, pos_V:1,... ,ner_O:6, ner_C:7,... ,tf:10}
    """

    def _add(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    # 将每个不同的pos标签，如NN，建立索引：dpos_NN:1
    if args.use_pos:
        for example in examples:
            for w in example['pos']:
                _add('pos_%s' % w)

    # 将每个不同的ner标签，如O，建立索引：dner_O:2
    if args.use_ner:
        for example in examples:
            for w in example['ner']:
                _add('ner_%s' % w)

    if args.use_tf:
        _add('tf')
    return feature_dict


def vectorize(example, model, single_class):
    """ 使用给定的特征（如exact_match、pos等）向量化一个example（embedding、align特征不在此）：
    :param single_class: 是否只使用第一个类别，在训练时为True，测试时为False
    :return t_words_indices: 文档单词的indices: document_length
    :return t_chars_indices: 文档每个单词的字符indices: document_length * max_document_word_length
    :return extra_features: 额外特征矩阵(文档单词个数 * feature_fields长度):
                    feature_fields
            word1   1.0, 1.0, ...
            word2   0.0, 1.0, ...
            ...     ...
            wordn   0.0, 0.0, ...
    :return tag: 类别(或类别 list)
    :return example['id'] 文本id
    """
    args = model.args
    words = model.words
    chars = model.chars
    extra_feature_fields = model.extra_feature_fields

    # 获得文本单词的indices: seq_len，此时还不是最大长度，在batchify中会padding到最大长度
    t_words_indices = torch.LongTensor([words[word] for word in example['text']])

    # char_cnn特征：seq_len * seq_word_length，此时还不是最大长度，在batchify中会padding到最大长度
    t_chars_indices = [[chars[char] for char in word] for word in example['text']]

    # 根据 文本长度 和 feature_fields中的特征向量 确定向量shape：文本长度 * feature_fields长度
    if len(extra_feature_fields) > 0:
        extra_features = torch.zeros(len(example['text']), len(extra_feature_fields))
    else:
        extra_features = None

    # pos特征
    if args.use_pos:
        for i, pos in enumerate(example['pos']):
            field = 'pos_%s' % pos
            if field in extra_feature_fields:
                extra_features[i][extra_feature_fields[field]] = 1.0

    # ner特征
    if args.use_ner:
        for i, ner in enumerate(example['ner']):
            field = 'ner_%s' % ner
            if field in extra_feature_fields:
                extra_features[i][extra_feature_fields[field]] = 1.0

    # tf特征（归一化），
    if args.use_tf:
        counter = Counter([word.lower() for word in example['text']])
        l = len(example['text'])
        for i, word in enumerate(example['text']):
            extra_features[i][extra_feature_fields['tf']] = counter[word.lower()] * 1.0 / l

    # 没有类别，则返回without target
    if 'class' not in example or len(example['class']) == 0:
        return t_words_indices, t_chars_indices, extra_features, example['id']

    # 如果single_answer，则返回第一个答案在dspan中开始和结束的index，否则返回indices list
    if single_class:
        assert (len(example['class']) > 0)
        tag = torch.LongTensor(1).fill_(example['class'][0])
    else:
        tag = example['class']

    return t_words_indices, t_chars_indices, extra_features, tag, example['id']


# ------------------------------------------------------------------------------
# batch相关
# ------------------------------------------------------------------------------

def batchify(batch):
    """ 将一个batch中的examples经过一些处理，人为合成为若干tensor。batch中每个example的格式：
    0. t_words_indices: 文档单词的indices: text_length
    1. t_chars_indices: 文档每个单词的字符indices: text_length * word_length
    2. extra_features: 额外特征矩阵(文档单词个数 * feature_fields长度):
                    feature_fields
            word1   1.0, 1.0, ...
            word2   0.0, 1.0, ...
            ...     ...
            wordn   0.0, 0.0, ...
    -2. tag: 类别(或类别 list)
    -1. example['id']: 文本id

    :return f:      batch * max_text_length * feature_fields长度
    :return d_w:    batch * max_text_length
    :return d_mask: batch * max_text_length
    :return d_c:    batch * max_text_length * max_text_word_length
    :return c:      LongTensor(batch) 或 list[batch * 类别个数]
    :return ids:    list[batch]
    """

    # 下面的4个变量比传入多了一维，按第一维stack，变成batch * 传入维数，并且将文档单词长度padding到max_text_length
    ids = [example[-1] for example in batch]
    t_words_indices = [example[0] for example in batch]
    t_chars_indices = [example[1] for example in batch]
    extra_features = [example[2] for example in batch]

    # 每个text_indices中不够max_text_length的地方用0填充，mask置为1
    max_text_length = max([text.size(0) for text in t_words_indices])
    # t_w、t_mask: batch * max_text_length
    t_w = torch.LongTensor(len(t_words_indices), max_text_length).zero_()
    t_mask = torch.ByteTensor(len(t_words_indices), max_text_length).fill_(1)

    # f: batch * max_text_length * feature_fields长度
    if extra_features[0] is None:
        f = None
    else:
        f = torch.zeros(len(t_words_indices), max_text_length, extra_features[0].size(1))

    # 逐一将documents_indices中每个document_indices复制到d中，并设置对应mask为0
    for i, text_indices in enumerate(t_words_indices):
        t_w[i, :text_indices.size(0)].copy_(text_indices)
        t_mask[i, :text_indices.size(0)].fill_(0)
        if f is not None:
            f[i, :text_indices.size(0)].copy_(extra_features[i])

    # ------------------------------------------------------------------------------
    # Char CNN相关
    # batch的seq中单词的最小长度，因为经过卷积后单词的长度变为max_seq_word_length - kernel_size + 1
    min_words_length = 7
    # batch的文档中单词的最大长度
    t_max_words_length = max([max([len(word) for word in text]) for text in t_chars_indices])
    t_max_words_length = t_max_words_length if t_max_words_length >= min_words_length else min_words_length

    t_c = torch.LongTensor(len(t_words_indices), max_text_length, t_max_words_length).zero_()
    for i, _ in enumerate(t_chars_indices):
        for j, chars_indices in enumerate(t_chars_indices[i]):
            t_c[i][j, : len(chars_indices)].copy_(torch.LongTensor(chars_indices))

    # ------------------------------------------------------------------------------

    # example中可能没有类别
    if len(batch[0]) == 4:
        return t_w, f, t_mask, ids
    # example中有类别
    elif len(batch[0]) == 5:
        # 只有一个类别时，class为LongTensor(1) => c: LongTensor(batch)
        if torch.is_tensor(batch[0][-2]):
            c = torch.cat([example[-2] for example in batch])
        # 有多个答案时，class为list[类别个数] => c: list[batch * 答案个数]
        else:
            c = [example[-2] for example in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return f, t_w, t_mask, t_c, c, ids


# ------------------------------------------------------------------------------
# 评估相关
# ------------------------------------------------------------------------------

def validate(data_loader, model, train_states, type):
    """ 每个example格式：
    -2. 类别     c: LongTensor(batch) 或 list[batch * 答案个数]
    """
    evaluation_time = Timer()
    exact_match = AverageMeter()

    # Make predictions
    examples_count = 0
    for examples_in_batch in data_loader:
        batch_size = examples_in_batch[0].size(0)
        # prediction_start, prediction_end：list[batch]
        prediction_scores, prediction_class = model.predict(examples_in_batch)

        target = examples_in_batch[-2]

        accuracy = evaluate(prediction_class, target)
        exact_match.update(accuracy, batch_size)

        # 如果是计算train_dataset的准确度，只计算前10000个
        examples_count += batch_size
        if type == 'train' and examples_count >= 1e4:
            break

    logger.info('%s validation: Epoch = %d | accuracy = %.2f | examples = %d | ' %
                (type, train_states['epoch'], exact_match.average, examples_count) +
                'validation time = %.2f (s)' % evaluation_time.total_time())

    return {'exact_match': exact_match.average}


def evaluate(prediction_class, target):
    """ 计算accuracy
    :param prediction_class,    list[batch]
    :param target,              LongTensor(batch) 或 list[batch * 类别个数]
                                => list[batch * list[long]] 或 list[batch * 答案个数]
    """

    # 将1维tensors转换为list[list[long]]
    if torch.is_tensor(target):
        target = [[t] for t in target]

    batch_size = len(prediction_class)
    accuracy = AverageMeter()
    for i in range(batch_size):
        # 匹配start
        if prediction_class[i] in target[i]:
            accuracy.update(1)
        else:
            accuracy.update(0)

    return accuracy.average * 100
