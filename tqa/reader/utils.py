import argparse
import json
import logging
import unicodedata
from collections import Counter

import numpy
import time
import torch

from tqa.classifier.models import ClassifierModel
from tqa.reader.models import ReaderModel
from tqa.reader.data import Dictionary

logger = logging.getLogger(__name__)

# 在模型训练和优化过程中关心的变量（可用新值替换）
MODEL_OPTIMIZER = {
    'fix_embeddings', 'optimizer', 'learning_rate', 'momentum', 'weight_decay',
    'rnn_padding', 'dropout_rnn', 'dropout_rnn_output', 'dropout_emb',
    'max_len', 'grad_clipping', 'tune_partial'
}


# ------------------------------------------------------------------------------
# 环境准备相关
# ------------------------------------------------------------------------------

def set_environment(use_cuda, gpu_device, random_seed):
    if use_cuda:
        torch.cuda.set_device(gpu_device)

    numpy.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed(random_seed)

    logger.info('Prepare completed')


def str2bool(str):
    return str.lower() in ('yes', 'true', 't', '1', 'y', 'True')


def str2int_list(str):
    """ [1, 2, 3, ..., n] """
    return [int(s.strip()) for s in str[1: -1].split(",")]


def str2str_list(str):
    """ ['1', '2', '3', ..., 'n'] """
    return [s.strip() for s in str[1: -1].split(",")]


# ------------------------------------------------------------------------------
# 数据加载相关
# ------------------------------------------------------------------------------

def normalize(text):
    return unicodedata.normalize('NFD', text)


def load_parsed_data(file_path, uncased_question=False, uncased_document=False, skip_no_answer=False):
    """ 读取parsed SQuAD文件，每行为一个json，格式为：
    {"id": "ID",
     "qtext": ["qword1",..."qwordn", "?",...],
     "dtext": ["dword1",..."dwordn",...],
     "dspan": [[0, 8], [9, 16], [17, 24],...], # 文档中每个词的span
     "aspan": [[1, 2]], # 答案在dspan中的span，如左边表示答案为dspan[[0, 8], [9, 16]]
     "qlemma": ["qword1",..."qwordn", "?",...],
     "dlemma": ["dword1",..."dwordn",...],
     "dpos": ["NN", "NN", "NN",...],
     "dner": ["O", "O", "O",...]}
    """
    logger.info('Load data files = %s' % file_path)
    with open(file_path) as file:
        examples = [json.loads(line) for line in file]

    if uncased_question or uncased_document:
        for json_line in examples:
            if uncased_question:
                json_line['qtext'] = [w.lower() for w in json_line['qtext']]
            if uncased_document:
                json_line['dtext'] = [w.lower() for w in json_line['dtext']]

    if skip_no_answer:
        examples = [example for example in examples if len(example['aspan']) > 0]

    return examples


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
        valid_words = get_embedded_words(args.embedded_corpus_path)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for example in examples:
        _add_words(example['qtext'])
        _add_words(example['dtext'])
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
        _add_chars(example['qtext'])
        _add_chars(example['dtext'])
    return chars


def get_embedded_words(embedding_file):
    words = set()
    with open(embedding_file) as file:
        for line in file:
            word = normalize(line.rstrip().split(' ')[0])
            words.add(word)
    return words


# ------------------------------------------------------------------------------
# 模型相关
# ------------------------------------------------------------------------------

def load_checkpoint(model_path, model_type='reader'):
    logger.info('Loading model %s' % model_path)
    # 在cpu上加载预先训练好的gpu模型，把所有的张量加载到cpu中
    saved_params = torch.load(model_path, map_location=lambda storage, loc: storage)
    # 把所有的张量加载到GPU 0中
    # saved_params = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))

    args = saved_params['args']
    words = saved_params['words']
    chars = saved_params['chars']
    extra_feature_fields = saved_params['extra_feature_fields']
    states = saved_params['states']
    epoch = saved_params['epoch']
    optimizer = saved_params['optimizer']

    model = ReaderModel(args, words, chars, extra_feature_fields, states) if model_type == 'reader' \
        else ClassifierModel(args, words, chars, extra_feature_fields, states)
    model.init_optimizer()

    return model, epoch


def load_model(model_path, new_args=None, model_type='reader'):
    global MODEL_OPTIMIZER

    logger.info('Loading model %s' % model_path)
    saved_params = torch.load(model_path, map_location=lambda storage, loc: storage)

    old_args = saved_params['args']
    words = saved_params['words']
    chars = saved_params['chars']
    extra_feature_fields = saved_params['extra_feature_fields']
    states = saved_params['states']

    args = old_args
    if new_args:
        # 遍历旧的和新的args，只替换在MODEL_OPTIMIZER中的变量值
        old_args, new_args = vars(old_args), vars(new_args)
        for k in old_args.keys():
            if k in new_args and old_args[k] != new_args[k]:
                if k in MODEL_OPTIMIZER:
                    logger.info('Overriding saved %s: %s --> %s' % (k, old_args[k], new_args[k]))
                    old_args[k] = new_args[k]
                else:
                    logger.info('Keeping saved %s: %s' % (k, old_args[k]))
        args = argparse.Namespace(**old_args)

    return ReaderModel(args, words, chars, extra_feature_fields, states) if model_type == 'reader' \
        else ClassifierModel(args, words, chars, extra_feature_fields, states)


def init_model_from_scratch(args, train_examples, dev_examples, model_type='reader'):
    """ 建立模型：建立训练集特征名称索引、建立Dictionary对象、初始化模型、加载embeddings """

    logger.info('-' * 100)
    logger.info('Generate features field indices')
    # 建立训练集特征名称索引
    if model_type == 'reader':
        extra_feature_fields = build_extra_feature_fields(args, train_examples)
    elif model_type == 'classifier':
        from tqa.classifier import utils as c_utils
        extra_feature_fields = c_utils.build_extra_feature_fields(args, train_examples)
    else:
        raise RuntimeError('Unsupported Model Type: %s' % model_type)

    logger.info('Num extra features = %d' % len(extra_feature_fields))
    logger.info(extra_feature_fields)

    # 获得问题和文档中的单词和字符集合，封装在Dictionary对象中
    logger.info('-' * 100)
    logger.info('Build words & chars dictionary')
    if model_type == 'reader':
        words, chars = build_dictionary(args, train_examples + dev_examples)
    elif model_type == 'classifier':
        from tqa.classifier import utils as c_utils
        words, chars = c_utils.build_dictionary(args, train_examples + dev_examples)
    else:
        raise RuntimeError('Unsupported Model Type: %s' % model_type)

    logger.info('Num distinct words = %d' % len(words))
    logger.info('Num distinct chars = %d' % len(chars))

    model = ReaderModel(args, words, chars, extra_feature_fields) if model_type == 'reader' else \
        ClassifierModel(args, words, chars, extra_feature_fields)

    # 使用glove对dictionary中的单词进行embedding
    if args.embedded_corpus_path:
        model.load_embeddings(words.tokens(), args.embedded_corpus_path)

    return model


# ------------------------------------------------------------------------------
# 词典相关
# ------------------------------------------------------------------------------

def build_dictionary(args, examples):
    """ 返回问题和文档中的单词和字符集合，封装在Dictionary对象中 """
    words_dictionary = Dictionary()
    chars_dictionary = Dictionary()
    for word in load_valid_words(args, examples):
        words_dictionary.add(word)
    for char in load_chars(examples):
        chars_dictionary.add(char)

    return words_dictionary, chars_dictionary


def get_top_question_words(examples, words, tune_top_k):
    """ 返回所有问题中出现次数top k的单词 """
    word_count = Counter()
    for example in examples:
        for word in example['qtext']:
            word = normalize(word)
            if word in words:
                word_count.update([word])
    return word_count.most_common(tune_top_k)


# ------------------------------------------------------------------------------
# 特征相关
# ------------------------------------------------------------------------------

def build_extra_feature_fields(args, examples):
    """ 建立特征名称索引，如：
    {exact_match_origin:0, exact_match_uncased:1, exact_match_lemma:2
     dpos_NN:3, dpos_V:4,... ,dner_O:6, dner_C:7,... ,tf:10}
    """

    def _add(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    if args.use_exact_match:
        _add('exact_match_origin')
        if args.language == 'en':
            _add('exact_match_uncased')
        if args.use_lemma:
            _add('exact_match_lemma')

    # 将每个不同的pos标签，如NN，建立索引：dpos_NN:1
    if args.use_pos:
        for example in examples:
            for w in example['dpos']:
                _add('dpos_%s' % w)

    # 将每个不同的ner标签，如O，建立索引：dner_O:2
    if args.use_ner:
        for example in examples:
            for w in example['dner']:
                _add('dner_%s' % w)

    if args.use_tf:
        _add('tf')
    return feature_dict


def vectorize(example, model, single_answer=False):
    """ 使用给定的特征（如exact_match、pos等）向量化一个example（embedding、align特征不在此）：
    :param single_answer: 是否只使用第一个答案，在训练时为True，测试时为False
    :return d_words_indices: 文档单词的indices: document_length
    :return q_words_indices: 问题单词的indices: question_length
    :return d_chars_indices: 文档每个单词的字符indices: document_length * max_document_word_length
    :return q_chars_indices: 问题每个单词的字符indices: question_length * max_question_word_length
    :return extra_features: 额外特征矩阵(文档单词个数 * feature_fields长度):
                    feature_fields
            word1   1.0, 1.0, ...
            word2   0.0, 1.0, ...
            ...     ...
            wordn   0.0, 0.0, ...
    :return start: 答案在dspan中开始的index(或indices list)
    :return end: 答案在dspan中结束的index(或indices list)
    :return example['id'] 问题id
    """
    args = model.args
    words = model.words
    chars = model.chars
    extra_feature_fields = model.extra_feature_fields

    # 获得文档和问题单词的indices: seq_len，此时还不是最大长度，在batchify中会padding到最大长度
    d_words_indices = torch.LongTensor([words[word] for word in example['dtext']])
    q_words_indices = torch.LongTensor([words[word] for word in example['qtext']])

    # char_cnn特征：seq_len * seq_word_length，此时还不是最大长度，在batchify中会padding到最大长度
    d_chars_indices = [[chars[char] for char in word] for word in example['dtext']]
    q_chars_indices = [[chars[char] for char in word] for word in example['qtext']]

    # 根据 文档长度 和 feature_fields中的特征向量 确定最终向量shape：文档长度 * feature_fields长度
    if len(extra_feature_fields) > 0:
        extra_features = torch.zeros(len(example['dtext']), len(extra_feature_fields))
    else:
        extra_features = None

    # exact_match特征
    if args.use_exact_match:
        qtext_origin = {word for word in example['qtext']}
        qtext_uncased = {word.lower() for word in example['qtext']} \
            if 'exact_match_uncased' in extra_feature_fields else None
        qlemma = {lemma for lemma in example['qlemma']} if args.use_lemma else None
        for i in range(len(example['dtext'])):
            # 文档中词[i] 原型 出现在问题中
            if example['dtext'][i] in qtext_origin:
                extra_features[i][extra_feature_fields['exact_match_origin']] = 1.0
            # 文档中词[i] 小写 出现在问题中
            if qtext_uncased and example['dtext'][i].lower() in qtext_uncased:
                extra_features[i][extra_feature_fields['exact_match_uncased']] = 1.0
            # 文档中词[i] 词元 出现在问题中
            if qlemma and example['dlemma'][i] in qlemma:
                extra_features[i][extra_feature_fields['exact_match_lemma']] = 1.0

    # pos特征
    if args.use_pos:
        for i, pos in enumerate(example['dpos']):
            field = 'dpos_%s' % pos
            if field in extra_feature_fields:
                extra_features[i][extra_feature_fields[field]] = 1.0

    # ner特征
    if args.use_ner:
        for i, ner in enumerate(example['dner']):
            field = 'dner_%s' % ner
            if field in extra_feature_fields:
                extra_features[i][extra_feature_fields[field]] = 1.0

    # tf特征（归一化），
    if args.use_tf:
        counter = Counter([word.lower() for word in example['dtext']])
        l = len(example['dtext'])
        for i, word in enumerate(example['dtext']):
            extra_features[i][extra_feature_fields['tf']] = counter[word.lower()] * 1.0 / l

    # 没有答案或答案为空，则返回without target
    if 'aspan' not in example or len(example['aspan']) == 0:
        return d_words_indices, q_words_indices, d_chars_indices, q_chars_indices, extra_features, example['id']

    # 如果single_answer，则返回第一个答案在dspan中开始和结束的index，否则返回indices list
    if single_answer:
        assert (len(example['aspan']) > 0)
        start = torch.LongTensor(1).fill_(example['aspan'][0][0])
        end = torch.LongTensor(1).fill_(example['aspan'][0][1])
    else:
        start = [aspan[0] for aspan in example['aspan']]
        end = [aspan[1] for aspan in example['aspan']]

    return d_words_indices, q_words_indices, d_chars_indices, q_chars_indices, extra_features, start, end, example['id']


# ------------------------------------------------------------------------------
# batch相关
# ------------------------------------------------------------------------------

def batchify(batch):
    """ 将一个batch中的examples经过一些处理，人为合成为若干tensor。batch中每个example的格式：
    0. d_words_indices: 文档单词的indices: document_length
    1. q_words_indices: 问题单词的indices: question_length
    2. d_chars_indices: 文档每个单词的字符indices: document_length * word_length
    3. q_chars_indices: 问题每个单词的字符indices: question_length * word_length
    4. extra_features: 额外特征矩阵(文档单词个数 * feature_fields长度):
                    feature_fields
            word1   1.0, 1.0, ...
            word2   0.0, 1.0, ...
            ...     ...
            wordn   0.0, 0.0, ...
    5. start: 答案在dspan中开始的index(或indices list)
    6. end: 答案在dspan中结束的index(或indices list)
    9. example['id']: 问题id

    :return f:      batch * max_document_length * feature_fields长度
    :return d_w:    batch * max_document_length
    :return d_mask: batch * max_document_length
    :return q_w:    batch * max_question_length
    :return q_mask: batch * max_question_length
    :return d_c:    batch * max_document_length * max_document_word_length
    :return q_c:    batch * max_question_length * max_question_word_length
    :return s:      LongTensor(batch) 或 list[batch * 答案个数]
    :return e:      LongTensor(batch) 或 list[batch * 答案个数]
    :return ids:    list[batch]
    """
    # 下面的4个变量比传入多了一维，按第一维stack，变成batch * 传入维数，并且将文档单词长度padding到max_document_length
    ids = [example[-1] for example in batch]
    d_words_indices = [example[0] for example in batch]
    q_words_indices = [example[1] for example in batch]
    d_chars_indices = [example[2] for example in batch]
    q_chars_indices = [example[3] for example in batch]
    extra_features = [example[4] for example in batch]

    # 每个document_indices中不够max_document_length的地方用0填充，mask置为1
    max_document_length = max([document.size(0) for document in d_words_indices])
    # d_w、d_mask: batch * max_document_length
    d_w = torch.LongTensor(len(d_words_indices), max_document_length).zero_()
    d_mask = torch.ByteTensor(len(d_words_indices), max_document_length).fill_(1)

    # f: batch * max_document_length * feature_fields长度
    if extra_features[0] is None:
        f = None
    else:
        f = torch.zeros(len(d_words_indices), max_document_length, extra_features[0].size(1))

    # 逐一将documents_indices中每个document_indices复制到d中，并设置对应mask为0
    for i, document_indices in enumerate(d_words_indices):
        d_w[i, :document_indices.size(0)].copy_(document_indices)
        d_mask[i, :document_indices.size(0)].fill_(0)
        if f is not None:
            f[i, :document_indices.size(0)].copy_(extra_features[i])

    # 每个questions_indices中不够max_question_length的地方用0填充，mask置为1
    max_question_length = max([q.size(0) for q in q_words_indices])
    # q_w、q_mask: batch * max_question_length
    q_w = torch.LongTensor(len(q_words_indices), max_question_length).zero_()
    q_mask = torch.ByteTensor(len(q_words_indices), max_question_length).fill_(1)
    for i, question_indices in enumerate(q_words_indices):
        q_w[i, :question_indices.size(0)].copy_(question_indices)
        q_mask[i, :question_indices.size(0)].fill_(0)

    # ------------------------------------------------------------------------------
    # Char CNN相关
    # batch的seq中单词的最小长度，因为经过卷积后单词的长度变为max_seq_word_length - kernel_size + 1
    min_words_length = 6
    # batch的文档中单词的最大长度
    d_max_words_length = max([max([len(word) for word in document]) for document in d_chars_indices])
    d_max_words_length = d_max_words_length if d_max_words_length >= min_words_length else min_words_length
    # batch的问题中单词的最大长度
    q_max_words_length = max([max([len(word) for word in question]) for question in q_chars_indices])
    q_max_words_length = q_max_words_length if q_max_words_length >= min_words_length else min_words_length

    d_c = torch.LongTensor(len(d_words_indices), max_document_length, d_max_words_length).zero_()
    q_c = torch.LongTensor(len(d_words_indices), max_question_length, q_max_words_length).zero_()

    for i, _ in enumerate(d_chars_indices):
        for j, chars_indices in enumerate(d_chars_indices[i]):
            d_c[i][j, : len(chars_indices)].copy_(torch.LongTensor(chars_indices))

    for i, _ in enumerate(q_chars_indices):
        for j, chars_indices in enumerate(q_chars_indices[i]):
            q_c[i][j, : len(chars_indices)].copy_(torch.LongTensor(chars_indices))

    # ------------------------------------------------------------------------------

    # example中可能没有答案
    if len(batch[0]) == 6:
        return f, d_w, d_mask, q_w, q_mask, d_c, q_c, ids
    # example中有答案
    elif len(batch[0]) == 8:
        # 只有一个答案时，start和end都为LongTensor(1) => s,e: LongTensor(batch)
        if torch.is_tensor(batch[0][5]):
            s = torch.cat([example[5] for example in batch])
            e = torch.cat([example[6] for example in batch])
        # 有多个答案时，start和end为list[答案个数] => s,e: list[batch * 答案个数]
        else:
            s = [example[5] for example in batch]
            e = [example[6] for example in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return f, d_w, d_mask, q_w, q_mask, d_c, q_c, s, e, ids


# ------------------------------------------------------------------------------
# 评估相关
# ------------------------------------------------------------------------------

def validate(data_loader, model, train_states, type):
    """ 每个example格式：
    0. 文档单词indices  d: batch * max_document_length
    1. 额外特征         f: batch * max_document_length * feature_fields长度
    2. 文档单词mask     d_mask: batch * max_document_length
    3. 问题单词indices  q: batch * max_question_length
    4. 问题单词mask     q_mask: batch * max_question_length
    -3. 答案开始span     s: LongTensor(batch) 或 list[batch * 答案个数]
    -2. 答案结束span     e: LongTensor(batch) 或 list[batch * 答案个数]
    7. 问题id          ids: list[batch]
    """
    evaluation_time = Timer()
    accuracy_start = AverageMeter()
    accuracy_end = AverageMeter()
    exact_match = AverageMeter()

    # Make predictions
    examples_count = 0
    for examples_in_batch in data_loader:
        batch_size = examples_in_batch[0].size(0)
        # prediction_start, prediction_end：list[batch]
        prediction_start, prediction_end, _ = model.predict(examples_in_batch)

        target_start, target_end = examples_in_batch[-3:-1]

        accuracies = evaluate(prediction_start, target_start, prediction_end, target_end)
        accuracy_start.update(accuracies[0], batch_size)
        accuracy_end.update(accuracies[1], batch_size)
        exact_match.update(accuracies[2], batch_size)

        # 如果是计算train_dataset的准确度，只计算前10000个
        examples_count += batch_size
        if type == 'train' and examples_count >= 1e4:
            break

    logger.info('%s validation: Epoch = %d | start = %.2f | ' %
                (type, train_states['epoch'], accuracy_start.average) +
                'end = %.2f | exact = %.2f | examples = %d | ' %
                (accuracy_end.average, exact_match.average, examples_count) +
                'validation time = %.2f (s)' % evaluation_time.total_time())

    return {'exact_match': exact_match.average}


def evaluate(prediction_start, target_start, prediction_end, target_end):
    """ 计算exact start/end/complete match accuracies for a batch
    :param prediction_start, prediction_end: list[batch]
    :param target_start, target_end: LongTensor(batch)     或 list[batch * 答案个数]
                                => list[batch * list[long]] 或 list[batch * 答案个数]
    """

    # 将1维tensors转换为list[list[long]]
    if torch.is_tensor(target_start):
        target_start = [[start] for start in target_start]
        target_end = [[end] for end in target_end]

    batch_size = len(prediction_start)
    accuracy_start = AverageMeter()
    accuracy_end = AverageMeter()
    exact_match = AverageMeter()
    for i in range(batch_size):
        # 匹配start
        if prediction_start[i] in target_start[i]:
            accuracy_start.update(1)
        else:
            accuracy_start.update(0)
        # 匹配end
        if prediction_end[i] in target_end[i]:
            accuracy_end.update(1)
        else:
            accuracy_end.update(0)

        # exact_match指start和end都匹配
        # zip(lista, listb)将lista和listb中相同位置的元素zip成一个tuple，如zip([1,2],[4,5]) = [(1,4),(2,5)]
        if any([1 for start, end in zip(target_start[i], target_end[i]) if
                start == prediction_start[i] and end == prediction_end[i]]):
            exact_match.update(1)
        else:
            exact_match.update(0)
    return accuracy_start.average * 100, accuracy_end.average * 100, exact_match.average * 100


# ------------------------------------------------------------------------------
# 计时相关
# ------------------------------------------------------------------------------

class Timer(object):
    """ 计时器"""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def total_time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


# ------------------------------------------------------------------------------
# 计算平均损失相关
# ------------------------------------------------------------------------------

class AverageMeter(object):
    """ 计算和存储当前值、总和、值个数、平均值 """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
