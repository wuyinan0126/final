import copy

from tqa import DEFAULTS


class Tokens(object):
    """ 用于表示一系列tokenized text，1个Tokens对象包含1个data，1个data包含多个(TEXT, TEXT_WS, SPAN, POS, LEMMA, NER) """

    TEXT = 0  # 单词: 'hello'
    TEXT_WS = 1  # 单词及其后的空白符（如果有的话）: 'hello '
    SPAN = 2  # 在句子中的位置: (begin, end): (0, 5)
    POS = 3  # part of speech: 词性: 'UH'
    LEMMA = 4  # 词元: 'hello'
    NER = 5  # named entity recognition: 命名实体识别: 'O'
    MARKS = ['.', '!', '?', '。', '！', '？', '\t']  # 切割标志

    def __init__(self, data, annotators=DEFAULTS['tokenizer_annotators'], opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        return len(self.data)

    def slice(self, i=None, j=None):
        """ 获取[i,j)的tokens """
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j] if j else self.data[i:]
        return new_tokens

    def untokenize(self):
        """ 获取文本 """
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def answer_sentence(self, i, j):
        """ 获取答案所在的句子 """
        new_tokens = copy.copy(self)
        p = 0
        for p in range(i - 1, -1, -1):
            if self.data[p][self.TEXT] in self.MARKS:
                break
        q = 0
        for q in range(j + 1, len(self.data), 1):
            if self.data[q][self.TEXT] in self.MARKS:
                break
        p = p if p == 0 else p + 1
        new_tokens.data = self.data[p: q]
        return new_tokens

    def words(self, uncased=False):
        """ 返回data中所有TEXT字段
        :param uncased True为转为小写字母
        """
        if uncased:
            return [data[self.TEXT].lower() for data in self.data]
        else:
            return [data[self.TEXT] for data in self.data]

    def words_ws(self):
        """ 返回data中所有TEXT_WS字段 """
        return [data[self.TEXT_WS] for data in self.data]

    def span(self):
        """ 返回data中所有SPAN字段：[start, end) """
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """ 返回data中所有POS字段"""
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemma(self):
        """ 返回data中所有LEMMA字段 """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def ner(self):
        """ 返回data中所有NER字段 """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """ 获得grams list
        :param uncased: True为小写化
        :param filter_fn: 判断是否保留某grams
        :param as_strings: 返回ngram字符串形式，每个gram直接连接，如2gram: ['a','ab','b','bc']
                           或 由(index_begin, index_end)组成的list形式
        """
        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not filter_fn(words[s:e + 1], 'any')]

        if as_strings:
            ngrams = ['{}'.format(''.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups
