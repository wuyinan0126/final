import unicodedata

import os
import regex
import numpy

from fastText import load_model

from tqa import DATA_DIR, DEFAULTS

STOPWORDS_ZH = {

}


class FastTextMatcher():
    def __init__(self, bin_path):
        fasttext_model_path = os.path.join(DATA_DIR, bin_path)
        self.fasttext = load_model(fasttext_model_path)

    def __word_filter(self, word):
        """ 去除停用词、标点和复数结尾 """
        word = unicodedata.normalize('NFD', word)
        if regex.match(r'^\p{P}+$', word): return True
        if word.lower() in STOPWORDS_ZH: return True
        return False

    def __grams_filter(self, grams, mode='any'):
        """ 判断是否保留该grams
        :param grams: 由最大长度为n个words组成的list
        """
        filtered = [self.__word_filter(word) for word in grams]
        return any(filtered)

    def __similarity(self, v1, v2):
        n1 = numpy.linalg.norm(v1)
        n2 = numpy.linalg.norm(v2)
        return (numpy.dot(v1, v2) / n1 / n2).item()

    def match(self, text_list):
        vectors = []
        for text in text_list:
            vectors.append(self.fasttext.get_sentence_vector(text))

        max_similarity = 0
        max_similarity_index = 0
        for i in range(1, len(vectors)):
            similarity = self.__similarity(vectors[0], vectors[i])
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = i

        return max_similarity_index, max_similarity

    def match_tokens(self, tokens_list):
        ngrams = []
        for tokens in tokens_list:
            # 获得去除了停用词、标点的grams list，每个gram用空格连接，如2gram: ['a','ab','b','bc']
            ngram = tokens.ngrams(n=2, uncased=False, filter_fn=self.__grams_filter, as_strings=True)
            ngrams.append(' '.join(ngram))
        return self.match(ngrams)


if __name__ == '__main__':
    # python tqa/reuser/fasttext_matcher.py
    matcher = FastTextMatcher(DEFAULTS['embedded_corpus_bin_path'])
    print(matcher.match([
        '梁紫媛 因 职务 便利 掌握 了 公司 业务 模型 分类 方面 的 大量 原始 数据',
        '李莉丝 、 黄悦 二人 先后 接触 梁紫媛，向 其 打听 印度 模型 制作 情况',
        '梁紫媛 明知 李莉丝 、 黄悦 二人 已经 离职 ， 仍 将 公司 重要 产品 数据 外泄',
        '入职 后 主要 负责 topbuzz 业务线 英语 分类 模型 的 数据 标注 和 模型 训练',
    ]))
