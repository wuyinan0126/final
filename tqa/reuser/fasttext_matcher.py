import json
import logging
import unicodedata

import os
import regex
import numpy

from fastText import load_model
from aip import AipNlp

from tqa import DATA_DIR, DEFAULTS

logger = logging.getLogger(__name__)

APP_ID = '11296977'
API_KEY = 'wUQGnSdwkjTDU3YxhlD0VMC1'
SECRET_KEY = 'PomMd9fNOsFHGV6zcfSe4Hev1CTiiSxk'

STOPWORDS_ZH = {

}


class FastTextMatcher():
    def __init__(self, bin_path):
        fasttext_model_path = os.path.join(DATA_DIR, bin_path)
        self.fasttext = load_model(fasttext_model_path)
        self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

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

    def get_top_k_similar(self, text_list, k=5):
        similarities = []
        source = self.fasttext.get_sentence_vector(text_list[0])
        for i in range(1, len(text_list)):
            target = self.fasttext.get_sentence_vector(text_list[i])
            similarity = self.__similarity(source, target)
            similarities.append((similarity, i))

        similarities = sorted(similarities, reverse=True)
        return similarities[:k]

    def match(self, tokens_list):
        ngrams = []
        for tokens in tokens_list:
            # 获得去除了停用词、标点的grams list，每个gram用空格连接，如2gram: ['a','ab','b','bc']
            ngram = tokens.ngrams(n=2, uncased=False, filter_fn=self.__grams_filter, as_strings=True)
            ngram = ' '.join(ngram)
            logger.info('Question ngram: ' + ngram)
            ngrams.append(ngram)

        # scores: [(score, index),]
        scores = []
        source = ''.join(tokens_list[0].words_ws())
        # top_k_similar: [(similarity, index),]
        top_k_similar = self.get_top_k_similar(ngrams)
        for s in top_k_similar:
            target = ''.join(tokens_list[s[1]].words_ws())
            scores.append((self.get_baidu_score(source, target), s[1]))

        scores = sorted(scores, reverse=True)
        return scores[0][0], scores[0][1]

    def get_baidu_score(self, source, target):
        def cut_text(text):
            return text if len(text) < 200 else text[:100] + text[-100:]

        source = cut_text(source)
        target = cut_text(target)
        baidu_similar = self.client.simnet(source, target, {"model": "CNN"})
        logger.info("Similar: " + str(baidu_similar))
        return baidu_similar['score']


if __name__ == '__main__':
    # python tqa/reuser/fasttext_matcher.py
    matcher = FastTextMatcher(DEFAULTS['embedded_corpus_bin_path'])
    # print(matcher.get_top_k_similar([
    #     '梁紫媛 因 职务 便利 掌握 了 公司 业务 模型 分类 方面 的 大量 原始 数据',
    #     '李莉丝 、 黄悦 二人 先后 接触 梁紫媛，向 其 打听 印度 模型 制作 情况',
    #     '梁紫媛 明知 李莉丝 、 黄悦 二人 已经 离职 ， 仍 将 公司 重要 产品 数据 外泄',
    #     '入职 后 主要 负责 topbuzz 业务线 英语 分类 模型 的 数据 标注 和 模型 训练',
    # ]))
    matcher.get_baidu_score(
        '梁紫媛因职务便利掌握了公司业务模型分类方面的大量原始数据',
        '入职后主要负责topbuzz业务线英语分类模型的数据标注和模型训练'
    )
