import argparse
import copy
import json
import logging

import os

from stanfordcorenlp import StanfordCoreNLP

from tqa import DEFAULTS, DATA_DIR
from tqa.retriever.tokens import Tokens
from tqa.retriever.utils import special_char

logger = logging.getLogger(__name__)


class CoreNlpTokenizer():
    def __init__(self, **kwargs):
        """
        :arg language: 语言
        :arg classpath: corenlp jars的目录
        :arg annotators: 一个可能包含'pos', 'lemma', 'ner'的集合
        :arg heap: java堆内存
        """
        self.language = kwargs.get('language', DEFAULTS['tokenizer_language'])
        self.annotators = copy.deepcopy(kwargs.get('annotators', DEFAULTS['tokenizer_annotators']))
        self.classpath = os.path.join(DATA_DIR, kwargs.get('classpath', DEFAULTS['tokenizer_classpath']))
        self.heap = kwargs.get('heap', DEFAULTS['tokenizer_heap'])

        # annotators: tokenize(分词), ssplit(断句), pos(词性标注), lemma(词元化), ner(命名实体识别)
        annotators = ['tokenize', 'ssplit']
        if 'ner' in self.annotators:
            annotators.extend(['pos', 'lemma', 'ner'])
        elif 'lemma' in self.annotators:
            annotators.extend(['pos', 'lemma'])
        elif 'pos' in self.annotators:
            annotators.extend(['pos'])
        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete', 'invertible=true'])

        self.nlp = StanfordCoreNLP(self.classpath, memory=self.heap, lang='zh')
        self.props = {
            'annotators': annotators,
            'pipelineLanguage': 'zh',
            'outputFormat': 'json',
            'prettyPrint': 'False',
            'tokenize.options': options,
        }

    def tokenize(self, text):
        """ 将text输入self.corenlp句柄
        :return: Tokens，Tokens中的data包括多个(TEXT, TEXT_WS, SPAN, POS, LEMMA, NER)
        """
        # logger.info(text[0:10] + "..." if len(text) > 10 else text)

        text = text.replace('\n', '\t')

        output = self.nlp.annotate(text, properties=self.props)
        """ 有效输出: 
        {
          "sentences": [
            {
              "index": 0,
              "entitymentions": [],
              "tokens": [
                {
                  "index": 1,
                  "word": "hello",
                  "originalText": "hello",
                  "lemma": "hello",
                  "characterOffsetBegin": 0,
                  "characterOffsetEnd": 5,
                  "pos": "UH",
                  "ner": "O",
                  "before": "",
                  "after": " "
                },
              ]
            }
          ]
        }"""
        output = json.loads(output.decode('utf-8'))

        data = []
        tokens = [t for s in output['sentences'] for t in s['tokens']]
        for i in range(len(tokens)):
            # 获得 单词 及 其后的空白符（如果有的话）
            start_whitespace = tokens[i]['characterOffsetBegin']
            if i + 1 < len(tokens):
                end_whitespace = tokens[i + 1]['characterOffsetBegin']
            else:
                end_whitespace = tokens[i]['characterOffsetEnd']

            data.append((
                special_char(tokens[i]['word']),
                text[start_whitespace: end_whitespace],
                (tokens[i]['characterOffsetBegin'], tokens[i]['characterOffsetEnd']),
                tokens[i].get('pos', None),
                tokens[i].get('lemma', None),
                tokens[i].get('ner', None)
            ))
        return Tokens(data, self.annotators)

    def close(self):
        self.nlp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('tokenizer.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--heap', type=str, default=DEFAULTS['tokenizer_heap'])
    parser.add_argument('--text', type=str, default=None)
    args = parser.parse_args()

    tokenizer = CoreNlpTokenizer(**{'heap': args.heap})
    print(tokenizer.tokenize(args.text).data)
