import argparse
import copy
import json
import logging

import os
import pexpect

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

        """
        # 英文分词 on ubuntu
        java -mx3g -cp "/home/wuyinan/Desktop/final/data/corenlp/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -tokenize.options untokenizable=noneDelete,invertible=true -outputFormat json -prettyPrint false
        # 中文分词 on ubuntu
        java -mx3g -cp "/home/wuyinan/Desktop/final/data/corenlp/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -tokenize.options untokenizable=noneDelete,invertible=true -outputFormat json -prettyPrint false
        """
        # 命令正常运行后将出现交互提示符NLP>
        if self.language == 'zh':
            self.cmd = ['java', '-mx' + self.heap, '-cp', '"%s"' % self.classpath,
                        'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                        '-props', 'StanfordCoreNLP-chinese.properties',
                        '-annotators', annotators,
                        '-tokenize.options', options, '-outputFormat', 'json', '-prettyPrint', 'false']
        else:
            self.cmd = ['java', '-mx' + self.heap, '-cp', '"%s"' % self.classpath,
                        'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', annotators,
                        '-tokenize.options', options, '-outputFormat', 'json', '-prettyPrint', 'false']

        # logger.info(' '.join(self.cmd))

        # 使用pexpect是使得corenlp子进程keep alive，并获得句柄self.corenlp使得后续还可调用
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=300)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(self.cmd))
        self.corenlp.delaybeforesend = 0.1
        # 在 输入命令执行 到 读取结果 需要时间，不能设为0，设为100ms
        self.corenlp.delayafterread = 0.1
        # expect_exact: 不使用正则表达式精确匹配NLP>，匹配则返回0
        self.corenlp.expect_exact('NLP>', searchwindowsize=-1, timeout=300)

    def tokenize(self, text):
        """ 将text输入self.corenlp句柄
        :return: Tokens，Tokens中的data包括多个(TEXT, TEXT_WS, SPAN, POS, LEMMA, NER)
        """
        logger.info(text[0:10] + "..." if len(text) > 10 else text)

        # 如果在text中出现了NLP>则返回错误
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # 输入q退出子进程
        if text.lower().strip() == 'q':
            token = text.strip()
            index = text.index(token)
            data = [(token, text[index:], (index, index + 1), 'NN', 'q', 'O')]
            return Tokens(data, self.annotators)

        text = text.replace('\n', '\t')
        # 输入text
        try:
            self.corenlp.sendline(text.encode('utf-8'))
            self.corenlp.expect_exact('NLP>', searchwindowsize=-1, timeout=10)
        except pexpect.exceptions.TIMEOUT as e:
            logger.info("ERROR in Tokenizer: " + (text[0:100] + "..." if len(text) > 100 else text))
            self.corenlp.sendline(' '.join(self.cmd))
            self.corenlp.expect_exact('NLP>', searchwindowsize=-1, timeout=300)
            return None

        # self.corenlp.before: 保存了到匹配到关键字为止，缓存里面已有的所有数据
        output = self.corenlp.before
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
        start = output.find(b'{"sentences":')
        output = json.loads(output[start:].decode('utf-8'))

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
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser('tokenizer.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--heap', type=str, default=DEFAULTS['tokenizer_heap'])
    parser.add_argument('--text', type=str, default=None)
    args = parser.parse_args()

    tokenizer = CoreNlpTokenizer(**{'heap': args.heap})
    print(tokenizer.tokenize(args.text).data)
