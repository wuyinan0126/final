import codecs
import json
import logging
import re
from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize

import os

from tqa.retriever.tokenizer import CoreNlpTokenizer

log_format = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(log_format)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console)

# "aspan": \[\[.*?\], \[

TOKENIZER = None


def pool_init(tokenizer_opts):
    global TOKENIZER
    TOKENIZER = CoreNlpTokenizer(**tokenizer_opts)
    # close不能加括号，这里传入的是一个方法，而不是方法结果
    Finalize(TOKENIZER, TOKENIZER.close, exitpriority=100)


def tokenize(text):
    """ 多线程tokenize """
    global TOKENIZER
    tokens = TOKENIZER.tokenize(text)
    tokens = {
        'words': tokens.words(),  # tokens的data中所有TEXT字段
        'span': tokens.span(),  # tokens的data中所有SPAN字段
        'pos': tokens.pos(),  # tokens的data中所有POS字段
        'lemma': tokens.lemma(),  # tokens的data中所有LEMMA字段
        'ner': tokens.ner(),  # tokens的data中所有NER字段
    }
    return tokens


def load(dataset_path, num_workers=1, language='zh'):
    """ 加载dataset txt文件，格式为：
        第1行：文档文本，以['.', '!', '?', '。', '！', '？', '\t']为切割标示
        第2-n行：问题，格式为问题1<\t>问题2<\t>...问题m<\t>答案文本
    """
    id_base = os.path.basename(dataset_path).split(".")[0]

    document = ''
    questions = []
    answers = []
    first_line = True
    with open(dataset_path, encoding="utf-8") as file:
        for line in file:
            if first_line:
                document = line.strip()
                first_line = False
            else:
                s = line.strip().split('\t')
                answer_text = "\t".join(s[1:])
                question_text = s[0]
                answer = [(m.start(), m.end()) for m in re.finditer(answer_text, document)]
                if len(answer) > 1:
                    print(">1: " + question_text)
                    for a in answer:
                        start = a[0] - 20 if a[0] - 20 >= 0 else a[0]
                        end = a[1] + 20 if a[1] + 20 < len(document) else a[1]
                        print("\t" + document[start: end])
                elif len(answer) < 1:
                    print("<1: " + question_text)

                questions.append(question_text)
                answers.append(answer)

    pool_maker = partial(Pool, num_workers, initializer=pool_init)
    logger.info("Document tokenizing...")
    pool = pool_maker(initargs=({'language': language, 'annotators': {'lemma', 'pos', 'ner'}},))
    document_tokens = pool.map(tokenize, [document])
    pool.close()
    pool.join()

    # print(answers)
    answers_span = []

    document_span = document_tokens[0]['span']
    for answer in answers:
        a_span = []
        for a in answer:
            start = -1
            end = -1
            for i in range(0, len(document_span)):
                if document_span[i][0] <= a[0] < document_span[i][1]:
                    start = i
                    break
            for j in range(i, len(document_span)):
                if document_span[j][0] <= a[1] < document_span[j][1]:
                    end = j
                    break
            a_span.append((start, end))
        answers_span.append(a_span)

    # print(answers_span)

    logger.info("Question tokenizing...")
    pool = pool_maker(initargs=({'language': language, 'annotators': {'lemma'}},))
    question_tokens = pool.map(tokenize, questions)
    pool.close()
    pool.join()

    for i in range(0, len(questions)):
        yield {
            'id': id_base + str(i),
            'qtext': question_tokens[i]['words'],
            'dtext': document_tokens[0]['words'],
            'dspan': document_tokens[0]['span'],
            'aspan': answers_span[i],
            'qlemma': question_tokens[i]['lemma'],
            'dlemma': document_tokens[0]['lemma'],
            'dpos': document_tokens[0]['pos'],
            'dner': document_tokens[0]['ner'],
        }


def label(dataset_path, parsed_dataset_path):
    with codecs.open(parsed_dataset_path, 'w', encoding='utf-8') as file:
        for data in load(dataset_path):
            file.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    label(
        "/home/wuyinan/Desktop/final/data/datasets/bigdata/大数据日知录_0.txt",
        "/home/wuyinan/Desktop/final/data/datasets/bigdata/大数据日知录_0.json.txt",
    )
