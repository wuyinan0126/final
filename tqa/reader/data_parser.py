import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize

from tqa import DEFAULTS, DATA_DIR
from tqa.retriever.tokenizer import CoreNlpTokenizer

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


def load(dataset_path):
    """ 加载dataset json文件，格式为：
    {"data": [{
      "title": "TITLE",
      "paragraphs": [{
        "context": "CONTEXT",
        "qas": [{
          "answers": [{
            "answer_start": START,
            "text": "TEXT"
            },],
          "question": "QUESTION",
          "id": "ID"
        },]
      },]
    },]}
    转换为：
    useful_parts['qids'] = ['id1', 'id2',...]
    useful_parts['questions'] = ['question1?', 'question2?',...]
    useful_parts['answers'] = [[{'answer_start': 9, 'text': 'answer1'}, {...}], [{'answer_start': 9, 'text': 'answer2'}],...]
    useful_parts['contexts']: ['context1 answer1', 'context2 answer2',...]
    useful_parts['qid2cid'] = [0, 1,...] # 第i个问题对应的context是dataset['qid2cid'][i]
    """
    with open(dataset_path) as file:
        data = json.load(file)['data']
    useful_parts = {'qids': [], 'questions': [], 'answers': [],
                    'contexts': [], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            useful_parts['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                useful_parts['qids'].append(qa['id'])
                useful_parts['questions'].append(qa['question'])
                useful_parts['qid2cid'].append(len(useful_parts['contexts']) - 1)
                if 'answers' in qa:
                    useful_parts['answers'].append(qa['answers'])
    return useful_parts


def process(language, useful_parts, num_workers):
    """ 对有用部分进行格式化，输入为：
    useful_parts['qids'] = ['id1', 'id2',...]
    useful_parts['questions'] = ['question1?', 'question2?',...]
    useful_parts['answers'] = [[{'answer_start': 9, 'text': 'answer1'}, {...}], [{'answer_start': 9, 'text': 'answer2'}],...]
    useful_parts['contexts']: ['context1 answer1', 'context2 answer2',...]
    useful_parts['qid2cid'] = [0, 1,...] # 第i个问题对应的context是dataset['qid2cid'][i]
    转换为：
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
    pool_maker = partial(Pool, num_workers, initializer=pool_init)

    print("Context tokenizing...")

    # tokenize context
    pool = pool_maker(initargs=({'language': language, 'annotators': {'lemma', 'pos', 'ner'}},))
    context_tokens = pool.map(tokenize, useful_parts['contexts'])
    pool.close()
    pool.join()

    print("Question tokenizing...")

    # tokenize question
    pool = pool_maker(initargs=({'language': language, 'annotators': {'lemma'}},))
    question_tokens = pool.map(tokenize, useful_parts['questions'])
    pool.close()
    pool.join()

    print("Tokenization done")

    for qindex in range(len(useful_parts['qids'])):
        question_words = question_tokens[qindex]['words']
        question_lemma = question_tokens[qindex]['lemma']
        document_words = context_tokens[useful_parts['qid2cid'][qindex]]['words']
        document_span = context_tokens[useful_parts['qid2cid'][qindex]]['span']
        document_lemma = context_tokens[useful_parts['qid2cid'][qindex]]['lemma']
        document_pos = context_tokens[useful_parts['qid2cid'][qindex]]['pos']
        document_ner = context_tokens[useful_parts['qid2cid'][qindex]]['ner']
        # answer_span是answer在document中涵盖的若干个document_spans的index
        answer_span = []
        if len(useful_parts['answers']) > 0:
            for answer in useful_parts['answers'][qindex]:
                answer_begin = answer['answer_start']
                answer_end = answer_begin + len(answer['text'])
                # 找到document spans中answer开始和结束的index，token为一个tuple记录document中每个word开始和结束的index
                answer_span_begin = [index for index, token in enumerate(document_span) if token[0] == answer_begin]
                answer_span_end = [index for index, token in enumerate(document_span) if token[1] == answer_end]
                # 应该只有一个或没有answer_span_begin和answer_span_end
                assert (len(answer_span_begin) <= 1)
                assert (len(answer_span_end) <= 1)
                # 找到开始和结束的span
                if len(answer_span_begin) == 1 and len(answer_span_end) == 1:
                    answer_span.append((answer_span_begin[0], answer_span_end[0]))
        yield {
            'id': useful_parts['qids'][qindex],
            'qtext': question_words,
            'dtext': document_words,
            'dspan': document_span,
            'aspan': answer_span,
            'qlemma': question_lemma,
            'dlemma': document_lemma,
            'dpos': document_pos,
            'dner': document_ner,
        }


def parse(language, dataset_path, parsed_dataset_path=None, num_workers=None):
    """ 先经过load()加载原始格式的SQuAD json文件，并提取有用部分得到useful_parts
        再使用process()将useful_parts格式化，最后保存在processed_dataset_path中
    """
    dataset_path = os.path.join(DATA_DIR, dataset_path)
    useful_parts = load(dataset_path)

    if not parsed_dataset_path:
        squad_dataset_dir = os.path.dirname(dataset_path)
        squad_dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        parsed_dataset_path = os.path.join(squad_dataset_dir, squad_dataset_name + ".txt")
    else:
        parsed_dataset_path = os.path.join(DATA_DIR, parsed_dataset_path)

    with open(parsed_dataset_path, 'w') as file:
        for data in process(language, useful_parts, num_workers):
            file.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('data_parser.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--language', type=str, default=DEFAULTS['tokenizer_language'])
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--parsed-dataset-path', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    args = parser.parse_args()

    parse(
        language=args.language,
        dataset_path=args.dataset_path,
        parsed_dataset_path=None,
        num_workers=args.num_workers
    )
