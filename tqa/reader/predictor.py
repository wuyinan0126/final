import argparse
import logging
import os
import time

import prettytable
import torch

from tqa import DEFAULTS, DATA_DIR
from tqa.reader import utils
from tqa.reader.utils import str2bool
from tqa.retriever.tokenizer import CoreNlpTokenizer

# log_format = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
# console = logging.StreamHandler()
# console.setFormatter(log_format)
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(console)
logger = logging.getLogger(__name__)



class ReaderPredictor(object):
    """ 加载SquadRnn模型并通过给定的问题和文档预测答案 """

    def __init__(self, model_path=None, embedded_corpus_path=None):
        """
        :param embedded_corpus_path: 如果不为None，则会使用里面训练好的embeddings扩充词典
        """
        logger.info('Initializing model...')
        model_path = model_path or os.path.join(DEFAULTS['model_dir'], DEFAULTS['model_name'] + '.mdl')
        self.model = utils.load_model(model_path)

        if embedded_corpus_path:
            logger.info('Expanding dictionary...')
            words = utils.get_embedded_words(embedded_corpus_path)
            added_words, _ = self.model.expand_dictionary(words, chars=None)
            self.model.load_embeddings(added_words, embedded_corpus_path)

        logger.info('Initializing tokenizer...')
        annotators = set()
        if self.model.args.use_pos:
            annotators.add('pos')
        if self.model.args.use_lemma:
            annotators.add('lemma')
        if self.model.args.use_ner:
            annotators.add('ner')

        self.tokenizer = CoreNlpTokenizer(language=self.model.args.language, annotators=annotators)

    def predict(self, document, question, top_n=1):
        """ 预测一个文档-问题对
        :param document 文档字符串
        :param question 问题字符串
        :param candidates
        :param top_n 得分前n的答案
        """
        results = self.predict_batch([(document, question,)], top_n)
        return results[0]

    def predict_batch(self, batch, top_n=1):
        """ 预测batch个文档-问题对 """
        documents, questions = [], []
        for b in batch:
            documents.append(b[0])
            questions.append(b[1])

        q_tokens = list(map(self.tokenizer.tokenize, questions))
        d_tokens = list(map(self.tokenizer.tokenize, documents))

        examples = []
        for i in range(len(questions)):
            examples.append({
                'id': i,
                'qtext': q_tokens[i].words(),
                'qlemma': q_tokens[i].lemma(),
                'dtext': d_tokens[i].words(),
                'dlemma': d_tokens[i].lemma(),
                'dpos': d_tokens[i].pos(),
                'dner': d_tokens[i].ner(),
            })

        logger.info("Batchify...")
        examples_in_batch = utils.batchify(
            [utils.vectorize(example, self.model, single_answer=False) for example in examples]
        )
        start, end, score = self.model.predict(examples_in_batch, top_n)

        # 从start, end生成答案
        results = []
        for i in range(len(start)):
            predictions = []
            for j in range(len(start[i])):
                span = d_tokens[i].slice(start[i][j], end[i][j] + 1).untokenize()
                predictions.append((span, score[i][j]))
            results.append(predictions)
        return results

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()


def predict(model_path, document, question,
            top_k_answers=5, use_cuda=False, gpu_device=0, embedded_corpus_path=None):
    start_time = time.time()

    use_cuda = use_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu_device)
        logger.info('CUDA enabled (GPU %d)' % gpu_device)
    else:
        logger.info('Running on CPU only.')

    predictor = ReaderPredictor(model_path, embedded_corpus_path)
    if args.use_cuda:
        predictor.cuda()

    predictions = predictor.predict(document, question, top_k_answers)
    table = prettytable.PrettyTable(['Rank', 'Span', 'Score'])
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p[0], p[1]])
    print(table)
    print('Time: %.4f' % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('predictor.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--document', type=str, default=None)
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--top-k-answers', type=int, default=DEFAULTS['top_k_answers'])
    parser.add_argument('--use-cuda', type='bool', default=DEFAULTS['use_cuda'])
    parser.add_argument('--gpu-device', type=int, default=DEFAULTS['gpu_device'])
    parser.add_argument('--embedded-corpus-path', type=str, default=None)
    args = parser.parse_args()

    args.model_path = os.path.join(DATA_DIR, args.model_path)
    args.embedded_corpus_path = os.path.join(DATA_DIR, args.embedded_corpus_path) if args.embedded_corpus_path else None
    predict(
        model_path=args.model_path,
        document=args.document,
        question=args.question,
        top_k_answers=args.top_k_answers,
        use_cuda=args.use_cuda,
        gpu_device=args.gpu_device,
        embedded_corpus_path=args.embedded_corpus_path
    )
