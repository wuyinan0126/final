import argparse
import json
import logging
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing.pool import Pool, ThreadPool
from multiprocessing.util import Finalize
from urllib.parse import urlparse, parse_qs, unquote

import os

import time
import torch

from tqa import DEFAULTS, DATA_DIR
from tqa.reader import utils
from tqa.reader.utils import load_model, get_embedded_words, str2bool, str2str_list
from tqa.retriever import utils as r_utils
from tqa.retriever.db import Db
from tqa.retriever.tfidf_ranker import TfidfRanker
from tqa.retriever.tokenizer import CoreNlpTokenizer
from tqa.reuser import fasttext_matcher
from tqa.reuser.fasttext_matcher import FastTextMatcher

log_format = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(log_format)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console)


class TqaHttpServer:
    def __init__(self, core=None):
        server = HTTPServer(('', 9126), TqaHttpRequestHandler)
        server.core = core
        print("Server started on 127.0.0.1:9126...")
        server.serve_forever()


class TqaHttpRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path)
        query_map = parse_qs(unquote(path.query))
        question = query_map["q"][0] if "q" in query_map else None
        questions = query_map["s"][0] if "s" in query_map else None

        content = ''

        if question:
            answers = self.server.core.answer(question)
            if answers:
                content = json.dumps({'answers': answers}, indent=2, separators=(',', ': '))
        elif questions:
            # questions: id#question_content$id#question_content => {"id": "1", "score": 0.5}
            id, score = self.server.core.reuse([q.split('#') for q in questions.split('$')])
            if id and score:
                content = json.dumps({'id': id, 'score': score}, indent=2, separators=(',', ': '))

        content = content.encode('utf-8').decode('raw-unicode-escape')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        # self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(bytes(content, "utf-8"))

    def do_POST(self):
        self.do_GET()


def pool_init(tokenizer_opts):
    global TOKENIZER, DB
    TOKENIZER = CoreNlpTokenizer(**tokenizer_opts)
    Finalize(TOKENIZER, TOKENIZER.close, exitpriority=100)
    DB = Db()
    Finalize(DB, DB.close, exitpriority=100)


def build_tokens(document_id, db_table):
    global DB
    return r_utils.build_token(DB, db_table=db_table, document_id=document_id)


def tokenize(question):
    global TOKENIZER
    return TOKENIZER.tokenize(question)


class TqaCore():
    def __init__(self, ranker_opts, reader_opts, reuser_opts, num_workers=None):
        logger.info('Initializing document rankers...')
        tfidf_model_paths = ranker_opts.get('tfidf_model_paths')
        self.tfidf_rank_k = ranker_opts.get('tfidf_rank_k', DEFAULTS['tfidf_rank_k'])
        self.rankers = {}
        for tfidf_model_path in tfidf_model_paths:
            db_table = os.path.basename(tfidf_model_path).split("_")[0]
            self.rankers[db_table] = TfidfRanker(tfidf_model_path)

        logger.info('Initializing document reader...')
        model_path = reader_opts.get('reader_model_path')
        self.reader = load_model(model_path, new_args=None, model_type='reader')
        embedded_corpus_path = reader_opts.get('embedded_corpus_path', None)
        if embedded_corpus_path:
            logger.info('Expanding dictionary...')
            words = get_embedded_words(embedded_corpus_path)
            added_words, _ = self.reader.expand_dictionary(words, chars=None)
            self.reader.load_embeddings(added_words, embedded_corpus_path)
        use_cuda = reader_opts.get('use_cuda', None) and torch.cuda.is_available()
        if use_cuda:
            self.reader.cuda()
        self.top_k_answers = reader_opts.get('top_k_answers', DEFAULTS['top_k_answers'])

        logger.info('Initializing tokenizer and retriever...')
        annotators = set()
        if self.reader.args.use_pos:
            annotators.add('pos')
        if self.reader.args.use_lemma:
            annotators.add('lemma')
        if self.reader.args.use_ner:
            annotators.add('ner')
        tokenizer_opts = {'language': self.reader.args.language, 'annotators': annotators}
        self.num_workers = num_workers
        self.pool = Pool(
            num_workers,
            initializer=pool_init,
            initargs=(tokenizer_opts,)
        )

        logger.info('Initializing matcher...')
        bin_path = reuser_opts.get('embedded_corpus_bin_path')
        self.matcher = FastTextMatcher(bin_path)

    def reuse(self, id_questions):
        ids = []
        questions = []
        for iq in id_questions:
            ids.append(iq[0])
            questions.append(iq[1])

        q_tokens = self.pool.map_async(tokenize, questions)
        index, score = self.matcher.match_tokens(q_tokens)
        return ids[index], score

    def answer(self, question):
        start_time = time.time()
        logger.info('Processing question: %s...' % question)
        logger.info('Retrieving top %d documents...' % self.tfidf_rank_k)

        with ThreadPool(self.num_workers) as threads:
            _rank = partial(self.rank, question=question)
            results = threads.map(_rank, self.rankers.keys())
        logger.info('Answer elapse = %d' % (time.time() - start_time))
        return results

    def rank(self, db_table, question):
        logger.info("Finding closest documents...")
        result = [self.rankers[db_table].closest_docs(query=question, k=self.tfidf_rank_k)]
        documents_ids, documents_scores = zip(*result)
        documents_ids = documents_ids[0]
        documents_scores = documents_scores[0]
        print(db_table, documents_ids, documents_scores)

        if len(documents_ids) == 0:
            return None

        logger.info("Tokenizing question...")
        q_tokens = self.pool.map_async(tokenize, [question])
        _build_tokens = partial(build_tokens, db_table=db_table)
        d_rank_k_tokens = self.pool.map_async(_build_tokens, documents_ids)

        q_tokens = q_tokens.get()
        d_rank_k_tokens = d_rank_k_tokens.get()

        examples = []
        for i in range(len(d_rank_k_tokens)):
            examples.append({
                'id': documents_ids[i],
                'qtext': q_tokens[0].words(),
                'qlemma': q_tokens[0].lemma(),
                'dtext': d_rank_k_tokens[i].words(),
                'dlemma': d_rank_k_tokens[i].lemma(),
                'dpos': d_rank_k_tokens[i].pos(),
                'dner': d_rank_k_tokens[i].ner(),
            })

        logger.info("Batchify...")
        examples_in_batch = utils.batchify(
            [utils.vectorize(example, self.reader, single_answer=False) for example in examples]
        )
        start, end, score = self.reader.predict(examples_in_batch, self.top_k_answers)

        # 从start, end生成答案
        results = []
        for i in range(len(start)):
            print(documents_ids[i])
            for j in range(len(start[i])):
                answer = d_rank_k_tokens[i].slice(start[i][j], end[i][j] + 1).untokenize()
                text = d_rank_k_tokens[i].answer_sentence(start[i][j], end[i][j] + 1).untokenize()
                results.append({'score': score[i][j].item(), 'answer': answer, 'text': text, 'id': documents_ids[i]})

        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('server.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', str2bool)
    parser.register('type', 'list', str2str_list)
    parser.add_argument('--tfidf-model-paths', type='list', default=None)
    parser.add_argument('--tfidf-rank-k', type=int, default=DEFAULTS['tfidf_rank_k'])
    parser.add_argument('--reader-model-path', type=str, default=None)
    parser.add_argument('--embedded-corpus-path', type=str, default=None)
    parser.add_argument('--embedded-corpus-bin-path', type=str, default=None)
    parser.add_argument('--use-cuda', type='bool', default=DEFAULTS['use_cuda'])
    parser.add_argument('--num-workers', type=int, default=DEFAULTS['num_workers'])
    parser.add_argument('--top-k-answers', type=int, default=DEFAULTS['top_k_answers'])

    args = parser.parse_args()
    args.reader_model_path = os.path.join(DATA_DIR, args.reader_model_path)
    args.embedded_corpus_path = os.path.join(DATA_DIR, args.embedded_corpus_path) if args.embedded_corpus_path else None
    args.tfidf_model_paths = [os.path.join(DATA_DIR, tfidf_model_path) for tfidf_model_path in args.tfidf_model_paths]

    core = TqaCore(
        ranker_opts={
            'tfidf_model_paths': args.tfidf_model_paths,
            'tfidf_rank_k': args.tfidf_rank_k
        },
        reader_opts={
            'reader_model_path': args.reader_model_path,
            'embedded_corpus_path': args.embedded_corpus_path,
            'use_cuda': args.use_cuda,
            'top_k_answers': args.top_k_answers
        },
        reuser_opts={
            'embedded_corpus_bin_path': args.embedded_corpus_path,
        },
        num_workers=args.num_workers,
    )

    server = TqaHttpServer(core)
