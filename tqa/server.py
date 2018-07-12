import argparse
import json
import logging

import re
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing.pool import Pool, ThreadPool
from multiprocessing.util import Finalize
from urllib.parse import urlparse, parse_qs, unquote, quote

import os

import time
import torch
from selenium import webdriver

from tqa import DEFAULTS, DATA_DIR, LOGS_DIR
from tqa.reader import utils
from tqa.reader.utils import load_model, get_embedded_words, str2bool, str2str_list
from tqa.retriever import utils as r_utils
from tqa.retriever.db import Db
from tqa.retriever.tfidf_ranker import TfidfRanker
from tqa.retriever.tokenizer import CoreNlpTokenizer
from tqa.reuser.fasttext_matcher import FastTextMatcher

logger = logging.getLogger()


def set_logger(log_file_path):
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(log_format)
    logger.addHandler(console)
    if log_file_path:
        file = logging.FileHandler(log_file_path, 'w')
        file.setFormatter(log_format)
        logger.addHandler(file)


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
            qs = question.split('@')
            question_title = qs[0]
            question_all = question_title if len(qs) < 2 else question_title + " " + qs[1]
            answers = self.server.core.answer(question_title, question_all)
            if answers:
                content = json.dumps({'answers': answers}, indent=2, separators=(',', ': '))
        elif questions:
            # questions: id#question_content$id#question_content => {"id": "1", "score": 0.5}
            questions = json.loads(questions)
            id, score = self.server.core.reuse(questions)
            if id and score:
                content = json.dumps({'id': id, 'score': score}, indent=2, separators=(',', ': '))
            else:
                content = json.dumps({'id': -1, 'score': -1}, indent=2, separators=(',', ': '))

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


class TqaCore(object):
    def __init__(self, ranker_opts, reader_opts, reuser_opts, num_workers=None, online=True):
        start = time.time()
        self.online = online
        if self.online:
            self.session = requests.Session()
            self.adapter = HTTPAdapter(pool_connections=5, pool_maxsize=5, max_retries=5)
            self.session.mount('http://', self.adapter)
            self.session.mount('https://', self.adapter)
            self.header = {
                'Content-Type': 'application/x-www-form-urlencode',
                'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
            }
            for key, value in enumerate(self.header):
                capability_key = 'phantomjs.page.customHeaders.{}'.format(key)
                webdriver.DesiredCapabilities.PHANTOMJS[capability_key] = value
            self.browser = webdriver.PhantomJS(executable_path='./phantomjs', service_log_path=os.path.devnull)

        logger.info('Initializing reuser...')
        bin_path = reuser_opts.get('embedded_corpus_bin_path')
        threshold = reuser_opts.get('threshold')
        self.matcher = FastTextMatcher(bin_path, threshold)

        logger.info('Initializing document rankers...')
        tfidf_model_paths = ranker_opts.get('tfidf_model_paths')
        self.tfidf_rank_k = ranker_opts.get('tfidf_rank_k', DEFAULTS['tfidf_rank_k'])
        self.rankers = {}
        for tfidf_model_path in tfidf_model_paths:
            db_table = os.path.basename(tfidf_model_path).split("_")[0]
            self.rankers[db_table] = TfidfRanker(tfidf_model_path)

        logger.info('Initializing document reader...')
        model_path = reader_opts.get('reader_model_path')
        self.reader = load_model(model_path, new_args=None)
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
        tokenizer_opts = {
            'language': self.reader.args.language,
            'annotators': annotators,
            # 'timeout': 10000,
        }
        self.num_workers = num_workers
        self.pool = Pool(
            num_workers,
            initializer=pool_init,
            initargs=(tokenizer_opts,)
        )

        end = time.time()
        logger.info('Server start elapse: {min}min {sec}sec'.format(
            min=int(end - start) // 60, sec=int(end - start) % 60)
        )

    def reuse(self, questions):
        ids = []
        titles = []
        descriptions = []
        for question in questions:
            ids.append(question['id'])
            titles.append(question['title'])
            descriptions.append(question['desc'])

        # q_tokens包含标题和描述
        q_tokens = self.pool.map_async(tokenize, [titles[i] + ' ' + descriptions[i] for i in range(0, len(titles))])
        q_tokens = q_tokens.get()
        score, index = self.matcher.match(q_tokens, titles, descriptions)
        return ids[index], score

    def answer(self, question_title, question_all):
        start_time = time.time()
        logger.info('Processing question: %s...' % question_title)
        logger.info('Retrieving top %d documents...' % self.tfidf_rank_k)

        results = None
        if self.online:
            results = self.online_rank(question_title=question_title, question_all=question_all)

        if not results:
            with ThreadPool(self.num_workers) as threads:
                _rank = partial(self.rank, question_title=question_title, question_all=question_all)
                results = threads.map(_rank, self.rankers.keys())

        logger.info('Answer elapse = %d' % (time.time() - start_time))
        return results

    def answerOne(self, question_title, question_all, d_tokens, d_ids):
        logger.info("Tokenizing question...")
        q_tokens = self.pool.map_async(tokenize, [question_title])
        q_tokens = q_tokens.get()

        examples = []
        for i in range(len(d_tokens)):
            examples.append({
                'id': d_ids[i],
                'qtext': q_tokens[0].words(),
                'qlemma': q_tokens[0].lemma(),
                'dtext': d_tokens[i].words(),
                'dlemma': d_tokens[i].lemma(),
                'dpos': d_tokens[i].pos(),
                'dner': d_tokens[i].ner(),
            })

        logger.info("Batchify...")
        examples_in_batch = utils.batchify(
            [utils.vectorize(example, self.reader, single_answer=False) for example in examples]
        )
        start, end, score = self.reader.predict(examples_in_batch, self.top_k_answers)

        # 从start, end生成答案
        results = []
        for i in range(len(start)):
            print(d_ids[i])
            for j in range(len(start[i])):
                answer = d_tokens[i].slice(start[i][j], end[i][j] + 1).untokenize()
                text = d_tokens[i].answer_sentence(start[i][j], end[i][j] + 1).untokenize()
                results.append({'score': score[i][j].item(), 'answer': answer, 'text': text, 'id': d_ids[i]})

        return results

    def online_rank(self, question_title, question_all):
        question_title = re.sub('[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', "", question_title)
        url = 'https://so.csdn.net/so/search/s.do?q=' + quote(question_title) + '&t=blog'
        self.browser.get(url)
        html = self.browser.page_source
        soup = BeautifulSoup(html, 'html.parser')
        link = soup.find_all('dl', {"class": "search-list J_search"})[0] \
            .find('dd', {'class': 'search-link'}) \
            .find('a')['href']
        logger.info('First Blog: %s' % link)

        resp = self.session.request('GET', link, params=None)
        soup = BeautifulSoup(resp.content, "html.parser")
        title = soup.find('h1', {'class': 'title-article'}).get_text()
        content = soup.find(id="article_content").get_text()
        content = re.sub(r"\t+|\n+|\r+", "", content)  # 去除非空格的空白符
        content = re.sub(r"\s{2,}", " ", content)
        # print(content)

        logger.info("Tokenizing document...")
        d_tokens = self.pool.map_async(tokenize, [content])
        d_tokens = d_tokens.get()

        return self.answerOne(question_title, question_all, d_tokens, ['blog@' + link + '@' + title])

    def rank(self, db_table, question_title, question_all):
        logger.info("Finding closest documents...")
        result = [self.rankers[db_table].closest_docs(query=question_all, k=self.tfidf_rank_k)]
        documents_ids, documents_scores = zip(*result)
        documents_ids = documents_ids[0]
        documents_scores = documents_scores[0]
        print(db_table, documents_ids, documents_scores)

        if len(documents_ids) == 0:
            return None

        logger.info("Tokenizing document...")
        _build_tokens = partial(build_tokens, db_table=db_table)
        d_rank_k_tokens = self.pool.map_async(_build_tokens, documents_ids)
        d_rank_k_tokens = d_rank_k_tokens.get()

        return self.answerOne(question_title, question_all, d_rank_k_tokens, documents_ids)


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
    parser.add_argument('--threshold', type=float, default=DEFAULTS['threshold'])
    parser.add_argument('--online', type='bool', default=DEFAULTS['online'])

    args = parser.parse_args()
    args.reader_model_path = os.path.join(DATA_DIR, args.reader_model_path)
    args.embedded_corpus_path = os.path.join(DATA_DIR, args.embedded_corpus_path) if args.embedded_corpus_path else None
    args.tfidf_model_paths = [os.path.join(DATA_DIR, tfidf_model_path) for tfidf_model_path in args.tfidf_model_paths]

    set_logger(os.path.join(LOGS_DIR, ("server_%s.log" % time.strftime("%Y%m%d_%H%M%S"))))

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
            'embedded_corpus_bin_path': args.embedded_corpus_bin_path,
            'threshold': args.threshold,
        },
        num_workers=args.num_workers,
        online=args.online,
    )

    server = TqaHttpServer(core)
