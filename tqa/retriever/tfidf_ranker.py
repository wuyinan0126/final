import argparse
import logging
import os
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy
import prettytable
from scipy import sparse

from tqa import DEFAULTS, DATA_DIR
from tqa.retriever import utils
from tqa.retriever.tokenizer import CoreNlpTokenizer

# log_format = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
# console = logging.StreamHandler()
# console.setFormatter(log_format)
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(console)
logger = logging.getLogger(__name__)


class TfidfRanker(object):
    """ Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, tfidf_model_path, strict=True):
        """
        :param tfidf_model_path: tfidf模型路径
        :param strict: 当不能解析query时，True：报错，False：继续执行但返回空结果
        """
        logger.info('Loading %s' % tfidf_model_path)
        matrix, metadata = utils.load_tfidf(tfidf_model_path)
        self.grams_docs_matrix = matrix
        self.ngram = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = CoreNlpTokenizer(annotators=set())
        self.freqs = metadata['freqs'].squeeze()
        self.dict = metadata['dict']
        self.num_docs = len(self.dict[0])
        self.strict = strict
        # print(metadata['freqs'])

    def get_doc_index(self, doc_id):
        """ doc_id --> doc_index """
        return self.dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """ doc_index --> doc_id """
        return self.dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """ 将query转化为tfidf weighted word vector，并与grams_docs_matrix相乘得到最相关的docs """
        # vector: [1 * hash_size]
        query_vector = self.text2vector(query)
        # self.grams_docs_matrix: [hash_size * doc_size]
        # reuslt: [1 * doc_size]
        result = query_vector * self.grams_docs_matrix

        # reuslt.data中非0元素个数
        if len(result.data) <= k:
            # numpy.argsort返回从小到大的元素index list
            closest_doc_indices = numpy.argsort(-result.data)
        else:
            # numpy.argpartition返回最小的前k个数的index，这k个数是无序的
            top_k_doc_indices = numpy.argpartition(-result.data, k)[0:k]
            # 对上面k个数进行排序
            closest_doc_indices = top_k_doc_indices[numpy.argsort(-result.data[top_k_doc_indices])]

        doc_scores = result.data[closest_doc_indices]
        doc_ids = [self.get_doc_id(doc_index) for doc_index in result.indices[closest_doc_indices]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngram, uncased=True,
                             filter_fn=utils.grams_filter)

    def text2vector(self, query):
        """ 根据query创建一个sparse tfidf-weighted word vector.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # 获得query的tokens
        tokens = self.tokenizer.tokenize(utils.normalize(query))
        # 获得去除了停用词、标点的grams list，每个gram直接连接，如2gram: ['a','ab','b','bc']
        grams = tokens.ngrams(n=self.ngram, uncased=True, filter_fn=utils.grams_filter, as_strings=True)
        logger.info("TF-IDF query: " + ",".join(grams))
        hashed_grams = [utils.hash(gram, self.hash_size) for gram in grams]
        # 如果经过grams_filter过滤后没有有效gram
        if len(hashed_grams) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                # 返回空结果：[1 * hash_size]个0
                return sparse.csr_matrix((1, self.hash_size))

        # hashed_grams去重，返回去重元素list及每个的个数list
        hashed_grams_unique, hashed_grams_counts = numpy.unique(hashed_grams, return_counts=True)

        # log(tf + 1)中的tf: [query_hashed_grams_size]，hashed_grams在query中出现的次数
        tf = numpy.log1p(hashed_grams_counts)
        # freqs的index就是gram hash后的值，freqs类型是numpy.ndarray[query_hashed_grams_size]，freqs[[0,1]]取出freqs中index为0和1的值
        Nt = self.freqs[hashed_grams_unique]
        idf = numpy.log((self.num_docs - Nt + 0.5) / (Nt + 0.5))
        # idf: [query_hashed_grams_size]
        idf[idf < 0] = 0
        # tfidf: [query_hashed_grams_size]
        tfidf = numpy.multiply(tf, idf)

        """ 
        >>> indptr = np.array([0, 2, 3, 6]) # 第0行有(2-0)=2个非0值，第1行有(3-2)=1个，第2行有(6-3)=3个
        >>> indices = np.array([0, 2, 2, 0, 1, 2]) # 非0值的列号
        >>> data = np.array([1, 2, 3, 4, 5, 6]) # 非0数据
        >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
        array([[1, 0, 2],
               [0, 0, 3],
               [4, 5, 6]])
        """
        # 第0行有len(hashed_grams_unique)个非0值
        indptr = numpy.array([0, len(hashed_grams_unique)])
        vector = sparse.csr_matrix((tfidf, hashed_grams_unique, indptr), shape=(1, self.hash_size))
        return vector


def rank(tfidf_model_path, query, k, strict=True):
    """ 查询前k个和query最相关的documents """
    ranker = TfidfRanker(tfidf_model_path, strict)
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('tfidf_builder.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tfidf-model-path', type=str, default=None)
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--k', type=int, default=DEFAULTS['tfidf_rank_k'])
    args = parser.parse_args()

    args.tfidf_model_path = os.path.join(DATA_DIR, args.tfidf_model_path)
    rank(
        tfidf_model_path=args.tfidf_model_path,
        query=args.query,
        k=args.k
    )
