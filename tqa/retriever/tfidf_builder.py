import argparse
import logging
import math
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize

import numpy
from scipy import sparse

from tqa import DEFAULTS, DATA_DIR
from tqa.retriever import utils
from tqa.retriever.db import Db
from tqa.retriever.utils import build_token

log_format = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(log_format)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console)

ID2INDEX = None
DB = None


def pool_init():
    global DB
    DB = Db()
    Finalize(DB, DB.close, exitpriority=100)


def count(ngram, hash_size, db_table, document_id):
    """ 从数据库中读取document_id的text，并计算hashed ngrams counts
    :param ngram: n元文法的n
    :param hash_size: hash空间大小
    """
    global ID2INDEX, DB
    row, col, data = [], [], []

    # 构建token
    tokens = build_token(DB, db_table, document_id)
    title = document_id.split('@')
    title = title[1] if len(title) > 1 else None

    if not tokens:
        return row, col, data

    # 获得去除了停用词、标点的grams list，每个gram用空格连接，如2gram: ['a','ab','b','bc']
    grams = tokens.ngrams(n=ngram, uncased=True, filter_fn=utils.grams_filter, as_strings=True)
    if title:
        grams.append(title)

    # 对每个gram进行哈希，并统计hashed gram出现次数
    counts = Counter([utils.hash(gram, hash_size) for gram in grams])

    # row为hashed grams作为行号，col为document_id_index作为列号，
    row.extend(counts.keys())
    col.extend([ID2INDEX[document_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(db_table, ngram, hash_size, num_workers=None):
    """ 建立一个gram - document count稀疏矩阵（倒排索引）：M[i, j] = gram i在document j中出现的次数
    :param: db_table, 数据库表名
    :param: ngram, n元文法
    :param: hash_size，hash空间大小
    :param: num_workers，并行处理数
    :return: count_matrix，稀疏矩阵M[i, j] = gram i在document j中出现的次数
    :return: (ID2INDEX, ids)，map{document_id: index}和list[all_document_ids]
    """
    global ID2INDEX
    with Db() as db:
        ids = db.get_ids(db_table)
    ID2INDEX = {document_id: index for index, document_id in enumerate(ids)}

    # initargs中的','必须要，否则传递一个str而不是tuple
    pool = Pool(num_workers, initializer=pool_init)

    logger.info('Mapping...')
    row, col, data = [], [], []
    gap = max(int(len(ids) / 10), 1)  # 每次处理gap个documents
    batches = [ids[i:i + gap] for i in range(0, len(ids), gap)]
    _count = partial(count, ngram, hash_size, db_table)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for batch_row, batch_col, batch_data in pool.imap_unordered(_count, batch):
            row.extend(batch_row)
            col.extend(batch_col)
            data.extend(batch_data)
    pool.close()
    pool.join()

    logger.info('Creating sparse matrix...')
    """ 稀疏矩阵csr压缩    
    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])
    """
    count_matrix = sparse.csr_matrix((data, (row, col)), shape=(hash_size, len(ids)))
    # 消除重复项 by adding them together
    count_matrix.sum_duplicates()
    return count_matrix, (ID2INDEX, ids)


def get_document_freqs(count_matrix):
    """ 得到每个gram在所有documents中出现的次数，在每个document中出现至多1次
    :return freqs: array[hash_size]
    """
    # 将count_matrix中大于0的项变为True再变为1
    binary = (count_matrix > 0).astype(int)
    # 将binary按列相加，得到[hash_size * 1]的矩阵，再变为[hash_size * 1]的array，再变为[hash_size]的array
    freqs = numpy.array(binary.sum(1)).squeeze()
    return freqs


def get_tfidf_matrix(count_matrix):
    """ 将gram - document count稀疏矩阵转换为tfidf矩阵
    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = 每个gram j在document i中出现的次数: [j * i]
    * N = documents数
    * Nt = 每个gram在所有documents中出现的次数: [j]
    """

    # 得到每个gram在所有documents中出现的次数 Nt: array[hash_size]
    Nt = get_document_freqs(count_matrix)
    # N: documents数
    N = count_matrix.shape[1]
    # idf: array[hash_size]
    idf = numpy.log((N - Nt + 0.5) / (Nt + 0.5))
    # 把idf中小于0的项赋为0
    idf[idf < 0] = 0
    # 将idf: array[hash_size]变为对角矩阵: matrix: [hash_size * hash_size]
    """ 
    sparse.diags([1, 2, 3], 0).toarray()
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    """
    idf = sparse.diags(idf, 0)
    # 每个gram在每个document中出现的次数 tf: [hash_size * doc_size]
    tf = count_matrix.log1p()
    # tfidf: [hash_size * doc_size]
    tfidf = idf.dot(tf)
    return tfidf


def build(db_table, tfidf_model_dir, ngram=2, hash_size=int(math.pow(2, 24)), num_workers=None):
    """
    :param db_table, mysql数据库中table名字（documents，baidu）
    :param tfidf_model_dir, 输出模型文件目录
    :param ngram: n元文法的n
    :param hash_size，hash空间大小
    :param num_workers，并行处理数
    """
    logging.info('Counting words...')
    count_matrix, document_dict = get_count_matrix(
        db_table, ngram=ngram, hash_size=hash_size, num_workers=num_workers
    )

    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = get_document_freqs(count_matrix)

    tfidf_model_name = db_table + ('_tfidf_%dgram_%dhash' % (ngram, hash_size))
    tfidf_model_path = os.path.join(tfidf_model_dir, tfidf_model_name)

    logger.info('Saving to %s.npz' % tfidf_model_path)
    metadata = {
        'freqs': freqs,
        'hash_size': hash_size,
        'ngram': ngram,
        'dict': document_dict
    }
    utils.save_tfidf(tfidf_model_path, tfidf, metadata)
    logger.info("TF-IDF Metadata: " + str(metadata))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('tfidf_builder.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--db-table', type=str, default=None)
    parser.add_argument('--tfidf-model-dir', type=str, default=DEFAULTS['tfidf_model_dir'])
    parser.add_argument('--ngram', type=int, default=DEFAULTS['tfidf_ngram'])
    parser.add_argument('--hash-size', type=int, default=DEFAULTS['tfidf_hash_size'])
    parser.add_argument('--num-workers', type=int, default=DEFAULTS['num_workers'])
    args = parser.parse_args()

    args.tfidf_model_dir = os.path.join(DATA_DIR, args.tfidf_model_dir)
    build(
        db_table=args.db_table,
        tfidf_model_dir=args.tfidf_model_dir,
        ngram=args.ngram,
        hash_size=args.hash_size,
        num_workers=args.num_workers
    )
