import math

import os

PROJECT_DIR = "/home/wuyinan/Desktop/final/"
# PROJECT_DIR = "/Users/wuyinan/Projects/python/final/"

TQA_DIR = os.path.join(PROJECT_DIR, "tqa/")
DATA_DIR = os.path.join(PROJECT_DIR, "data/")
LOGS_DIR = os.path.join(DATA_DIR, "logs/")

DEFAULTS = {
    'num_workers': 1,
    # ------------------------------------------------------------------------------
    # db_builder 相关参数
    # ------------------------------------------------------------------------------
    'db_host': '10.2.3.83',
    'db_user': 'root',
    'db_password': 'tac',
    'db_database': 'tqa',
    # ------------------------------------------------------------------------------
    # core_nlp_tokenizer 相关参数
    # ------------------------------------------------------------------------------
    'tokenizer_language': 'zh',
    'tokenizer_annotators': ['pos', 'ner', 'lemma'],
    'tokenizer_classpath': 'corenlp/',
    'tokenizer_heap': '5g',
    'tokenizer_timeout': 600000,
    # ------------------------------------------------------------------------------
    # tfidf_builder/tfidf_ranker 相关参数
    # ------------------------------------------------------------------------------
    'tfidf_model_dir': 'models/retriever/',
    'tfidf_ngram': 2,
    'tfidf_hash_size': int(math.pow(2, 24)),
    'tfidf_rank_k': 10,
    'tfidf_rank_strict': False,
    # ------------------------------------------------------------------------------
    # reader/trainer 相关参数
    # ------------------------------------------------------------------------------
    # 环境相关
    'use_cuda': True,
    'gpu_device': 0,
    'gpu_parallel': False,
    'random_seed': 126,  # 为了可重现性
    'batch_size': 10,
    'data_loaders': 10,
    'num_epochs': 40,
    'log_every_num_batches': 50,
    'dev_batch_size': 64,
    # 文件相关
    'reader_model_dir': 'models/reader/',  # 存放日志和模型文件目录
    'model_name': None,  # 如果是中断后继续训练，则需指定: ${language}_YYmmddHHMMSS
    'reader_train_data_path': 'datasets/squad/train-v1.1.txt',  # parsed SQuAD train 数据集
    'reader_dev_data_path': 'datasets/squad/dev-v1.1.txt',  # parsed SQuAD dev 数据集
    'embedded_corpus_path': 'models/embeddings/en.glove.840B.300d.txt',
    'checkpoint': True,  # 中断后继续训练
    'pretrained_model_path': None,
    # 数据预处理相关
    'uncased_question': False,  # 小写化问题
    'uncased_document': False,  # 小写化文档
    'only_use_corpus': True,
    # 模型结构相关
    'rnn_type': 'lstm',
    'embedding_size_w': 300,
    'embedding_size_c': 200,
    'hidden_size': 256,
    'num_layers_d': 3,
    'num_layers_q': 3,
    'kernel_sizes': '[1, 2, 3, 4]',
    'kernel_feature_sizes': '[100, 100, 100, 100]',
    'num_layers_c': 3,
    'use_reattention': False,
    'reattention_round': 2,
    'gamma_e': 0.5,
    'gamma_b': 0.5,
    # 特征选择相关
    'use_align': True,
    'use_exact_match': True,
    'use_pos': True,
    'use_ner': True,
    'use_lemma': True,
    'use_tf': True,
    # 优化相关
    'tune_top_k': 10000,
    'optimizer': 'adamax',
    'learning_rate': 0.002,
    'momentum': 0,
    'weight_decay': 0,
    'embedding_dropout_rate': 0.3,
    'rnn_dropout_rate': 0.3,
    'concat_rnn_layers': True,
    'dropout_rnn_output': True,
    'grad_clipping': 10,
    'max_span_length': 20,
    'reader_validation_metric': 'exact_match',
    'cnn_param_init': 0,
    # ------------------------------------------------------------------------------
    # reader/predictor 相关参数
    # ------------------------------------------------------------------------------
    'document': None,
    'question': None,
    'top_k_answers': 5,
    # ------------------------------------------------------------------------------
    # classifier/trainer 相关参数
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # reuser 相关参数
    # ------------------------------------------------------------------------------
    'embedded_corpus_bin_path': 'models/embeddings/cc.zh.300.bin',
    'threshold': 0.6,
    # ------------------------------------------------------------------------------
    # server 相关参数
    # ------------------------------------------------------------------------------
    'server_log_path': 'models/',
    'online': True,
}
