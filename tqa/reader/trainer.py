import argparse
import json
import logging

import time

import torch
import os

from torch.utils.data import DataLoader

from tqa import DEFAULTS, DATA_DIR
from tqa.reader import utils
from tqa.reader.data import ReaderDataset, ReaderSampler

logger = logging.getLogger()


def set_logger(log_file_path, checkpoint):
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(log_format)
    logger.addHandler(console)
    if log_file_path:
        if checkpoint:
            file = logging.FileHandler(log_file_path, 'a')
        else:
            file = logging.FileHandler(log_file_path, 'w')
        file.setFormatter(log_format)
        logger.addHandler(file)


def prepare_args(parser):
    parser.register('type', 'bool', utils.str2bool)
    parser.register('type', 'list', utils.str2int_list)

    # ------------------------------------------------------------------------------
    # 环境相关
    # ------------------------------------------------------------------------------
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--language', type=str, default=DEFAULTS['tokenizer_language'],
                         help='模型语言')
    runtime.add_argument('--use-cuda', type='bool', default=DEFAULTS['use_cuda'],
                         help='是否使用GPU训练模型')
    runtime.add_argument('--gpu-device', type=int, default=DEFAULTS['gpu_device'],
                         help='如果使用GPU训练，指定GPU设备')
    runtime.add_argument('--gpu-parallel', type='bool', default=DEFAULTS['gpu_parallel'],
                         help='是否使用多个指定的GPU')
    runtime.add_argument('--random-seed', type=int, default=DEFAULTS['random_seed'],
                         help='随机数种子')
    runtime.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'],
                         help='训练batch大小')
    runtime.add_argument('--data-loaders', type=int, default=DEFAULTS['data_loaders'],
                         help='数据加载子进程个数')
    runtime.add_argument('--num-epochs', type=int, default=DEFAULTS['num_epochs'],
                         help='训练数据迭代次数')
    runtime.add_argument('--log-every-num-batches', type=int, default=DEFAULTS['log_every_num_batches'],
                         help='在每个epoch中训练每n个batches打印一次log')
    runtime.add_argument('--dev-batch-size', type=int, default=DEFAULTS['dev_batch_size'],
                         help='测试batch大小')
    # ------------------------------------------------------------------------------
    # 文件相关
    # ------------------------------------------------------------------------------
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=DEFAULTS['reader_model_dir'],
                       help='用于存放模型、checkpoints和日志的目录')
    files.add_argument('--model-name', type=str, default='',
                       help='唯一的模型文件标识 (.mdl, .txt, .checkpoint)')
    files.add_argument('--train-data-path', type=str, default=DEFAULTS['reader_train_data_path'],
                       help='parsed SQuAD 训练数据文件路径')
    files.add_argument('--dev-data-path', type=str, default=DEFAULTS['reader_dev_data_path'],
                       help='parsed SQuAD 测试数据文件路径')
    files.add_argument('--embedded-corpus-path', type=str, default=DEFAULTS['embedded_corpus_path'],
                       help='预训练好的embedded语料库')
    files.add_argument('--checkpoint', type='bool', default=DEFAULTS['checkpoint'],
                       help='是否在每次epoch结束时保存模型和optimizer')
    files.add_argument('--pretrained-model-path', type=str, default=DEFAULTS['pretrained_model_path'],
                       help='预训练好的模型路径，用于模型状态初始化')
    # ------------------------------------------------------------------------------
    # 数据预处理相关
    # ------------------------------------------------------------------------------
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=DEFAULTS['uncased_question'],
                            help='是否将问题中的单词小写化')
    preprocess.add_argument('--uncased-document', type='bool', default=DEFAULTS['uncased_document'],
                            help='是否将文档中的单词小写化')
    preprocess.add_argument('--only-use-corpus', type='bool', default=DEFAULTS['only_use_corpus'],
                            help='是否只使用embedded_corpus_path中训练好的词向量')
    # ------------------------------------------------------------------------------
    # 模型结构相关
    # ------------------------------------------------------------------------------
    model_arch = parser.add_argument_group('Model Architecture')
    model_arch.add_argument('--rnn-type', type=str, default=DEFAULTS['rnn_type'],
                            help='RNN模型类型: lstm, gru, rnn')
    model_arch.add_argument('--embedding-size-w', type=int, default=DEFAULTS['embedding_size_w'],
                            help='如果embedded_corpus_path没有给则需指定单词embedding向量大小')
    model_arch.add_argument('--embedding-size-c', type=int, default=DEFAULTS['embedding_size_c'],
                            help='字符embedding向量大小')
    model_arch.add_argument('--hidden-size', type=int, default=DEFAULTS['hidden_size'],
                            help='RNN单元隐含层向量大小')
    model_arch.add_argument('--num-layers-d', type=int, default=DEFAULTS['num_layers_d'],
                            help='RNN document encoder层数')
    model_arch.add_argument('--num-layers-q', type=int, default=DEFAULTS['num_layers_q'],
                            help='RNN question encoder层数')
    model_arch.add_argument('--kernel-sizes', type='list', default=DEFAULTS['kernel_sizes'],
                            help='卷积核大小')
    model_arch.add_argument('--kernel-feature-sizes', type='list', default=DEFAULTS['kernel_feature_sizes'],
                            help='卷积核个数')
    model_arch.add_argument('--num-layers-c', type=int, default=DEFAULTS['num_layers_c'],
                            help='CNN Linear层数')
    model_arch.add_argument('--use-reattention', type='bool', default=DEFAULTS['use_reattention'],
                            help='使用多层attention')
    model_arch.add_argument('--reattention-round', type=int, default=DEFAULTS['reattention_round'],
                            help='多层attention数')
    model_arch.add_argument('--gamma-e', type=float, default=DEFAULTS['gamma_e'],
                            help='gamma_e')
    model_arch.add_argument('--gamma-b', type=float, default=DEFAULTS['gamma_b'],
                            help='gamma_b')

    # ------------------------------------------------------------------------------
    # 特征选择相关
    # ------------------------------------------------------------------------------
    features = parser.add_argument_group('Features Select')
    features.add_argument('--use-align', type='bool', default=DEFAULTS['use_align'],
                          help='是否使用对齐特征')
    features.add_argument('--use-exact-match', type='bool', default=DEFAULTS['use_exact_match'],
                          help='是否使用0/1特征：完全匹配问题中的单词（原形式origin、小写形式uncased、词元形式lemma）')
    features.add_argument('--use-pos', type='bool', default=DEFAULTS['use_pos'],
                          help='是否使用词性pos特征')
    features.add_argument('--use-ner', type='bool', default=DEFAULTS['use_ner'],
                          help='是否使用实体ner特征')
    features.add_argument('--use-lemma', type='bool', default=DEFAULTS['use_lemma'],
                          help='是否使用词元lemma特征')
    features.add_argument('--use-tf', type='bool', default=DEFAULTS['use_tf'],
                          help='是否使用词频tf特征')
    # ------------------------------------------------------------------------------
    # 优化相关
    # ------------------------------------------------------------------------------
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--tune-top-k', type=int, default=DEFAULTS['tune_top_k'],
                       help='Backprop through only the top N question words')
    optim.add_argument('--optimizer', type=str, default=DEFAULTS['optimizer'],
                       help='Optimizer: sgd, adamax')
    optim.add_argument('--learning-rate', type=float, default=DEFAULTS['learning_rate'],
                       help='学习速率，只用于SGD Optimizer')
    optim.add_argument('--momentum', type=float, default=DEFAULTS['momentum'],
                       help='Momentum factor')
    optim.add_argument('--weight-decay', type=float, default=DEFAULTS['weight_decay'],
                       help='Weight decay factor')
    optim.add_argument('--embedding-dropout-rate', type=float, default=DEFAULTS['embedding_dropout_rate'],
                       help='Word embeddings的dropout率')
    optim.add_argument('--rnn-dropout-rate', type=float, default=DEFAULTS['rnn_dropout_rate'],
                       help='RNN输入的dropout率')
    optim.add_argument('--concat-rnn-layers', type='bool', default=DEFAULTS['concat_rnn_layers'],
                       help='是否连接每一层的output')
    optim.add_argument('--dropout-rnn-output', type='bool', default=DEFAULTS['dropout_rnn_output'],
                       help='是否dropout最终的RNN output')
    optim.add_argument('--grad-clipping', type=float, default=DEFAULTS['grad_clipping'],
                       help='梯度裁剪阈值')
    optim.add_argument('--max-span-length', type=int, default=DEFAULTS['max_span_length'],
                       help='预测答案时最长考虑跨度大小')
    optim.add_argument('--validation-metric', type=str, default=DEFAULTS['reader_validation_metric'],
                       help='选择模型的评估方法')
    optim.add_argument('--cnn-param-init', type=float, default=DEFAULTS['cnn_param_init'],
                       help='Conv层参数初始化区间[-cnn_param_init, cnn_param_init]')

    args = parser.parse_args()
    args.model_dir = os.path.join(DATA_DIR, args.model_dir)
    args.train_data_path = os.path.join(DATA_DIR, args.train_data_path)
    args.dev_data_path = os.path.join(DATA_DIR, args.dev_data_path)
    args.embedded_corpus_path = os.path.join(DATA_DIR, args.embedded_corpus_path) if args.embedded_corpus_path else None
    args.pretrained_model_path = os.path.join(DATA_DIR, args.pretrained_model_path) \
        if args.pretrained_model_path else None
    return args


def train(args):
    # 如果是中断后继续训练，则需指定model_name: ${language}_${rnn_type}_${hidden_size}_${tune_top_k}_YYmmddHHMMSS
    if not args.model_name:
        args.model_name = ("%s_%s" % (args.language, time.strftime("%Y%m%d_%H%M%S")))
    # ------------------------------------------------------------------------------
    # 日志记录相关
    # ------------------------------------------------------------------------------
    log_file_path = os.path.join(args.model_dir, args.model_name + '.log')

    set_logger(log_file_path, args.checkpoint)
    # ------------------------------------------------------------------------------
    # 环境准备相关
    # ------------------------------------------------------------------------------
    logger.info('-' * 100)
    use_cuda = args.use_cuda and torch.cuda.is_available()

    utils.set_environment(use_cuda, args.gpu_device, args.random_seed)
    # ------------------------------------------------------------------------------
    # 数据处理相关
    # ------------------------------------------------------------------------------
    logger.info('-' * 100)
    # train_examples和dev_examples区别是是否skip_no_answer
    train_examples = utils.load_parsed_data(
        args.train_data_path, args.uncased_question, args.uncased_document, skip_no_answer=True
    )
    logger.info('Number of train data = %d' % len(train_examples))

    dev_examples = utils.load_parsed_data(
        args.dev_data_path, args.uncased_question, args.uncased_document, skip_no_answer=True
    )
    logger.info('Number of dev data = %d' % len(dev_examples))
    # ------------------------------------------------------------------------------
    # 模型建立相关
    # ------------------------------------------------------------------------------
    logger.info('-' * 100)
    model_path = os.path.join(args.model_dir, args.model_name + '.mdl')
    model_checkpoint_path = model_path + '.checkpoint'

    start_epoch = 0  # 如果从checkpoint继续训练，则start_epoch可能不为0
    # 如果checkpoint=True并且中断模型文件存在，则继续训练
    if args.checkpoint and os.path.isfile(model_path + '.checkpoint'):
        logger.info('Found a checkpoint...')
        model, start_epoch = utils.load_checkpoint(model_checkpoint_path)
    else:
        # 从以前训练好的模型开始训练，模型状态初始化使用训练好的模型
        if args.pretrained_model_path:
            logger.info('Training model from pretrained model...')
            model = utils.load_model(args.pretrained_model_path, new_args=None)
            logger.info('Expanding dictionary for new data...')
            # 得到examples和glove词典中都出现的单词集合
            valid_words = utils.load_valid_words(args, train_examples + dev_examples)
            new_chars = utils.load_chars(train_examples + dev_examples)
            # 扩展词典：原glove词典中的词 + train/dev数据集中的词
            added_words, added_chars = model.expand_dictionary(valid_words, new_chars)
            # 将glove词典中存在的added_words的词向量加载到embedding层的data中
            if args.embedded_corpus_path:
                model.load_embeddings(added_words, args.embedded_corpus_path)

        # 从头开始训练，模型状态初始化使用随机方式
        else:
            logger.info('Training model from scratch...')
            # 建立模型：建立训练集特征名称索引、建立Dictionary对象、初始化模型、加载embeddings
            model = utils.init_model_from_scratch(args, train_examples, dev_examples)

        # 微调问题中出现频率top_k的单词的embeddings
        if args.tune_top_k > 0:
            logger.info('-' * 100)
            logger.info('Counting top %d most frequent question words' % args.tune_top_k)
            top_question_words = utils.get_top_question_words(train_examples, model.words, args.tune_top_k)
            for word in top_question_words[:5]:
                logger.info(word)
            logger.info('...')
            for word in top_question_words[-6:-1]:
                logger.info(word)
            # top_question_words ~= [('a', 5), ('b', 4), ('c', 3)]
            model.tune_embeddings([word[0] for word in top_question_words])

        # 初始化optimizer
        model.init_optimizer()

    logger.info('Arguments:\n%s' % ' '.join([str(k) + "=" + str(v) for k, v in vars(args).items()]))
    # logger.info('Arguments:\n%s' % json.dumps(vars(model.args), indent=4, sort_keys=True))

    if use_cuda:
        model.cuda()

    if args.gpu_parallel:
        model.gpu_parallelize()
    # ------------------------------------------------------------------------------
    # 数据迭代相关
    # ------------------------------------------------------------------------------
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = ReaderDataset(train_examples, model, single_answer=True)
    # 采样器中的example indices顺序根据example中的文档长度和问题长度排序
    train_sampler = ReaderSampler(train_dataset.lengths(), args.batch_size, shuffle=True)
    # collate_fn把每个batch中的examples整理为tensor（一般使用默认调用default_collate(batch)）
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_loaders,
        collate_fn=utils.batchify,
        pin_memory=use_cuda,
    )
    dev_dataset = ReaderDataset(dev_examples, model, single_answer=False)
    dev_sampler = ReaderSampler(dev_dataset.lengths(), args.dev_batch_size, shuffle=False)
    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=args.dev_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_loaders,
        collate_fn=utils.batchify,
        pin_memory=use_cuda,
    )
    # ------------------------------------------------------------------------------
    # 开始训练
    # ------------------------------------------------------------------------------
    logger.info('-' * 100)
    logger.info('Starting training...')
    train_states = {'timer': utils.Timer(), 'epoch': 0, 'best_validation': 0}
    for epoch in range(start_epoch, args.num_epochs):
        train_states['epoch'] = epoch

        # 开始训练
        train_loss = utils.AverageMeter()
        epoch_timer = utils.Timer()

        for batch_index, examples_in_batch in enumerate(train_loader):
            train_loss.update(*model.update(examples_in_batch))

            if batch_index % args.log_every_num_batches == 0:
                logger.info(
                    'train: Epoch = %d | iter = %d/%d | ' % (train_states['epoch'], batch_index, len(train_loader)) +
                    'loss = %.2f | elapsed time = %.2f (s)' % (train_loss.average, train_states['timer'].total_time())
                )
                train_loss.reset()

        logger.info('train: Epoch %d done. elapsed time = %.2f (s)' % (train_states['epoch'], epoch_timer.total_time()))
        # 保存检查点
        if args.checkpoint:
            model.checkpoint(model_checkpoint_path, train_states['epoch'] + 1)

        # 每个epoch结束后评估
        utils.validate(train_loader, model, train_states, type='train')
        result = utils.validate(dev_loader, model, train_states, type='dev')

        # 保存最好的评估结果的模型
        if result[args.validation_metric] > train_states['best_validation']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.validation_metric, result[args.validation_metric], train_states['epoch'], model.updates))
            model.save(model_path)
            train_states['best_validation'] = result[args.validation_metric]
        # 如果准确率不上升，则调低学习率
        # else:
        #     for param_group in model.optimizer.param_groups:
        #         logger.info('Adjust learning rate from {old_lr} to {new_lr}'.format(
        #             old_lr=param_group['lr'],
        #             new_lr=param_group['lr'] * 0.5
        #         ))
        #         param_group['lr'] = param_group['lr'] * 0.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tqa Trainer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = prepare_args(parser)
    train(args)
