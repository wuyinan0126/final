#!/bin/bash

# ------------------------------------------------------------------------------
# 环境准备相关
# ------------------------------------------------------------------------------
pip install -r ./requirements.txt
export PYTHONPATH=/home/wuyinan/Desktop/final
# ------------------------------------------------------------------------------
# retriever 相关
# ------------------------------------------------------------------------------
# 测试tokenizer
python retriever/tokenizer.py \
    --heap 5g \
    --text ""
# ------------------------------------------------------------------------------
# wiki documents retriever相关
# 爬取Wiki => 994014个词条
cd ~/Desktop/
mkdir wiki
python2 wikiextractor-master/WikiExtractor.py \
    --processes 32 \
    --output ./wiki/ \
    --bytes 1M \
    --json \
    ./zhwiki-20180301-pages-articles.xml.bz2

# 繁体转简体
cd ~/Desktop/wiki
for dir in AA AB AC AD AE AF AG AH AI AJ AK AL; do
    mkdir -p ./wiki/${dir}
    for (( i=0; i<=99; i++ )); do
        if [ ${#i} -lt 2 ]; then
            file=wiki_0$i
        else
            file=wiki_$i
        fi
        if [ -f ./${dir}/${file} ]; then
#            echo ./${dir}/${file}
            opencc -i ./${dir}/${file} -o ./wiki/${dir}/${file}.json -c zht2zhs.ini
        fi
    done
done

# 建立documents wiki db => 983502/994014
python retriever/db_builder.py \
    --documents-dir documents/wiki/ \
    --tokenizer-heap 5g \
    --num-workers 10

# 建立tfidf wiki文件
python retriever/tfidf_builder.py \
    --db-table wiki \
    --tfidf-model-dir models/retriever/ \
    --num-workers 24

# 测试tfidf course rank
python retriever/tfidf_ranker.py \
    --tfidf-model-path models/retriever/wiki_tfidf_2gram_16777216hash.npz \
    --k 5 \
    --query '什么是数组？'
# ------------------------------------------------------------------------------
# course documents retriever相关
# 建立documents course db
python retriever/db_builder.py \
    --documents-dir documents/course/ \
    --tokenizer-heap 5g \
    --num-workers 4

# 建立tfidf course文件
python retriever/tfidf_builder.py \
    --db-table course \
    --tfidf-model-dir models/retriever/ \
    --num-workers 12

# 测试tfidf course rank
python retriever/tfidf_ranker.py \
    --tfidf-model-path models/retriever/course_tfidf_2gram_16777216hash.npz \
    --k 5 \
    --query '什么是广义表？'
# ------------------------------------------------------------------------------
# reader 相关
# ------------------------------------------------------------------------------
# 中文reader相关
# 解析数据集 [中文WebQA train数据集] => 140897行
python reader/data_parser.py \
    --language zh \
    --dataset-path datasets/webqa/train.json \
    --num-workers 8

# 解析数据集 [中文WebQA valid数据集] => 3018行
python reader/data_parser.py \
    --language zh \
    --dataset-path datasets/webqa/valid.json \
    --num-workers 8

# 训练数据集 [从头开始训练]
python reader/trainer.py \
    --language zh \
    --use-cuda True \
    --gpu-device 0 \
    --checkpoint False \
    --train-data-path datasets/webqa/train.txt \
    --dev-data-path datasets/webqa/valid.txt \
    --embedded-corpus-path models/embeddings/cc.zhs.300.vec
#    --embedded-corpus-path models/embeddings/zh.fasttext.wiki.300d.txt

# 训练数据集 [从已有模型开始训练]
python reader/trainer.py \
    --language zh \
    --use-cuda True \
    --gpu-device 0 \
    --checkpoint False \
    --train-data-path datasets/webqa/train.txt \
    --dev-data-path datasets/webqa/valid.txt \
    --embedded-corpus-path models/embeddings/cc.zhs.300.vec \
    --pretrained-model-path models/reader/zh_lstm_256h_10000k_20180314_004818.mdl

# 训练数据集 [恢复检查点]

# ------------------------------------------------------------------------------
# 英文reader相关
# 解析数据集 [SQuAD train数据集] => 87599行
python reader/data_parser.py \
    --language en \
    --dataset-path datasets/squad/train-v1.1.json \
    --num-workers 12

# 解析数据集 [SQuAD dev数据集] => 10570行
python reader/data_parser.py \
    --language en \
    --dataset-path datasets/squad/dev-v1.1.json \
    --num-workers 12

# 训练数据集 [从头开始训练] 英文
python reader/trainer.py \
    --language en \
    --use-cuda True \
    --gpu-device 0 \
    --checkpoint False \
    --train-data-path datasets/squad/train-v1.1.txt \
    --dev-data-path datasets/squad/dev-v1.1.txt \
    --embedded-corpus-path models/embeddings/en.glove.840B.300d.txt
# ------------------------------------------------------------------------------
# 给定问题和文档预测答案
python reader/predictor.py \
    --model-path models/reader/reader_20180314_184532.mdl \
    --document "小明有双蓝色的鞋子，一件白色的毛衣和一只红色的羊。" \
    --question "小明的毛衣是什么颜色的？" \
    --top-k-answers 5 \
    --use-cuda True \
    --gpu-device 0 \
    --embedded-corpus-path models/embeddings/zh.fasttext.wiki.300d.txt
# ------------------------------------------------------------------------------
# classifier 相关
# ------------------------------------------------------------------------------
# 中文classifier相关
# 解析数据集 [posts数据集] => 行

# 训练数据集 [从头开始训练]
python classifier/trainer.py \
    --language zh \
    --use-cuda True \
    --gpu-device 0 \
    --checkpoint False \
    --num-classes 3 \
    --tune-top-k 0 \
    --train-data-path datasets/posts/test.txt \
    --dev-data-path datasets/posts/test.txt \
    --embedded-corpus-path models/embeddings/zh.fasttext.wiki.300d.txt
# ------------------------------------------------------------------------------
# server 相关
# ------------------------------------------------------------------------------
python server.py \
    --tfidf-model-paths '[models/retriever/wiki_tfidf_2gram_16777216hash.npz, models/retriever/course_tfidf_2gram_16777216hash.npz]' \
    --tfidf-rank-k 1 \
    --top-k-answers 1 \
    --reader-model-path models/reader/reader_20180525_103013.mdl \
    --use-cuda False \
    --num-workers 12 \
    --embedded-corpus-path models/embeddings/cc.zhs.300.vec \
    --embedded-corpus-bin-path models/embeddings/cc.zh.300.bin \
    --threshold 0.6
