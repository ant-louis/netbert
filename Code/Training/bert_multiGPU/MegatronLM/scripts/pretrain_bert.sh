#!/bin/bash

RANK=0
WORLD_SIZE=1

python pretrain_bert.py \
       --batch-size 64 \
       --train-iters 1000000 \
       --train-data cisco \
       --lazy-loader \
       --presplit-sentences \
       --resume-dataloader \
       --pretrained-bert \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-base-cased \
       --save model_checkpoints/base_cased \
       --load model_checkpoints/base_cased \
       --save-interval 1000 \
       --tensorboard-dir ./tensorboard \
       --cache-dir cache \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --seq-length 128 \
       --max-preds-per-seq 20 \
       --max-position-embeddings 128 \
       --split 939,40,11 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --fp32-embedding |& tee ./model_checkpoints/base_cased/logs.txt