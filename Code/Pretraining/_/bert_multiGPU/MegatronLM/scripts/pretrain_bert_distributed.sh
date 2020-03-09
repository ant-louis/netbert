#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --batch-size 192 \
       --train-iters 1000000 \
       --train-data cisco_train \
       --valid-data cisco_dev \
       --test-data cisco_test \
       --lazy-loader \
       --presplit-sentences \
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