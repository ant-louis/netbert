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
       --pretrained-bert \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 6 \
       --seq-length 128 \
       --max-preds-per-seq 20 \
       --max-position-embeddings 128 \
       --train-iters 1000 \
       --save models/base_cased/bert_model \
       --use-tfrecords \
       --train-data /raid/antoloui/Master-thesis/Data/bert/L128/tf_examples.tfrecord13 \
       --valid-data /raid/antoloui/Master-thesis/Data/bert/L128/tf_examples.tfrecord13 \
       --test-data /raid/antoloui/Master-thesis/Data/bert/L128/tf_examples.tfrecord13 \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-base-cased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 2e-5 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --fp32-layernorm \
       --fp32-embedding |& tee logs/pretrain_bert_tfrecords_distributed.txt