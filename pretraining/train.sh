#!/bin/sh

export TRAIN_FILE=/raid/antoloui/Master-thesis/Data/Cleaned/train.raw
export OUT_DIR=./output/base_cased/
export DEV_FILE=/raid/antoloui/Master-thesis/Data/Cleaned/dev.raw
export CACHE=../_cache

python tools/run_language_modeling.py \
    --model_type=bert \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size=14 \
    --num_train_epochs=20 \
    --warmup_steps=10000 \
    --save_steps=1000 \
    --save_total_limit=10 \
    --mlm \
    --mlm_probability=0.15 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-6 \
    --learning_rate=5e-05 \
    --output_dir=$OUT_DIR \
    --overwrite_output_dir \
    --cache_dir=$CACHE \
    --do_eval \
    --eval_data_file=$DEV_FILE |& tee ./output/logs/bert_base_cased.txt
