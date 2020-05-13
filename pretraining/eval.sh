#!/bin/sh

export OUT_DIR=/raid/antoloui/Master-thesis/_models/
export DEV_FILE=/raid/antoloui/Master-thesis/_data/cleaned/dev.raw
export CACHE=/raid/antoloui/Master-thesis/_cache/

python -W ignore -u tools/run_language_modeling.py \
    --model_type=bert \
    --model_name_or_path='bert-base-cased' \
    --do_eval \
    --eval_data_file=$DEV_FILE \
    --output_dir=$OUT_DIR \
    --cache_dir=$CACHE \
    --per_gpu_eval_batch_size=128 \
    --mlm \
    --eval_all_checkpoints
