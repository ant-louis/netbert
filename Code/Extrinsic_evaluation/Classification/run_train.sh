#!/bin/bash

export TRAIN_FILE=/raid/antoloui/Master-thesis/Data/Classification/cam_query_to_doctype.csv
export MODEL=../models/netbert/checkpoint-1027000/   #bert-base-cased
export EPOCHS=4
export OUT_DIR=./output/finetuned_netbert/   #./output/finetuned_bertbase/
export GPU=0   #1
export LOGS=./output/logs/logs_netbert.txt   #./output/logs/logs_bertbase.txt

python -W ignore -u train.py \
    --filepath $TRAIN_FILE \
    --model_name_or_path $MODEL \
    --num_epochs $EPOCHS \
    --output_dir $OUT_DIR \
    --gpu_id $GPU |& tee $LOGS
