#!/bin/bash

export TRAIN_FILE=/raid/antoloui/Master-thesis/Data/Classification/cam_query_to_doctype.csv
export MODEL=bert-base-cased   #../models/netbert/checkpoint-1027000/
export EPOCHS=4
export OUT_DIR=./output/finetuned_bertbase/  #./output/finetuned_netbert/ 
export GPU=1   #0
export LOGS=./output/logs/logs_bertbase.txt   #./output/logs/logs_netbert.txt

python -W ignore -u train.py \
    --filepath $TRAIN_FILE \
    --model_name_or_path $MODEL \
    --num_epochs $EPOCHS \
    --output_dir $OUT_DIR \
    --gpu_id $GPU |& tee $LOGS
