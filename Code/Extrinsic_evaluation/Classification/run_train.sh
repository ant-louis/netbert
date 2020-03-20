#!/bin/bash

export MODEL=/raid/antoloui/Master-thesis/Code/_models/netbert-1027000
export TRAIN_FILE=/raid/antoloui/Master-thesis/Data/Classification/cam_query_to_doctype.csv
export EPOCHS=4
export BATCHES=256
export LOGS=./output/logs/logs_netbert.txt

python -W ignore -u train.py \
    --model_name_or_path $MODEL \
    --filepath $TRAIN_FILE \
    --num_epochs $EPOCHS \
    --batch_size $BATCHES \
    --balanced |& tee $LOGS
