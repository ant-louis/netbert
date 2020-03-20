#!/bin/bash

export MODEL=bert_base_cased
#export MODEL=netbert-300000
#export MODEL=netbert-830000
#export MODEL=netbert-1027000
#export MODEL=netbert-1880000

export MODELS_PATH=/raid/antoloui/Master-thesis/Code/_models/
export TRAIN_FILE=/raid/antoloui/Master-thesis/Data/Classification/cam_query_to_doctype.csv
export EPOCHS=4
export BATCHES=256

python -W ignore -u train.py \
    --model_name_or_path $MODELS_PATH/$MODEL \
    --filepath $TRAIN_FILE \
    --num_epochs $EPOCHS \
    --batch_size $BATCHES \
    --balanced |& tee ./output/$MODEL/training_logs.txt
