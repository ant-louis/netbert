#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

#export MODEL=bert_base_cased
export MODEL=netbert-final

export MODELS_PATH=/raid/antoloui/Master-thesis/_models/
export TRAIN_FILE=/raid/antoloui/Master-thesis/_data/classification/cam_query_to_doctype.csv
export LABELS=5

export EPOCHS=6
export BATCHES=32  #16, 32
export LR=2e-5  #5e-5, 3e-5, 2e-5

python -W ignore -u train.py \
    --model_name_or_path $MODELS_PATH/$MODEL \
    --do_train \
    --do_val \
    --training_filepath $TRAIN_FILE \
    --num_labels $LABELS \
    --num_epochs $EPOCHS \
    --batch_size $BATCHES \
    --learning_rate $LR
