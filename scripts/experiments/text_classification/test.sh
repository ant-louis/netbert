#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

#export MODEL=bert_base_cased
export MODEL=netbert-final

export MODELS_PATH=/raid/antoloui/Master-thesis/experiments/text_classification/output
export FILE=/raid/antoloui/Master-thesis/_data/classification/cam_query_to_doctype.csv
export LABELS=5
export BS=768

python -W ignore -u train.py \
    --model_name_or_path $MODELS_PATH/$MODEL \
    --do_test \
    --filepath $FILE \
    --num_labels $LABELS \
    --batch_size $BS \
#    --do_compare