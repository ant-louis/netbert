#!/bin/bash

export MODEL=netbert-1880000
export MODELS_PATH=/raid/antoloui/Master-thesis/Code/Extrinsic_evaluation/Classification/output/
export EVAL_FILE=/raid/antoloui/Master-thesis/Code/Extrinsic_evaluation/Classification/output/bert_base_cased/eval_right_preds.csv

python -W ignore -u train.py \
    --model_name_or_path $MODELS_PATH/$MODEL \
    --do_eval \
    --eval_filepath $EVAL_FILE \
    --gpu_id 0 \
