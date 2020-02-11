export TRAIN_FILE=/raid/antoloui/Master-thesis/Data/Cleaned/train.raw
export OUT_DIR=./output/bert_base/
export DEV_FILE=/raid/antoloui/Master-thesis/Data/Cleaned/dev.raw

python run_language_modeling.py \
    --model_type=bert \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size=14 \
    --num_train_epochs 3 \
    --warmup_steps=1000 \
    --save_steps=500 \
    --save_total_limit=10 \
    --mlm \
    --mlm_probability=0.15 \
    --output_dir=$OUT_DIR \
    --cache_dir=./cache \
    --do_eval \
    --eval_data_file=$DEV_FILE |& tee ./output/base_cased/training_logs.txt


