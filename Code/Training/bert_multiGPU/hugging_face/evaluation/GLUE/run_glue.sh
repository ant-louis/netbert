export GLUE_DIR=./GLUE_data
export TASK_NAME=MRPC #CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./output/Eval/$TASK_NAME/ |& tee ./output/Eval/$TASK_NAME/logs.txt