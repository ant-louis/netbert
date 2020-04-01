#!/bin/bash

export IN_DIR=/raid/antoloui/Master-thesis/Data/Cleaned/
export OUT_DIR=/raid/antoloui/Master-thesis/Data/Embeddings/
export CACHE=/raid/antoloui/Master-thesis/Code/_cache/

export MODEL=/raid/antoloui/Master-thesis/Code/_models/netbert-830000/
export BATCH_SIZE=1408


python -W ignore -u tools/embed_corpus.py \
       --input_dir=$IN_DIR \
       --output_dir=$OUT_DIR \
       --cache_dir=$CACHE \
       --model_name_or_path=$MODEL \
       --batch_size=$BATCH_SIZE \
       --dataparallelmodel |& tee $OUT_DIR/logs.txt
