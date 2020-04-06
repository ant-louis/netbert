#!/bin/bash

#---------------------------------------------------------------------
# CISCO CORPUS DATA
#---------------------------------------------------------------------
#export IN_DIR=/raid/antoloui/Master-thesis/Data/Cleaned/split/
#export OUT_DIR=/raid/antoloui/Master-thesis/Data/Embeddings/

#---------------------------------------------------------------------
# QA DATA
#---------------------------------------------------------------------
export IN_DIR=/raid/antoloui/Master-thesis/Data/QA/
export OUT_DIR=/raid/antoloui/Master-thesis/Data/QA/embeddings/

#---------------------------------------------------------------------
# MODEL AND PARAMATERS
#---------------------------------------------------------------------
export MODEL=/raid/antoloui/Master-thesis/Code/_models/bert_base_cased/
export CACHE=/raid/antoloui/Master-thesis/Code/_cache/
export BATCH_SIZE=1024

#---------------------------------------------------------------------
# COMMAND
#---------------------------------------------------------------------
python -W ignore -u tools/embed_corpus.py \
       --input_dir $IN_DIR \
       --output_dir $OUT_DIR \
       --cache_dir $CACHE \
       --model_name_or_path $MODEL \
       --batch_size $BATCH_SIZE \
       --dataparallelmodel |& tee $OUT_DIR/logs.txt
