#!/bin/bash

#---------------------------------------------------------------------
# DATA DIRECTORIES
#---------------------------------------------------------------------
export IN_DIR=/raid/antoloui/Master-thesis/_data/search/rfc/processed/
export OUT_DIR=/raid/antoloui/Master-thesis/_data/search/rfc/embeddings/

#---------------------------------------------------------------------
# MODEL AND PARAMATERS
#---------------------------------------------------------------------
export MODEL=/raid/antoloui/Master-thesis/_models/netbert-final/
export CACHE=/raid/antoloui/Master-thesis/_cache/
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
