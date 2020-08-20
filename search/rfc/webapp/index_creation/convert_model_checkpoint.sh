#!/bin/sh

export DIR=$1 #/raid/antoloui/THESIS/Master-thesis/_models/netbert
 
python -W ignore -u tools/convert_model_checkpoint.py \
    --model_name $DIR \
    --pytorch_model_path $DIR/pytorch_model.bin \
    --tf_cache_dir $DIR/tensorflow
