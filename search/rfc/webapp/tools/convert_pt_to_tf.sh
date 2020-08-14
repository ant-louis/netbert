#!/bin/sh

export DIR=/raid/antoloui/THESIS/Master-thesis/_models/
export DEV_FILE=/raid/antoloui/Master-thesis/_data/cleaned/dev.raw
export CACHE=/raid/antoloui/Master-thesis/_cache/
 
python -W ignore -u convert_pytorch_checkpoint_to_tf.py \
    --model_name $DIR/netbert-final \
    --pytorch_model_path $DIR/netbert-final/pytorch_model.bin \
    --tf_cache_dir $DIR/netbert-final_tf
