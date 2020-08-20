#!/bin/sh

export DIR=/raid/antoloui/THESIS/Master-thesis/_models/netbert
 
python -W ignore -u tools/convert_pytorch_checkpoint_to_tf.py \
    --model_name $DIR \
    --pytorch_model_path $DIR/pytorch_model.bin \
    --tf_cache_dir $DIR/tensorflow
