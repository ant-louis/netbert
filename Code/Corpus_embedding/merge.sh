#!/bin/bash

export DIRPATH=/raid/antoloui/Master-thesis/Data/Embeddings/

python -W ignore -u tools/merge_embeddings.py \
       --input_dir=$DIRPATH \
       --output_dir=$DIRPATH 
