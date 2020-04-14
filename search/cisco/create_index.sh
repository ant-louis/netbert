#!/bin/bash

INDIR=/raid/antoloui/Master-thesis/_data/embeddings/
OUTDIR=/raid/antoloui/Master-thesis/_data/search/cisco/
N_GPU=8
METHOD=ip


python -W ignore -u tools/create_faiss_index.py \
       --input_dir $INDIR \
       --output_dir $OUTDIR \
       --n_gpu $N_GPU \
       --method $METHOD
