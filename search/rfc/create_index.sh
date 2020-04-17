#!/bin/bash

INDIR=/raid/antoloui/Master-thesis/_data/search/rfc/embeddings/
OUTDIR=/raid/antoloui/Master-thesis/_data/search/rfc/index/
N_GPU=8
METHOD=cos  #l2, ip, cos


python -W ignore -u tools/create_faiss_index.py \
       --input_dir $INDIR \
       --output_dir $OUTDIR \
       --n_gpu $N_GPU \
       --method $METHOD
