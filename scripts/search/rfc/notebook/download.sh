#!/bin/bash

export OUT_DIR=$1 #/raid/antoloui/Master-thesis/search/rfc/_data

python -W ignore -u tools/download_all.py \
       --outdir $OUT_DIR
