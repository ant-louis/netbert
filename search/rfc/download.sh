#!/bin/bash

export OUT_DIR=/raid/antoloui/THESIS/Master-thesis/_data/search/rfc

python -W ignore -u tools/download_all.py \
       --outdir $OUT_DIR
