#!/bin/bash

export DATA_DIR=$1 #/raid/antoloui/Master-thesis/search/rfc/_data

python -W ignore -u tools/clean_all.py \
       --dirpath $DATA_DIR
