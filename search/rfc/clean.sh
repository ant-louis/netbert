#!/bin/bash

export DIR=/raid/antoloui/THESIS/Master-thesis/_data/search/rfc

python -W ignore -u tools/clean_all.py \
       --dirpath $DIR
