#!/bin/sh

export DATA_DIR=/raid/antoloui/Master-thesis/search/rfc/webapp/_data/processed/
 
python -W ignore -u tools/convert_data_format.py \
    --data_dir $DATA_DIR
