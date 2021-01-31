#!/bin/sh

export INDEX=./tools/index.json
export NAME=rfcsearch
 
python -W ignore -u tools/create_index.py \
    --index_file $INDEX \
    --index_name $NAME 
