#!/bin/sh

export DATA=../_data/example.csv
export NAME=rfcsearch
 
python -W ignore -u tools/create_documents.py \
    --data $DATA \
    --index_name $NAME 
