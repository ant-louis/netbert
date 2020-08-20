#!/bin/sh

export DIR=/raid/antoloui/Master-thesis/search/rfc/webapp/_data/
export FILE=example.csv
export NAME=rfcsearch
 
python -W ignore -u tools/create_documents.py \
    --data $DIR/$FILE \
    --save $DIR/documents.json \
    --index_name $NAME 
