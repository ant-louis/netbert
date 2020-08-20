#!/bin/sh

export DIR=$1 #/raid/antoloui/Master-thesis/search/rfc/webapp/_data/
export FILE=$2 #example.csv
export NAME=rfcsearch
 
python -W ignore -u tools/create_documents.py \
    --data $DIR/$FILE \
    --save $DIR/documents.json \
    --index_name $NAME 
