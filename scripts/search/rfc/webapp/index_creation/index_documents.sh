#!/bin/sh

export DATA_DIR=$1 #/raid/antoloui/Master-thesis/search/rfc/webapp/_data
export DOCUMENTS=$DATA_DIR/documents.json
 
python -W ignore -u tools/index_documents.py \
    --data $DOCUMENTS 
