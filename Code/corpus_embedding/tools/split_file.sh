#!/bin/bash

export DIRPATH=/raid/antoloui/Master-thesis/Data/Cleaned
export FILENAME=dev # test, dev, train

export INFILE=$DIRPATH/$FILENAME.raw
export OUT_NAME=$DIRPATH/$FILENAME   

export NB_LINES=1000000
export SUFFIX=.raw


split -d -l $NB_LINES $INFILE $OUT_NAME --additional-suffix=$SUFFIX