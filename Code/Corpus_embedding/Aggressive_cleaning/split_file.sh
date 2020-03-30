#!/bin/bash

export INFILE=train.raw  # test.raw, dev.raw, train.raw
export OUT_NAME=train   # test, dev, train

export NB_LINES=1000000
export SUFFIX=.raw


split -d -l $NB_LINES $INFILE $OUT_NAME --additional-suffix=$SUFFIX