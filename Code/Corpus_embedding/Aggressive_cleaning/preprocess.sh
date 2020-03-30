#!/bin/bash

# Script to extract and preprocess text, 
#     including sanitization, Moses tokenization,
#     and sentence segmentation using NLTK.

# Syntax to run this script:
# ./preprocess <data_path>

set -e

# Check number of arguments
if [ $# -eq 1 ]
then
    echo "Running script ..."
else
    echo "Only 1 argument must be provided!"
    exit 1
fi

# Set language.
lg=en

# Path to raw and processed data
DATA_PATH=$1
DATA_DIR=`dirname $DATA_PATH`
DATA_NAME=`basename $DATA_PATH`
OUTPUT_DIR=$DATA_DIR/aggressive_cleaning/
output=$DATA_NAME.recleaned.nc

# Create folder to save processed data.
mkdir -p $OUTPUT_DIR


# Tokenizer, cleaner and sentence splitter.
CLEANER=./clean_text.py
TOKENIZER=./tokenize.sh
SENT_SPLITTER=./split_sentences.py


function preprocess {

    f=$DATA_PATH
    fo=$OUTPUT_DIR/$output

    echo "Processing $f ..."
        
    # Apply aggressive heuristics to filter data
    if [ ! -f  "$fo" ]; then
        # Clean, tokenize and split sentences
        python $CLEANER -p "$f" -r \
        | grep -P -v '^\s*$' \
        | grep -P '.{50,}' \
        | grep -P -v '[|\t\[\]\{\}]' \
        | grep -P -v '\\{2,}' \
        | grep -P -v '\(en savoir plus\)' \
        | grep -P -v '(?:([-[\](){}><]+ *\w* *[-[\](){}><]+) *\w* *){5,}' \
        | grep -P -o '(^\p{Lu}|(?<=[.!?]\s))\p{Lu}.{50,}(\w\.|\s\!|\s\?)+' \
        | grep -P -v '\d+ Fax|Tel \(' \
        | grep -P -v '[eE]mail|[fF]ax|[pP]hone|[tT]el|[cC]ontact|[i|I]nfo *[@:]+' \
        | grep -P -v '^:' \
        | grep -P -v '(\/ ){3,}|(\/){3,}' \
        | grep -P -v '[A-Za-z0-9]{25,}' \
        | grep -P -v '(\w{4,}\d{2,})|(\d{2,}\w{4,})' \
        | grep -P '^(?!.*(.)\1{5,})' \
        | grep -P '^(?!.*(..)\1{5,})' \
        | grep -v "<br|br/>" \
        | grep -v "^<doc id=" \
        | grep -v "</doc>\$" \
        | grep -P -v 'noinclude|pagequality|user=|\{\{|\}\}|\\|<\/\w+>|\|\w*\|\/>|<\w+>|<section|style=' \
        | perl -CSD -Mutf8 -pe 's/\p{Sk}+|\p{So}+|\p{Cn}+|\p{Co}+|\p{Cs}+|\p{M}+|\p{Lo}+//g' \
        | $TOKENIZER $lg \
        | python $SENT_SPLITTER \
        | grep -P -v '(\w ){10,}' \
        | grep -P -v '(\w |\w\w ){10,}' \
        | grep -P -v '^(\/ [>.*\d])' \
        | grep -P -v '^(: \d+)|^(: [()"-:+])' \
        | grep -P -v '^\s*$' \
        | grep -P '.{50,}' \
        > "$fo"
        echo "Finished cleaning and tokenizing data. Processed files are saved in $OUTPUT."
    else
        echo "Data has already been processed and saved in $OUTPUT."
    fi
}


# Process corpus.
preprocess $DATA_PATH
