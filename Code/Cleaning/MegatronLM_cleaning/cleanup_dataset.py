# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ftfy
import json
from langdetect import detect
import numpy as np
import time
import os
import sys
import argparse

MIN_DOCUMENT_LENGTH = 128


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="/raid/antoloui/Master-thesis/Data/Original/13.json",
                        help="Input file.")
    parser.add_argument("--out_file", type=str, default="/raid/antoloui/Master-thesis/Data/Original/cleaned_13.json",
                        help="Output file.")
    arguments, _ = parser.parse_known_args()
    return arguments



def print_progress(prefix, start_time, num_docs, num_written_docs, num_fixed_text,
                   num_non_english_docs, chars_non_english_docs,
                   num_small_docs, chars_small_docs):
    """
    """
    string = prefix + ' | '
    string += 'elapsed time: {:.2f} | '.format(time.time() - start_time)
    string += 'documents: {} | '.format(num_docs)
    string += 'written documents: {} | '.format(num_written_docs)
    string += 'fixed text: {} | '.format(num_fixed_text)
    string += 'non-english: {} | '.format(num_non_english_docs)
    string += 'non-english chars: {} | '.format(chars_non_english_docs)
    string += 'small docs: {} | '.format(num_small_docs)
    string += 'small docs chars: {}'.format(chars_small_docs)
    print(string, flush=True)


    
def filter_corpus(in_filename, out_filename, print_interval=10000):
    """
    """
    print(' > Filtering {}...'.format(in_filename))
    num_docs = 0
    num_written_docs = 0
    num_small_docs = 0
    num_fixed_text = 0
    num_non_english_docs = 0
    chars_non_english_docs = 0
    chars_small_docs = 0
    docs_list = []
    
    start_time = time.time()
    with open(out_filename, 'wb') as f_out:
        with open(in_filename, 'r+') as f_in:
            data = json.load(f_in)  # data is a list of dict of the form: {'text':['...'], 'uri':['...']}
            for doc in data:
                num_docs += 1
                # Extract uri
                doc['uri'] = doc['uri'][0]
                doc['text'] = doc['text'][0]
                    
                # Fix text
                text = ftfy.fix_text(doc['text'])
                if text != doc['text']:
                    num_fixed_text += 1
                doc['text'] = text
                    
                ## Detect language.
                #if detect(text) != 'en':
                #    print('[non-english text]', doc)
                #    num_non_english_docs += 1
                #    chars_non_english_docs += len(text)
                #    continue
                    
                # Skip small documents
                if len(text.split()) < MIN_DOCUMENT_LENGTH:
                    #print('[small document, skipping]:', doc)
                    num_small_docs += 1
                    chars_small_docs += len(text)
                    continue
                        
                # Write to output file
                myjson = json.dumps(doc, ensure_ascii=False)
                f_out.write(myjson.encode('utf-8'))
                f_out.write('\n'.encode('utf-8'))
                num_written_docs += 1
                if num_docs % print_interval == 0:
                    print_progress('[PROGRESS]', start_time, num_docs, num_written_docs,
                                    num_fixed_text, num_non_english_docs,
                                    chars_non_english_docs,
                                    num_small_docs, chars_small_docs)      
                                         
    print_progress('[FINAL]', start_time, num_docs, num_written_docs,
                   num_fixed_text, num_non_english_docs,
                   chars_non_english_docs,
                   num_small_docs, chars_small_docs)
                

if __name__ == "__main__":
    args = parse_arguments()
    filter_corpus(args.in_file, args.out_file)
