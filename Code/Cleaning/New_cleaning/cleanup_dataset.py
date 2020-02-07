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
from multiprocessing import Process

MIN_DOCUMENT_LENGTH = 128


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_id", type=int,
                        help="Id of the input file.")
    parser.add_argument("--all", type=bool, default=False,
                        help="Create pretraining data for all files.")
    arguments, _ = parser.parse_known_args()
    return arguments



def save_result(file_id, start_time, num_docs, num_written_docs, num_fixed_text,
                   num_non_english_docs, chars_non_english_docs,
                   num_small_docs, chars_small_docs):
    """
    """
    string = 'elapsed time: {:.2f} | '.format(time.time() - start_time)
    string += 'documents: {} | '.format(num_docs)
    string += 'written documents: {} | '.format(num_written_docs)
    string += 'fixed text: {} | '.format(num_fixed_text)
    string += 'non-english: {} | '.format(num_non_english_docs)
    string += 'non-english chars: {} | '.format(chars_non_english_docs)
    string += 'small docs: {} | '.format(num_small_docs)
    string += 'small docs chars: {}'.format(chars_small_docs)
    
    filename = '/raid/antoloui/Master-thesis/Data/Cleaned/New_cleaning/info_' + str(file_id) + '.txt'
    with open(filename, "w") as text_file:
        text_file.write(string)


    
def filter_corpus(file_id):
    """
    """
    directory = '/raid/antoloui/Master-thesis/Data/'
    in_filename = directory + 'Original/' + str(file_id) + '.json'
    out_filename = directory + 'Cleaned/New_cleaning/' + str(file_id) + '.json'
    
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
                                         
    save_result(file_id, start_time, num_docs, num_written_docs,
                num_fixed_text, num_non_english_docs,
                chars_non_english_docs,
                num_small_docs, chars_small_docs)
    

def main(args):
    """
    """
    """
    Run processes in parallel if --all=True, otherwise run one process.
    """
    if args.all:
        # Instantiating process with arguments
        process_list = [Process(target=filter_corpus, args=(str(i),)) for i in range(1,14)]
        for i, p in enumerate(process_list):
            print('Process {} is starting...'.format(i+1))
            p.start()
            time.sleep(1)

        # Complete the processes
        for p in process_list:
            p.join()
            
    else:
        i = args.file_id
        filter_corpus(i)
                

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
