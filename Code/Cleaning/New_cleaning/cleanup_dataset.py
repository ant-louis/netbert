import ftfy
import json
from langdetect import detect
import numpy as np
import time
import os
import sys
import re
import argparse
from multiprocessing import Process

MIN_DOCUMENT_LENGTH = 128


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/raid/antoloui/Master-thesis/Data/',
                        help="Path of the data directory.")
    parser.add_argument("--file_id", type=int,
                        help="Id of the input file.")
    parser.add_argument("--all", type=bool, default=False,
                        help="Create pretraining data for all files.")
    arguments, _ = parser.parse_known_args()
    return arguments



def save_result(data_dir, file_id, start_time, num_docs, num_written_docs, num_fixed_text, num_small_docs):
    """
    """
    string = 'Elapsed time: {:.2f} s| '.format(time.time() - start_time)
    string += 'Total documents: {} | '.format(num_docs)
    string += 'Small documents: {} | '.format(num_small_docs)
    string += 'Fixed documents: {} | '.format(num_fixed_text)
    string += 'Written documents: {}'.format(num_written_docs)
    
    filename = data_dir + 'Cleaned/New_cleaning/info_' + str(file_id) + '.txt'
    with open(filename, "w") as text_file:
        text_file.write(string)


    
def filter_corpus(file_id, data_dir):
    """
    """
    in_filename = data_dir + 'Original/' + str(file_id) + '.json'
    out_filename = data_dir + 'Cleaned/New_cleaning/cleaned_' + str(file_id) + '.json'
    
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
            # Load data: data is a list of dict of the form: {'text':['...'], 'uri':['...']}
            data = json.load(f_in)
            for doc in data:
                num_docs += 1
                
                if doc.get('text') is not None:
                    doc['text'] = ' '.join(doc['text'])
                    
                    # Fix text with ftfy
                    text = ftfy.fix_text(doc['text'])
                    
                    # Replace two or more spaces with one
                    text = re.sub('\s{2,}', ' ', text)
                    
                    # Remove sequences of special characters
                    spec_char = set(',?;.:/=+%`¨*$€-_())°!§\'\"&@#~®†ºπ‡¬≈©◊~∞µ…÷≠<>^')
                    text = ' '.join([x for x in text.split() if len(x)<=2 or not all(c in spec_char for c in x)])
                    
                    # Count number of fixed docs
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
                        continue

                    # Write to output file
                    myjson = json.dumps(doc, ensure_ascii=False)
                    f_out.write(myjson.encode('utf-8'))
                    f_out.write('\n'.encode('utf-8'))
                    num_written_docs += 1
                                         
    save_result(data_dir, file_id, start_time, num_docs, num_written_docs, num_fixed_text, num_small_docs)
    

def main(args):
    """
    Run processes in parallel if --all=True, otherwise run one process.
    """
    if args.all:
        # Instantiating process with arguments
        process_list = [Process(target=filter_corpus, args=(i, args.data_dir,)) for i in range(1,14)]
        for i, p in enumerate(process_list):
            print('Process {} is starting...'.format(i+1))
            p.start()
            time.sleep(1)

        # Complete the processes
        for p in process_list:
            p.join()
            
    else:
        i = args.file_id
        filter_corpus(i, args.data_dir)
                

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
