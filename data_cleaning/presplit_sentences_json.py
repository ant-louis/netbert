import sys
import json
import argparse
import time
import os
from multiprocessing import Process

import nltk
nltk.download('punkt')


MIN_WORDS = 2
MAX_WORDS = 200


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/raid/antoloui/Master-thesis/Data/Cleaned/',
                        help="Path of the data directory.")
    parser.add_argument("--infile", type=str,
                        help="Name of the input file.")
    parser.add_argument("--all", type=bool, default=False,
                        help="Create pretraining data for all files.")
    arguments, _ = parser.parse_known_args()
    return arguments


def split_sentences(infile, data_dir):
    """
    """
    in_filename = data_dir + infile
    out_filename = data_dir + 'split_' + infile

    line_seperator = "\n"
    with open(in_filename, 'r') as ifile:
        with open(out_filename, "w") as ofile:
            for doc in ifile.readlines():
                parsed = json.loads(doc)

                # Split text to sentences
                list_sent = []
                for line in parsed['text'].split('\n'):
                    if line != '\n':
                        list_sent.extend(nltk.tokenize.sent_tokenize(line))

                        # If line begins with a number, remove the number   
                        list_sent = [sent.split(maxsplit=1)[1] if (len(sent.split(maxsplit=1))>1 and sent.split(maxsplit=1)[0].isdigit()) else sent for sent in list_sent]

                        # If line begins with a unique special char, remove that char
                        spec_char = set(',?;.:/=+%`¨*$€–-_())°!§\'\"&@#~®†ºπ‡¬≈©◊~∞µ…÷≠<>™^')
                        list_sent = [sent.split(maxsplit=1)[1] if (len(sent.split(maxsplit=1))>1 and len(sent.split(maxsplit=1)[0])==1 and sent.split(maxsplit=1)[0] in spec_char) else sent for sent in list_sent]

                        # Keep only sentences with more than 2 words and less than 200 words
                        list_sent = [sent for sent in list_sent if (len(sent.split())>MIN_WORDS and len(sent.split())<MAX_WORDS)]
    
                # Write to output file
                parsed['text'] = line_seperator.join(list_sent)
                ofile.write(json.dumps(parsed)+'\n')
    print("{} : sentence segmentation done !".format(infile))


def main(args):
    """
    Run processes in parallel if --all=True, otherwise run one process.
    """
    if args.all:
        # Get the file paths
        filenames = [f for f in os.listdir(args.data_dir) if f.startswith('cleaned_') and f.endswith('.json')]
        
        # Instantiating process with arguments
        process_list = [Process(target=split_sentences, args=(f, args.data_dir,)) for f in filenames]
        for i, p in enumerate(process_list):
            print('Process {} is starting...'.format(i+1))
            p.start()
            time.sleep(1)
        # Complete the processes
        for p in process_list:
            p.join()       
    else:
        file = args.infile
        split_sentences(file, args.data_dir)

        

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
