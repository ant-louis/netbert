import os
from tqdm import tqdm
import argparse


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



def run(args):
    """
    """
    directory = '/home/antoloui/Master-thesis/Data/'
    
    if args.all:
        for i in tqdm(range(1,14)):
            command = '''\
                python create_pretraining_data.py \
                --input_file={in_filepath} \
                --output_file={out_filepath} \
                --vocab_file=./models/base_cased/vocab.txt \
                --do_lower_case=False \
                --max_seq_length=128 \
                --max_predictions_per_seq=20 \
                --masked_lm_prob=0.15 \
                --random_seed=12345 \
                --dupe_factor=5
            '''.format(in_filepath=directory+'Preprocessed/text_'+str(i)+'.txt', out_filepath=directory+'bert/tf_examples.tfrecord'+str(i))
            os.system(command)
    
    else:
        i = args.file_id
        command = '''\
            python create_pretraining_data.py \
            --input_file={in_filepath} \
            --output_file={out_filepath} \
            --vocab_file=./models/base_cased/vocab.txt \
            --do_lower_case=False \
            --max_seq_length=128 \
            --max_predictions_per_seq=20 \
            --masked_lm_prob=0.15 \
            --random_seed=12345 \
            --dupe_factor=5
            '''.format(in_filepath=directory+'Preprocessed/text_'+str(i)+'.txt', out_filepath=directory+'bert/tf_examples.tfrecord'+str(i))
        os.system(command)
        
        
        
if __name__=="__main__":
    args = parse_arguments()
    run(args)
