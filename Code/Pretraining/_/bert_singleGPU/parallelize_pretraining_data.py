import os
import time
import argparse
from multiprocessing import Process



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


        
def run_proc(file_id):
    """
    Run one process (create pretraining data from one txt file).
    """
    directory = '/raid/antoloui/Master-thesis/Data/'
    command = '''\
                python create_pretraining_data.py \
                --input_file={in_filepath} \
                --output_file={out_filepath} \
                --vocab_file=./models/base_cased/vocab.txt \
                --do_lower_case=False \
                --max_seq_length={max_seq_length} \
                --max_predictions_per_seq={max_predictions_per_seq} \
                --masked_lm_prob=0.15 \
                --random_seed=12345 \
                --dupe_factor=5
            '''.format(in_filepath=directory+'Preprocessed/text_'+str(file_id)+'.txt', 
                       out_filepath=directory+'bert/L512/tf_examples.tfrecord'+str(file_id),
                       max_seq_length=512,
                       max_predictions_per_seq=75)  #You should set max_predictions_per_seq to around max_seq_length*masked_lm_prob.
    os.system(command)
    


def main(args):
    """
    Run processes in parallel if --all=True, otherwise run one process.
    """
    if args.all:
        # Instantiating process with arguments
        process_list = [Process(target=run_proc, args=(str(i),)) for i in range(1,14)]
        for i, p in enumerate(process_list):
            print('Process {} is starting...'.format(i))
            p.start()
            time.sleep(1)

        # Complete the processes
        for p in process_list:
            p.join()
            
    else:
        i = args.file_id
        run_proc(i)
        
        
        
if __name__=="__main__":
    args = parse_arguments()
    main(args)
