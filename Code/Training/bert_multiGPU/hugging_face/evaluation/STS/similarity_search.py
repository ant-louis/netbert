import numpy as np
import argparse
import math
import json
from bert_serving.client import BertClient


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default='./sample.txt',
                        help="Path of the file containing the sentences to encode.")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of most similar sentences to retrieve.")
    parser.add_argument("--output", type=str, default='./output/similarities.json',
                        help="Path where to save output.")
    arguments, _ = parser.parse_known_args()
    return arguments


def extract_sentences(filepath):
    """
    """
    sentences = []
    with open(filepath) as infile:
        for line in infile:
            sentences.append(line)
    return sentences


def main(args):
    """
    """
    with BertClient() as bc:
        # Init dict
        sim_dict = dict()
        
        # Extract sentences from file and encode them
        sentences = extract_sentences(args.filepath)
        sentences_vecs = bc.encode(sentences)
    
        for query_idx, query_vec in enumerate(sentences_vecs):
            # compute normalized dot product as score
            score = np.sum(query_vec * sentences_vecs, axis=1) / np.linalg.norm(sentences_vecs, axis=1)
            topk_idx = np.argsort(score)[::-1][1:args.topk]
            query_dict = dict()
            for i, idx in enumerate(topk_idx):
                tmp_dict = dict()
                tmp_dict['Sentence'] = sentences[idx]
                tmp_dict['Score'] = str(score[idx])
                query_dict[str(i+1)] = tmp_dict
            sim_dict[sentences[query_idx]] = query_dict
            
        with open(args.output, 'w') as outfile:
            json.dump(sim_dict, outfile)
    
    
if __name__=="__main__":
    args = parse_arguments()
    main(args)
