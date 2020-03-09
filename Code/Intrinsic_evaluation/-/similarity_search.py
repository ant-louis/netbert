import numpy as np
import argparse
import math
import json
from operator import itemgetter
from Levenshtein import distance as levenshtein_distance
from bert_serving.client import BertClient


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default='./data/sample.txt',
                        help="Path of the file containing the sentences to encode.")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of most similar sentences to retrieve.")
    parser.add_argument("--output", type=str, default='./output/similarities_netbert.json',
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


def cosine_similarity_matrix(query_vec, sentences_vecs):
    """
    Given a query vector and an array of sentence vectors, return an array of cosine
    similarity scores between the query and each sentence.
    """
    return np.sum(query_vec * sentences_vecs, axis=1) / (np.linalg.norm(query_vec) * np.linalg.norm(sentences_vecs, axis=1))


def cosine_similarity(vec1, vec2):
    """
    Given two vectors, return the cosine similarity score.
    """
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def manhattan_distance(vec1, vec2):
    """
    Given two vectors, return the manhattan distance.
    """
    return np.sum(np.abs(vec1-vec2))


def euclidean_distance(vec1, vec2):
    """
    Given two vectors, return the euclidean distance.
    """
    return np.linalg.norm(vec1-vec2)
    


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
            # Compute edit distances between query and each sentence
            edit_distances = [levenshtein_distance(sentences[query_idx], sent) for sent in sentences]
            
            # Take the top 10% most dissimilar sentences according to the edit distance
            topk = int(0.1 * len(sentences))
            topk_edit_idx = np.argsort(edit_distances)[::-1][:topk]
            
            # Among these most dissimilar sentences, take the top 5 with highest cosine similarity
            scores = [(cosine_similarity(query_vec, sentences_vecs[i]), i) for i in topk_edit_idx]
            if args.topk > topk:
                print("ERROR: must choose a topk value under {}".format(topk))
                break
            topk_cosine = [(x[0],x[1]) for x in sorted(scores, key=itemgetter(0), reverse=True)][:args.topk]
            
            # Create the json output
            query_dict = dict()
            for i, (cos, idx) in enumerate(topk_cosine):
                tmp_dict = dict()
                tmp_dict['Sentence'] = sentences[idx]
                tmp_dict['Edit distance'] = str(edit_distances[idx])
                tmp_dict['Cosine similarity'] = str(cos)
                query_dict[str(i+1)] = tmp_dict
            sim_dict[sentences[query_idx]] = query_dict
            
        with open(args.output, 'w') as outfile:
            json.dump(sim_dict, outfile)
    
    
if __name__=="__main__":
    args = parse_arguments()
    main(args)
