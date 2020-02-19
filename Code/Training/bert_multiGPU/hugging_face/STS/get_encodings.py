import argparse
import numpy as np
import pandas as pd
import string

from bert_serving.client import BertClient


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default='./data/little_sample.txt',
                        help="Path of the file containing the sentences to encode.")
    parser.add_argument("--output", type=str, default='./output/encodings.csv',
                        help="Path where to save sentence encodings.")
    parser.add_argument("--sentences", action="store_true",
                        help="Wether to encode sentences.")
    arguments, _ = parser.parse_known_args()
    return arguments


def extract_sentences(filepath):
    """
    Extract the sentences in the given file.
    """
    sentences = []
    with open(filepath) as infile:
        for line in infile:
            sentences.append(line)
    return sentences


def extract_words(filepath):
    """
    Extract the words in the given file.
    """
    words = []
    with open(filepath) as infile:
        for line in infile:
            # Lowercase sentence and split it into words
            tokens = line.lower().split()
            print(tokens)
            # Remove punctuation from each word
            tokens = [tok.translate(str.maketrans('', '', string.punctuation)) for tok in tokens]
            words.extend(tokens)
    return list(set(words))
            
    

def main(args):
    """
    """
    # Extract strings from file
    if args.sentences:
        strings = extract_sentences(args.filepath)
    else:
        strings = extract_words(args.filepath)
        print(strings)
        
    # Encode strings via bert-as-service
    with BertClient() as bc:
        encodings = bc.encode(strings)
        
    # Create dataframe
    cols = ['feat'+str(i) for i in range(encodings.shape[1])]
    df = pd.DataFrame(data=encodings[:,:], columns=cols)
    df['text'] = strings
    
    # Save encodings
    df.to_csv(args.output, index=False, sep=',', encoding='utf-8', float_format='%.10f', decimal='.')
    print("Encodings saved !")


if __name__=="__main__":
    args = parse_arguments()
    main(args)
    