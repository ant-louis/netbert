import argparse
from numpy import save

from bert_serving.client import BertClient


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default='./data/little_sample.txt',
                        help="Path of the file containing the sentences to encode.")
    parser.add_argument("--output", type=str, default='./output/encodings.npy',
                        help="Path where to save sentence encodings.")
    parser.add_argument("--sentences", type=bool, default=True,
                        help="Wether to encode sentences (if set to False, will encode words).")
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
            words.extend(line.split())
    return list(set(words))
            
    

def main(args):
    """
    """
    # Extract strings from file
    if args.sentences:
        strings = extract_sentences(args.filepath)
    else:
        strings = extract_words(args.filepath)
        
    
    # Encode strings via bert-as-service
    with BertClient() as bc:
        encodings = bc.encode(strings)
        
       
    print(encodings)
    
    # Save encodings to npy file
    save(args.output, encodings)
    print("Encodings saved !")


if __name__=="__main__":
    args = parse_arguments()
    main(args)
    