import argparse
from numpy import save

from bert_serving.client import BertClient


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default='./sample.txt',
                        help="Path of the file containing the sentences to encode.")
    parser.add_argument("--output", type=str, default='.output/encodings.npy',
                        help="Path where to save sentence encodings.")
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
    # Extract sentences from file
    sentences = extract_sentences(args.filepath)
    
    # Encode them via bert-as-a-service
    with BertClient() as bc:
        encodings = bc.encode(sentences)
    
    # Save encodings to npy file
    save(args.output, encodings)
    print("Encodings saved !")


if __name__=="__main__":
    args = parse_arguments()
    main(args)
    