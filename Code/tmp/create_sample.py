import json
import argparse


def create_sample(input_file, output_file, nb_docs):
    """
    Create a sample file.
    """
    # Open and read json file
    with open(input_file) as json_file:
        data = json.load(json_file)

    # Loop over each document
    for i, doc in enumerate(data):
        # Get the text
        text = doc.get('text')

        # Append the text
        with open(output_file,'a') as f:
            f.write(text[0] + "\n")

        # Take only sample of all docs
        if i > nb_docs:
            break


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="1.json",
                        help="Input file.")
    parser.add_argument("--out_file", type=str, default="sample.txt",
                        help="Output file.")
    parser.add_argument("--nb_docs", type=int, default=2,
                        help="Number of documents to consider.")
    arguments, _ = parser.parse_known_args()
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    create_sample(args.in_file, args.out_file, args.nb_docs)
    