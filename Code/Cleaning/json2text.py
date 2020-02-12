import glob
import sys
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str,
        help="path where the json file is located")
    parser.add_argument("--output_file", type=str,
        help="filename where the raw text should go")
    arguments, _ = parser.parse_known_args()
    return arguments


def main(args):
    """
    """
    data_dir = '/raid/antoloui/Master-thesis/Data/Cleaned/'
    fname = data_dir + args.json_file
    out_file = data_dir + args.output_file

    with open(out_file, 'w') as outfile:
        with open(fname, 'r') as infile:
            for row in tqdm(infile):
                doc = json.loads(row)
                text = doc.get('text')
                outfile.write(text + '\n\n')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
