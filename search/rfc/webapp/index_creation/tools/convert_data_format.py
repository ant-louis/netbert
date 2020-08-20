import os
import glob
import argparse

import pandas as pd
from tqdm import tqdm



def parse_arguments():
    """
    url, errors_filepath, outdir
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default='/raid/antoloui/Master-thesis/search/rfc/_data/processed/',
                        help="Path to the data directory.",
    )
    arguments, _ = parser.parse_known_args()
    return arguments


def main(args):
    """
    """
    # Get paths of all files in repo.
    filepaths = [f for f in glob.glob(args.data_dir+"*.txt")]
    
    # Create global dataframe to store all.
    global_df = pd.DataFrame(columns=['Title', 'Text'])
    
    for i, filename in enumerate(tqdm(filepaths)):
        # Read file.
        with open(filename, 'r+') as f:
            lines = f.readlines()
            
        # For each line, extract title and text.
        titles = []
        texts = []
        for line in lines:
            titles.append(line.split('*')[1].strip())
            texts.append(line.split('*')[2].strip())

        # Create dataframe.
        d = {'Title':titles,'Text':texts}
        df = pd.DataFrame(d)
        
        # Concat to global one.
        global_df = pd.concat([global_df, df], axis=0)
        
    # Save global dataframe.
    global_df.to_csv(args.data_dir + '../data.csv', sep=',', encoding='utf-8', float_format='%.10f', decimal='.')
    return


if __name__=="__main__":
    args = parse_arguments()
    main(args)
