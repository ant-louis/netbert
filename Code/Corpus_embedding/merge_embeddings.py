import glob
import argparse
from tqdm import tqdm
import pandas as pd


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", "-d", 
                        type=str, default='/raid/antoloui/Master-thesis/Data/Embeddings/', 
                        help="Path of the input directory.")
    arguments, _ = parser.parse_known_args()
    return arguments


def main(args):
    """
    """
    # Get the paths of the files.
    filepaths = glob.glob(args.dirpath + '*.h5')

    # Create dataframe.
    cols = ['feat'+str(i+1) for i in range(768)]
    cols.append('Chunk')
    df = pd.DataFrame(columns=cols)

    # Concat dataframes from all files.
    print("Merging files' embeddings...")
    for file in tqdm(filepaths, desc='Files'):
        df_file = pd.read_hdf(file)
        df = pd.concat([df, df_file], ignore_index=True, sort=False)
        
    # Save final dataframe.
    print("Saving all embeddings into one dataframe...")
    output_path = args.dirpath + '/all.h5'
    df.to_hdf(output_path, key='df', mode='w')
    print("Done.")


if __name__=="__main__":
    args = parse_arguments()
    main(args)
