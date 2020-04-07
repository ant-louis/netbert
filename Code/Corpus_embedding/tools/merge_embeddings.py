import glob
import argparse
from tqdm import tqdm
import pandas as pd


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        "-i",
                        type=str,
                        required=True,
                        help="Path of the input directory."
    )
    parser.add_argument("--output_dir",
                        "-o",
                        type=str, 
                        required=True,
                        help="Path of the output directory."
    )
    arguments, _ = parser.parse_known_args()
    return arguments


def main(args):
    """
    """
    # Create dataframe.
    cols = ['feat'+str(i+1) for i in range(768)]
    cols.append('Chunk')
    df = pd.DataFrame(columns=cols)

    print("Concat embeddings from files in {}...".format(args.input_dir))
    filepaths = glob.glob(args.input_dir + '*.h5')
    for file in tqdm(filepaths, desc='Files'):
        df_file = pd.read_hdf(file)
        df_file['Chunk'] = df_file['Chunk'].astype(str)
        df = pd.concat([df, df_file], ignore_index=True, sort=False)
    print("  - Done.")
    
    print("Checking for duplicated chunks in the concatenated dataframe...")
    print("  - Found {} duplicated chunks, keeping only first instance...".format(df[df.duplicated(['Chunk'])].shape[0]))
    df.drop_duplicates(subset=['Chunk'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("  - Done.")
    
    # NB: saving does not work for now (dataframe too large)
    print("Saving concatenated embeddings into {}...".format(args.output_dir))
    output_path = args.output_dir + '/all.h5'
    df.to_hdf(output_path, key='df', mode='w')
    print("  - Done.")


if __name__=="__main__":
    args = parse_arguments()
    main(args)
