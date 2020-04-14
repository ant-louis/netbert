import os
import sys
import glob
import pickle
import argparse
import time
import datetime

import pandas as pd
from tqdm import tqdm

import faiss
import numpy as np
from sklearn import preprocessing



def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Path to the input directory where all the .h5 files are located.",
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default="",
                        help="Path to output directory. If not mentioned, same as input directory.",
    )
    parser.add_argument("--n_gpu",
                        type=int,
                        default=0,
                        help="Number of GPUs to use for creating FAISS index.",
    )
    parser.add_argument("--method",
                        "-m",
                        type=str,
                        default='l2',
                        choices=['l2','ip', 'cos'],
                        help="Distance method used for comparing embeddings.",
    )
    arguments, _ = parser.parse_known_args()
    return arguments


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def load_embeddings(input_dir):
    """
    """
    # Create dataframe.
    cols = ['feat'+str(i+1) for i in range(768)]
    cols.append('Chunk')
    df = pd.DataFrame(columns=cols)

    #Concat embeddings from all files.
    filepaths = glob.glob(input_dir + '*.h5')
    for file in tqdm(filepaths, desc='Files'):
        df_file = pd.read_hdf(file)
        df_file['Chunk'] = df_file['Chunk'].astype(str)
        df = pd.concat([df, df_file], ignore_index=True, sort=False)

    #Check for duplicated chunks in the concatenated dataframe.
    df.drop_duplicates(subset=['Chunk'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Get chunks and their embeddings.
    chunks = df.iloc[:,-1].values
    embeddings = df.iloc[:,:-1].values
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32) # Necessary for FAISS indexing afterwards.
    
    return chunks, embeddings


def create_faiss_index(vecs, method='l2', n_gpu=0):
    """
    Create FAISS index on GPU(s).
    To create a GPU index with FAISS, one first needs to create it on CPU then copy it on GPU. 
    Note that a "flat" index means that it is brute-force, with no approximation techniques.
    """
    # Build flat CPU index given the chosen method.
    if method=='l2':
        index = faiss.IndexFlatL2(vecs.shape[1])  # Exact Search for L2
    elif method=='ip':
        index = faiss.IndexFlatIP(vecs.shape[1])  # Exact Search for Inner Product
    elif method=='cos':
        # Cosime similarity comes down to normalizing the embeddings beforehand and then applying inner product.
        vecs = preprocessing.normalize(vecs, norm='l2')
        index = faiss.IndexFlatIP(vecs.shape[1])
    else:
        print("Error: Please choose between L2 Distance ('l2'), Inner Product Distance ('ip') or Cosine Distance ('cos') as brute-force method for exact search. Exiting...")
        sys.exit(0)
    
    # Convert to flat GPU index.
    if n_gpu > 0:
        co = faiss.GpuMultipleClonerOptions()  # If using multiple GPUs, enable sharding so that the dataset is divided across the GPUs rather than replicated.
        co.shard = True
        index = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=n_gpu)  # Convert CPU index to GPU index.
    
    # Add vectors to GPU index.
    index.add(vecs)
    
    # Convert back to cpu index (needed for saving it to disk).
    index = faiss.index_gpu_to_cpu(index)

    return index


def main(args):
    """
    """
    t0 = time.time()
    print("\nLoad all embeddings of Cisco corpus from {}...".format(args.input_dir))
    chunks, embeddings = load_embeddings(args.input_dir)
    
    print("Create FAISS (GPU) index...")
    index = create_faiss_index(vecs=embeddings, n_gpu=args.n_gpu)
    
    # Make sure output_dir exists, otherwise save in input_dir.
    if not args.output_dir:
        args.output_dir = args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nSave index to {}...".format(args.output_dir))
    faiss.write_index(index, os.path.join(args.output_dir, args.method + ".index"))
    
    print("\nSave chunks to {}...".format(args.output_dir))
    with open(os.path.join(args.output_dir, "chunks.txt"), "wb") as f:
        pickle.dump(chunks, f)

    print("\nDone.  -  Took: {}\n".format(format_time(time.time() - t0)))
    
    
    
if __name__=="__main__":
    args = parse_arguments()
    main(args)
