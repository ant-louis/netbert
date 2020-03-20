import os
import time
import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import BertModel, BertTokenizer
from keras.preprocessing.sequence import pad_sequences



def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", 
                        type=str, 
                        default='/raid/antoloui/Master-thesis/Data/Cleaned/dev.raw',
                        help="Path of the file containing the sentences to encode."
                       )
    parser.add_argument("--output_dir", 
                        type=str, 
                        default='/raid/antoloui/Master-thesis/Data/Embeddings/',
                        help="Path of the directory to save output."
                       )
    parser.add_argument("--model_name_or_path",
                        default='/raid/antoloui/Master-thesis/Code/_models/netbert/checkpoint-1027000/',
                        type=str,
                        #required=True,
                        help="Path to pre-trained model or shortcut name.",
    )
    parser.add_argument("--cache_dir",
                        default='/raid/antoloui/Master-thesis/Code/_cache',
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3.",
    )
    parser.add_argument("--batch_size",
                        default=128,
                        type=int, 
                        help="Batch size per GPU/CPU."
    )
    arguments, _ = parser.parse_known_args()
    return arguments


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_sentences(filepath):
    """
    Given a file of raw sentences, return the list of these sentences.
    """
    with open(filepath) as myfile:
        sentences = [line for line in myfile]          
    return sentences


def sort_sentences(sentences):
    """
    Given a list of sentences, sort them according to their length.
    This is done in order for the batches to have approximately the
    same length, avoiding mixing very short and very long sentences,
    within the same batch, which results in padding the very short 
    sentences and thus messing up the embedding when averaging the word
    embeddings with lots of padding tokens.
    """
    return sorted(sentences, key=len)


def encode_sentences(args, sentences):
    """
    Encode corpus of sentences with CPU or GPU(s).
    
    Note that multi-GPU encoding is quite imbalanced due to the use
    of 'torch.nn.DataParallel' which loads the model in the main GPU
    but also gathers the output of all other GPUs back to the main one.
    As a result, the main GPU is three times more loaded than the others.
    Although the other GPUs are not fully loaded, one can not increase the
    batch size as it would result in an 'out of memory' error in the main GPU.
    Here, the max batch size is 128.
    
    Note that the GPU utilisation with 'torch.nn.DataParallel' is very volatile.
    GPUs are never running at 100%, which slows the process.
    """
    print("   Loading pretrained model/tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertModel.from_pretrained(args.model_name_or_path, output_hidden_states=True, cache_dir=args.cache_dir) # Will output all hidden_states.
    
    print("   Setting up CUDA & GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        gpu_ids = list(range(0, n_gpu))
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[-1])
    model.to(device)
    
    print("   Encoding sentences...")
    all_embeddings = []
    iterator = range(0, len(sentences), args.batch_size)
    for batch_idx in tqdm(iterator, desc="Batches"):
        
        # Get the batch.
        batch_start = batch_idx
        batch_end = min(batch_start + args.batch_size, len(sentences))
        batch_sentences = sentences[batch_start:batch_end]
        
        # Tokenize each sentence of the batch.
        tokenized = [tokenizer.encode(sent, add_special_tokens=True) for sent in batch_sentences]
        
        # Pad/Truncate sentences to max_len or 512.
        lengths = [len(i) for i in tokenized]
        max_len = max(lengths) if max(lengths) <= 512 else 512
        padded = pad_sequences(tokenized, maxlen=max_len, dtype="long", 
                          value=0, truncating="post", padding="post")
        
        # Create attention masks.
        attention_mask = np.where(padded != 0, 1, 0)  #returns ndarray which is 1 if padded != 0 is True and 0 if False.
        
        # Convert inputs to torch tensors.
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)
        
        # Push inputs to GPUs.
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Encode batch.
        model.eval()
        with torch.no_grad():
            # output is a 2-tuple where:
            #  - output[0] is the last_hidden_state, i.e a tensor of shape (batch_size, sequence_length, hidden_size).
            #  - output[1] is the pooler_output, i.e. a tensor of shape (batch_size, hidden_size) being the last layer hidden-state of the first token of the sequence (classification token).
            #  - output[2] are all hidden_states, i.e. a 13-tuple of torch tensors of shape (batch_size, sequence_length, hidden_size): 12 encoders-outputs + initial embedding outputs.
            output = model(input_ids, attention_mask=attention_mask)
        
        # For each sentence, take the embeddings of its word from the last layer and represent that sentence by their average.
        last_hidden_states = output[0]
        sentence_embeddings = [torch.mean(embeddings, dim=0).to('cpu').numpy() for embeddings in last_hidden_states]
        all_embeddings.extend(sentence_embeddings)
    
    # Create dataframe for storing embeddings.
    all_embeddings = np.array(all_embeddings)
    cols = ['feat'+str(i+1) for i in range(all_embeddings.shape[1])]
    df = pd.DataFrame(data=all_embeddings[:,:], columns=cols)
    df['Sentence'] = sentences
    return df


def main(args):
    """
    Main function.
    """
    print("\n===================================================")
    print("Loading sentences from {}...".format(args.filepath))
    print("===================================================\n")
    t0 = time.time()
    sentences = load_sentences(args.filepath)
    print("   {} sentences loaded.  -  Took: {}".format(len(sentences), format_time(time.time() - t0)))
    
    print("\n===================================================")
    print("Sorting sentences by length...")
    print("===================================================\n")
    t0 = time.time()
    sorted_sentences = sort_sentences(sentences)
    #sorted_sentences = sorted_sentences[::-1]
    print("   {} sentences sorted.  -  Took: {}".format(len(sentences), format_time(time.time() - t0)))
    
    print("\n===================================================")
    print("Encoding {} sentences...".format(len(sentences)))
    print("===================================================\n")
    t0 = time.time()
    df = encode_sentences(args, sorted_sentences)
    elapsed = time.time() - t0
    print("   {} sentences embedded. -  Took: {:}  ({:.2f} s/sentences)".format(len(sentences), format_time(elapsed), elapsed/len(sorted_sentences)))
    
    print("\n===================================================")
    print("Saving dataframe to {}...".format(args.output_dir))
    print("===================================================\n")
    t0 = time.time()
    filename = os.path.splitext(os.path.basename(args.filepath))[0]
    output_path = args.output_dir + filename + '.csv'
    df.to_csv(output_path, sep=',', encoding='utf-8', float_format='%.10f', decimal='.', index=False)
    print("   Dataframe saved. -  Took: {}\n".format(format_time(time.time() - t0)))
    

if __name__=="__main__":
    args = parse_arguments()
    main(args)
