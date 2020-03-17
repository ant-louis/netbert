import os
import glob
import time
import datetime
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import BertModel, BertTokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import parallel


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", 
                        type=str, 
                        default='/raid/antoloui/Master-thesis/Data/Cleaned/Splitted/',
                        help="Path of the input directory."
                       )
    parser.add_argument("--output_dir", 
                        type=str, 
                        default='/raid/antoloui/Master-thesis/Data/Embeddings/',
                        help="Path of the output directory."
                       )
    parser.add_argument("--model_name_or_path",
                        default='/raid/antoloui/Master-thesis/Code/_models/netbert/checkpoint-1825000/',
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
                        default=1024,
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


def gather_sentence_outputs(outputs):
    """
    'outputs' is a list of 'output' of each GPU. As a reminder, each 'output' is a 3-tuple where:
        - output[0] is the last_hidden_state, i.e a tensor of shape (batch_size, sequence_length, hidden_size).
        - output[1] is the pooler_output, i.e. a tensor of shape (batch_size, hidden_size) being the last layer hidden-state of the first token of the sequence (classification token).
        - output[2] are all hidden_states, i.e. a 13-tuple of torch tensors of shape (batch_size, sequence_length, hidden_size): 12 encoders-outputs + initial embedding outputs.
    """
    # Extract the last_hidden_state in each GPU output ('gathered' is a list of nb_gpu x torch tensors)
    gathered = [output[0] for output in outputs]
    
    # Concatenate the samples for that batch.
    gathered = torch.cat(gathered, dim=0)
    return gathered


def gather_word_outputs(outputs):
    """
    'outputs' is a list of 'output' of each GPU. As a reminder, each 'output' is a 3-tuple where:
        - output[0] is the last_hidden_state, i.e a tensor of shape (batch_size, sequence_length, hidden_size).
        - output[1] is the pooler_output, i.e. a tensor of shape (batch_size, hidden_size) being the last layer hidden-state of the first token of the sequence (classification token).
        - output[2] are all hidden_states, i.e. a 13-tuple of torch tensors of shape (batch_size, sequence_length, hidden_size): 12 encoders-outputs + initial embedding outputs.
    """
    # Extract all hidden_states in each GPU output ('gathered' is a list of nb_gpu x 13-tuple)
    gathered = [output[2] for output in outputs]
    
    # Concatenate the tensors for all layers. We use `stack` here to create a new dimension in the tensor.
    gathered = [torch.stack(output, dim=0) for output in gathered]
    
    # Switch around the “layers” and “tokens” dimensions with permute.
    gathered = [output.permute(1,2,0,3) for output in gathered]
    
    # Concatenate the samples for that batch.
    gathered = torch.cat(gathered, dim=0)
    return gathered


def encode_sentences(sentences, tokenizer, model, device, batch_size):
    """
    Encoding sentences with CPU/GPU(s).
    
    Note that here 'parallel.DataParallelModel' is used, where 'parallel.py' is a script imported
    from the ' PyTorch-Encoding' package: https://github.com/zhanghang1989/PyTorch-Encoding
    The DataParallelModel deals better with balanced load on multi-GPU than torch.nn.DataParallel,
    allowing to significantly increase the batch size per GPU.

    However, once again, the utilisation of the GPUs is very volatile (never at 100% all the time).
    """
    all_embeddings = []
    iterator = range(0, len(sentences), batch_size)
    for batch_idx in tqdm(iterator, desc="   Batches"):
        
        # Get the batch.
        batch_start = batch_idx
        batch_end = min(batch_start + batch_size, len(sentences))
        batch_sentences = sentences[batch_start:batch_end]
        
        # Tokenize each sentence of the batch.
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)  # No warning on sample size (I deal with that below).
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
            # outputs is a list of 3-tuples where each 3-tuple is such that:
            #  - output[0] is the last_hidden_state, i.e a tensor of shape (batch_size, sequence_length, hidden_size).
            #  - output[1] is the pooler_output, i.e. a tensor of shape (batch_size, hidden_size) being the last layer hidden-state of the first token of the sequence (classification token).
            #  - output[2] are all hidden_states, i.e. a 13-tuple of torch tensors of shape (batch_size, sequence_length, hidden_size): 12 encoders-outputs + initial embedding outputs.
            outputs = model(input_ids, attention_mask=attention_mask)
            
        # Gather outputs from the different GPUs.
        last_hidden_states = gather_sentence_outputs(outputs)

        # For each sentence, take the embeddings of its word from the last layer and represent that sentence by their average.
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
    """
    print("\n===================================================")
    print("Loading pretrained model/tokenizer...")
    print("===================================================\n")
    t0 = time.time()
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertModel.from_pretrained(args.model_name_or_path, output_hidden_states=True, cache_dir=args.cache_dir) # Will output all hidden_states.
    print("   Loaded checkpoint '{}'.  -  Took: {}".format(args.model_name_or_path, format_time(time.time() - t0)))
    
    print("\n===================================================")
    print("Setting up CPU / CUDA & GPUs...")
    print("===================================================\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = parallel.DataParallelModel(model)
    model.to(device)
    print("   Using {}.\n".format(device))
    
    print("\n===================================================")
    print("Encoding sentences...")
    print("===================================================\n")
    files = glob.glob(args.input_dir + '*.raw')
    for file in tqdm(files, desc="Files"):
        print("   Loading sentences from {}...".format(file))
        t0 = time.time()
        sentences = load_sentences(file)
        sorted_sentences = sort_sentences(sentences)
        sorted_sentences = sorted_sentences[::-1]
        print("   {} sentences loaded.  -  Took: {}".format(len(sentences), format_time(time.time() - t0)))

        print("   Encoding sentences...")
        t0 = time.time()
        df = encode_sentences(sorted_sentences, tokenizer, model, device, args.batch_size)
        elapsed = time.time() - t0
        print("   {} sentences embedded. -  Took: {:}  ({:.2f} s/sentences)".format(len(sorted_sentences), format_time(elapsed), elapsed/len(sorted_sentences)))

        print("   Saving dataframe to {}...".format(args.output_dir))
        t0 = time.time()
        filename = os.path.splitext(os.path.basename(file))[0]
        output_path = args.output_dir + filename + '.csv'
        df.to_csv(output_path, sep=',', encoding='utf-8', float_format='%.10f', decimal='.', index=False)
        print("   Dataframe saved. -  Took: {}\n".format(format_time(time.time() - t0)))
    

if __name__=="__main__":
    args = parse_arguments()
    main(args)
