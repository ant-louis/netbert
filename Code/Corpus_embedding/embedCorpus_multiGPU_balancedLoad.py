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

import parallel
from torch.nn.parallel.scatter_gather import gather



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
                        default=100,
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



def encode_sentences(args, sentences):
    """
    """
    print("   Loading pretrained model/tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertModel.from_pretrained(args.model_name_or_path, output_hidden_states=True, cache_dir=args.cache_dir) # Will output all hidden_states.
    
    print("   Setting up CUDA & GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        parallel_model = parallel.DataParallelModel(model)
    parallel_model.to(device)
    
    
    print("   Encoding sentences...")
    
    # Create dataframe for storing embeddings.
    cols = ['feat'+str(i+1) for i in range(768)]
    df = pd.DataFrame(columns=cols)
    df['Sentence'] = None
    
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
        parallel_model.eval()
        with torch.no_grad():
            # output is a 2-tuple where:
            #  - output[0] is the last_hidden_state, i.e a tensor of shape (batch_size, sequence_length, hidden_size).
            #  - output[1] is the pooler_output, i.e. a tensor of shape (batch_size, hidden_size) being the last layer hidden-state of the first token of the sequence (classification token).
            #  - output[2] are all hidden_states, i.e. a 13-tuple of torch tensors of shape (batch_size, sequence_length, hidden_size): 12 encoders-outputs + initial embedding outputs.
            output = parallel_model(input_ids, attention_mask=attention_mask)
        
        print(len(output))
        # Gather all the tensors on the cpu.
        gathered_output = gather(output)
        
        print(type(gathered_output))
        
        
        # Concatenate the tensors for all layers. We use `stack` here to create a new dimension in the tensor.
        hidden_states = torch.stack(output[2], dim=0)

        # Switch around the “layers” and “tokens” dimensions with permute.
        hidden_states = hidden_states.permute(1,2,0,3)
        
        # For each sentence, take the embeddings of its word from the last layer and represent that sentence by their average.
        last_hidden_states = output[0].detach().cpu()
        sentence_embeddings = [torch.mean(embeddings, dim=0).numpy() for embeddings in last_hidden_states]
        sentence_embeddings = np.array(sentence_embeddings)
    
        # Append batch dataframe to full dataframe.
        batch_df = pd.DataFrame(data=sentence_embeddings[:,:], columns=cols)
        batch_df['Sentence'] = batch_sentences
        df = pd.concat([df, batch_df], axis=0)
        
    return df






def main(args):
    """
    Main function.
    """
    print("\n===================================================")
    print("Loading sentences from {}...".format(args.filepath))
    print("===================================================\n")
    sentences = load_sentences(args.filepath)
    
    print("\n===================================================")
    print("Encoding {} sentences...".format(len(sentences)))
    print("===================================================\n")
    t0 = time.time()
    df = encode_sentences(args, sentences)
    elapsed = time.time() - t0
    print("   Encoding took: {:}  ({:.2f} s/sentences)\n".format(format_time(elapsed), elapsed/len(sentences)))
    
    print("\n===================================================")
    print("Saving dataframe to {}...".format(args.output_dir))
    print("===================================================\n")
    filename = os.path.splitext(os.path.basename(args.filepath))[0]
    output_path = args.output_dir + filename + '.csv'
    df.to_csv(output_path, sep=',', encoding='utf-8', float_format='%.10f', decimal='.', index=False)
    



if __name__=="__main__":
    args = parse_arguments()
    main(args)
