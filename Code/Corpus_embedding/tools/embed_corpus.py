import os
import glob
import time
import datetime
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from keras.preprocessing.sequence import pad_sequences

import parallel
from transformers import BertModel


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
    parser.add_argument("--model_name_or_path",
                        "-m",
                        type=str,
                        required=True,
                        help="Path to pre-trained model or shortcut name.",
    )
    parser.add_argument("--cache_dir",
                        "-c",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3.",
    )
    parser.add_argument("--batch_size",
                        "-b",
                        default=64,
                        type=int, 
                        help="Batch size per GPU/CPU."
    )
    parser.add_argument("--dataparallelmodel",
                        "-p",
                        action='store_true',
                        help="Use full capacity of GPUs parallel embedding."
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
    # Load sentences from file.
    with open(filepath) as myfile:
        sentences = [line for line in myfile if line != '\n']
    
    # Only keep unique sentences.
    sentences = list(dict.fromkeys(sentences).keys())
    
    # Only keep sentences with less than 1300 char (bigger sentences are messed up).
    sentences = [sent for sent in sentences if len(sent) <= 1300]
    return sentences


def tokenize(tokenizer, sentences):
    """
    Given a list of sentences, convert words to vocab ids (0 -> 30522) in each sentence.
    This function is supposed to use BertTokenizer from the transformers library.
    """
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)  # No warning on sample size (I deal with that below).
    indexed_tokens = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]    
    return indexed_tokens


def tokenize_fast(tokenizer, sentences):
    """
    Given a list of sentences, convert words to vocab ids (0 -> 30522) in each sentence.
    This function is supposed to use BertWordPieceTokenizer from the tokenizers library.
    """
    outputs = [tokenizer.encode(sent) for sent in sentences]
    indexed_tokens = [out.ids for out in outputs]
    return indexed_tokens


def create_chunks(sentences, indexed_tokens):
    """
    Given the tokenized sentences, return chunks of multiple sentences under 512 tokens.
    """
    token_chunks = [] 
    sentence_chunks = []

    # While list of tokenized sentences is not empty...
    while indexed_tokens:

        # Create the new chunk.
        chunk_tok = []
        chunk_sent = []

        # If:
        #  - The current chunk is empty;
        #  - There is still sentences to pop in the list;
        #  - The next sentence to pop is bigger than 512 tokens;
        if (not chunk_tok) and (indexed_tokens) and (len(indexed_tokens[0]) > 512):
            # Append the next sentence to the chunk (this chunk will be truncated later).
            chunk_tok.extend(indexed_tokens.pop(0))
            chunk_sent.append(sentences.pop(0))

        # While:
        #   - There is still sentences to pop in the list;
        #   - The length of the current chunk combined with the length of the next sentence to pop is lower than 512 tokens;
        while (indexed_tokens) and ((len(chunk_tok) + len(indexed_tokens[0])) <= 512):
            # Pop the next tokenized sentence and append it to the chunk.
            chunk_tok.extend(indexed_tokens.pop(0))
            chunk_sent.append(sentences.pop(0))

        # Concat all sentences in a chunk and remove \n.
        chunk_sent = ' '.join(chunk_sent).replace('\n', ' ')

        # Add that chunk to my list of chunks.
        token_chunks.append(chunk_tok)
        sentence_chunks.append(chunk_sent)
        
    return token_chunks, sentence_chunks


def pad_and_truncate_chunks(token_chunks):
    """
    Given a list of tokenized chunks, pad/truncate them so that they all have the same length.
    """
    # Define length of longest tokenized chunk in the batch.
    lengths = [len(i) for i in token_chunks]
    max_len = max(lengths) if max(lengths) <= 512 else 512

    # Pad/truncate chunks.
    padded_chunks = pad_sequences(token_chunks, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    return padded_chunks
    

def create_attention_masks(padded_chunks):
    """
    Given a list of tokenized padded chunks, create the attention masks for each of them.
    """
    attention_masks = np.where(padded_chunks != 0, 1, 0)  #returns ndarray which is 1 if padded_tokens != 0 is True and 0 if False.
    return attention_masks


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


def encode_chunks(args, model, sentence_chunks, padded_chunks, attention_masks):
    """
    Encoding sentences with CPU/GPU(s).
    
    Note that here 'parallel.DataParallelModel' is used, where 'parallel.py' is a script imported
    from the ' PyTorch-Encoding' package: https://github.com/zhanghang1989/PyTorch-Encoding
    The DataParallelModel deals better with balanced load on multi-GPU than torch.nn.DataParallel,
    allowing to significantly increase the batch size per GPU.

    However, once again, the utilisation of the GPUs is very volatile (never at 100% all the time).
    """
    all_embeddings = []
    iterator = range(0, len(sentence_chunks), args.batch_size)
    for batch_idx in tqdm(iterator, desc="   Batches"):
        
        # Get the batch indices.
        batch_start = batch_idx
        batch_end = min(batch_start + args.batch_size, len(sentence_chunks))
        
        # Get the current batch.
        batch_input_ids = padded_chunks[batch_start:batch_end]
        batch_attention_masks = attention_masks[batch_start:batch_end]
        
        # Convert model inputs to torch tensors and push them to GPUs.
        batch_input_ids = torch.tensor(batch_input_ids)
        batch_input_ids = batch_input_ids.to(args.device)
        batch_attention_masks = torch.tensor(batch_attention_masks)
        batch_attention_masks = batch_attention_masks.to(args.device)
        
        # Encode batch.
        model.eval()
        with torch.no_grad():
            # outputs is a list of 3-tuples where each 3-tuple is such that:
            #  - output[0] is the last_hidden_state, i.e a tensor of shape (batch_size, sequence_length, hidden_size).
            #  - output[1] is the pooler_output, i.e. a tensor of shape (batch_size, hidden_size) being the last layer hidden-state of the first token of the sequence (classification token).
            #  - output[2] are all hidden_states, i.e. a 13-tuple of torch tensors of shape (batch_size, sequence_length, hidden_size): 12 encoders-outputs + initial embedding outputs.
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
        
        if args.dataparallelmodel:
            # Gather outputs from the different GPUs.
            last_hidden_states = gather_sentence_outputs(outputs)
        else:
            last_hidden_states = outputs[0]

        # For each sentence, take the embeddings of its word from the last layer and represent that sentence by their average.
        chunk_embeddings = [torch.mean(embeddings[:torch.squeeze((masks == 1).nonzero(), dim=1).shape[0]], dim=0).to('cpu').numpy() for embeddings, masks in zip(last_hidden_states, batch_attention_masks)]
        all_embeddings.extend(chunk_embeddings)
        
    # Create dataframe for storing embeddings.
    all_embeddings = np.array(all_embeddings)
    cols = ['feat'+str(i+1) for i in range(all_embeddings.shape[1])]
    df = pd.DataFrame(data=all_embeddings[:,:], columns=cols)
    df['Chunk'] = sentence_chunks
    
    return df



def main(args):
    print("\n===================================================")
    print("Loading pretrained model/tokenizer...")
    print("===================================================\n")
    t0 = time.time()
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    fast_tokenizer = BertWordPieceTokenizer(args.model_name_or_path + 'vocab.txt',
                                            add_special_tokens=True,
                                            lowercase=False,
                                            clean_text=True, 
                                            handle_chinese_chars=True,
                                            strip_accents=True)
    model = BertModel.from_pretrained(args.model_name_or_path, 
                                      output_hidden_states=True, # Will output all hidden_states.
                                      cache_dir=args.cache_dir)
    print("   Loaded checkpoint '{}'.  -  Took: {}".format(args.model_name_or_path, format_time(time.time() - t0)))
    
    print("\n===================================================")
    print("Setting up CPU / CUDA & GPUs...")
    print("===================================================\n")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    if args.n_gpu > 1:
        if args.dataparallelmodel:
            model = parallel.DataParallelModel(model)
        else:
            gpu_ids = list(range(0, args.n_gpu))
            model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[-1])
    model.to(args.device)
    print("   Using {}.\n".format(args.device))
    
    print("\n===================================================")
    print("Encoding sentences...")
    print("===================================================\n")
    files = glob.glob(args.input_dir + '*.raw')
    for file in tqdm(files, desc="Files"):
        print("   Loading sentences from {}...".format(file))
        t0 = time.time()
        sentences = load_sentences(file)
        print("     - {} sentences loaded.  -  Took: {}".format(len(sentences), format_time(time.time() - t0)))
        
        print("   Tokenizing sentences...")
        t0 = time.time()
        #indexed_tokens = tokenize(tokenizer, sentences)
        indexed_tokens = tokenize_fast(fast_tokenizer, sentences)
        print("     - {} sentences tokenized.  -  Took: {}".format(len(indexed_tokens), format_time(time.time() - t0)))
        
        print("   Creating chunks of max 512 tokens...")
        t0 = time.time()
        token_chunks, sentence_chunks = create_chunks(sentences, indexed_tokens)
        print("     - {} chunks created.  -  Took: {}".format(len(token_chunks), format_time(time.time() - t0)))
        
        print("   Padding/truncating the chunks...")
        t0 = time.time()
        padded_chunks = pad_and_truncate_chunks(token_chunks)
        print("     - {} chunks padded/truncated.  -  Took: {}".format(len(padded_chunks), format_time(time.time() - t0)))
        
        print("   Creating attention masks...")
        t0 = time.time()
        attention_masks = create_attention_masks(padded_chunks)
        print("     - {} attention masks created.  -  Took: {}".format(len(attention_masks), format_time(time.time() - t0)))

        print("   Encoding chunks...")
        t0 = time.time()
        df = encode_chunks(args, model, sentence_chunks, padded_chunks, attention_masks)
        elapsed = time.time() - t0
        print("     - {} chunks encoded. -  Took: {:}  ({:.2f} s/chunks)".format(len(padded_chunks), format_time(elapsed), elapsed/len(padded_chunks)))

        print("   Saving dataframe to {}...".format(args.output_dir))
        t0 = time.time()
        filename = os.path.splitext(os.path.basename(file))[0]
        output_path = args.output_dir + filename + '.h5'
        df.to_hdf(output_path, key='df', mode='w')
        #df.to_csv(output_path, sep=',', encoding='utf-8', float_format='%.10f', decimal='.', index=False)
        print("     - Dataframe saved. -  Took: {}\n".format(format_time(time.time() - t0)))
    

if __name__=="__main__":
    args = parse_arguments()
    main(args)
