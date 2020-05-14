import json
import argparse
import sys
import time
import datetime
import random
import os
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        type=str,
                        required=True,
                        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to launch training.",
    )
    parser.add_argument("--training_filepath",
                        default=None,
                        type=str,
                        help="Path of the file containing the sentences to encode.",
    )
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to evaluate the model.",
    )
    parser.add_argument("--eval_filepath",
                        default=None,
                        type=str,
                        help="Path of the file containing the sentences to encode.",
    )
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--cache_dir",
                        default='/raid/antoloui/Master-thesis/Code/_cache/',
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3.",
    )
    parser.add_argument("--num_labels",
                        required=True,
                        type=int,
                        help="Number of classification labels.",
    )
    parser.add_argument('--test_percent',
                        default=0.1,
                        type=float,
                        help='Percentage of available data to use for val/test dataset ([0,1]).',
    )
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed for initialization.",
    )
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32 per GPU/CPU.",
    )
    parser.add_argument("--num_epochs",
                        default=4,
                        type=int,
                        help="Total number of training epochs to perform. Authors recommend 2,3 or 4.",
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam. The authors recommend 5e-5, 3e-5 or 2e-5."
    )
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument("--gpu_id",
                        default=None,
                        type=int,
                        help="Id of the GPU to use if multiple GPUs available.",
    )
    parser.add_argument("--logging_steps",
                        default=10,
                        type=int,
                        help="Log every X updates steps.",
    )
    parser.add_argument("--balanced",
                        action='store_true',
                        help="Should the training dataset be balanced or not.",
    )
    arguments, _ = parser.parse_known_args()
    return arguments


def load_data(args, interest_classes=None):
    """
    Filepath must be a csv file with 2 columns:
    - First column is a set of sentences;
    - Second column are the labels (strings) associated to the sentences.
    
    NB:
    - The delimiter is a comma;
    - The csv file must have a header;
    - The first column is the index column;
    """
    if args.do_train and args.training_filepath is not None:
        print("Loading training data...")
        filepath = args.training_filepath
    elif args.do_eval and args.eval_filepath is not None:
        print("Loading validation data...")
        filepath = args.eval_filepath
    else:
        print("Error: Neither training or validation file provided with appropriate flag.")
        sys.exit()
    
    # Load the dataset into a pandas dataframe.
    df = pd.read_csv(filepath, delimiter=',', index_col=0)
    
    # Rename columns.
    df.columns = ['Sentence', 'Class']

    # Keep only rows with class of interest.
    if interest_classes is not None:
        df = df[df.Class.isin(interest_classes)]
        
    # Deal with duplicates.
    df.drop_duplicates(subset=['Sentence', 'Class'], keep='first', inplace=True)  # For duplicated queries with same class, keep first instance.
    df.drop_duplicates(subset=['Sentence'], keep=False, inplace=True)  # For duplicated queries with different classes, remove them.
    df.reset_index(drop=True, inplace=True)
    
    # Create a balanced dataset.
    if args.do_train and args.balanced:
        # Get the maximum number of samples of the smaller class. 
        # Note that the classes with under 1500 samples are not taken into account.
        count = df['Class'].value_counts()
        count = count[count > 1500]
        nb_samples = min(count)

        # Randomly select 'nb_samples' for all classes.
        balanced_df = pd.DataFrame(columns=['Sentence', 'Class'])
        for i, cat in enumerate(count.index.tolist()):
            tmp_df = df[df['Class']==cat].sample(n=nb_samples, replace=False, random_state=2)
            balanced_df = pd.concat([balanced_df,tmp_df], ignore_index=True)
        df = balanced_df.copy(deep=True)

    # Add categories ids column.
    categories = df.Class.unique()
    df['Class_id'] = df.apply(lambda row: np.where(categories == row.Class)[0][0], axis=1)
    
    # Save mapping between class and id.
    mapping = dict(enumerate(categories))
    with open(os.path.join(args.output_dir, 'map_classes.json'), 'w') as f:
        json.dump(mapping, f)
    
    return df, categories


def tokenize_sentences(tokenizer, df):
    """
    Tokenize all sentences in dataset with BertTokenizer.
    """    
    # Tokenize each sentence of the dataset.
    tokenized = df['Sentence'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    
    lengths = [len(i) for i in tokenized]
    max_len = max(lengths) if max(lengths) <= 512 else 512
    
    # Pad and truncate our sequences so that they all have the same length, max_len.
    print('Max sentence length: {}'.format(max_len))
    print('-> Padding/truncating all sentences to {} tokens...'.format(max_len))
    tokenized = pad_sequences(tokenized, maxlen=max_len, dtype="long", 
                              value=0, truncating="post", padding="post") # "post" indicates that we want to pad and truncate at the end of the sequence, as opposed to the beginning.
    
    return tokenized


def create_masks(tokenized):
    """
    Given a list of tokenized sentences, create the corresponding attention masks.
    - If a token ID is 0, then it's padding, set the mask to 0.
    - If a token ID is > 0, then it's a real token, set the mask to 1.
    """
    attention_masks = []
    for sent in tokenized:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


def split_data(dataset, test_percent, seed):
    """
    Split dataset to train/test.
    """
    tokenized, class_ids, attention_masks, sentences = dataset
    
    if test_percent < 0.0 or test_percent > 1.0:
        print("Error: '--test_percent' must be between [0,1].")
        sys.exit()
        
    # Use 90% for training and 10% for validation.
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(tokenized, class_ids, 
                                                                random_state=seed, test_size=test_percent)
    # Do the same for the masks.
    train_masks, val_masks, _, _ = train_test_split(attention_masks, class_ids,
                                                 random_state=seed, test_size=test_percent)
    # Do the same for the sentences.
    train_sentences, val_sentences, _, _ = train_test_split(sentences, class_ids,
                                                 random_state=seed, test_size=test_percent)
    
    return (train_inputs, train_labels, train_masks, train_sentences), (val_inputs, val_labels, val_masks, val_sentences)
    

def create_dataloader(dataset, batch_size, training_data=True):
    """
    """
    inputs, labels, masks, _ = dataset
                                                       
    # Convert all inputs and labels into torch tensors, the required datatype for our model.
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)                                                  
    
    # Create the DataLoader.
    data = TensorDataset(inputs, masks, labels)
    if training_data:                                           
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)                                               
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    
    return data, sampler, dataloader


def compute_metrics(preds, labels, classes):
    """
    Compute metrics for the classification task.
    """
    # Create dict to store scores.
    result = dict()
    
    # Accuracy.
    result['Accuracy'] = accuracy_score(y_true=labels, y_pred=preds)  #accuracy = (preds==labels).mean()
    
    # NB: Averaging methods:
    #  - "macro" simply calculates the mean of the binary metrics, giving equal weight to each class.
    #  - "weighted" accounts for class imbalance by computing the average of binary metrics in which each class’s score is weighted by its presence in the true data sample.
    #  - "micro" gives each sample-class pair an equal contribution to the overall metric.
    # Precision.
    result['Precision'] = precision_score(y_true=labels, y_pred=preds, average='macro')
    
    # Recall.
    result['Recall'] = recall_score(y_true=labels, y_pred=preds, average='macro')
    
    # F1 score.
    result['F1 score'] = f1_score(y_true=labels, y_pred=preds, average='macro')
    
    # Matthews correlation coefficient (MCC): used for imbalanced classes.
    result['MCC'] = matthews_corrcoef(y_true=labels, y_pred=preds)
    
    # Confusion matrix.
    conf_matrix = confusion_matrix(y_true=labels, y_pred=preds, normalize='true', labels=range(len(classes)))
    result['conf_matrix'] = conf_matrix.tolist()
    
    return result


def plot_confusion_matrix(cm, classes, outdir):
    """
    This function prints and plots the confusion matrix.
    """
    cm = np.array(cm)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    plt.figure(figsize = (10,7))
    ax = sn.heatmap(df_cm, annot=True)
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, horizontalalignment='right', rotation=45) 
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    
    plt.title('Confusion matrix', fontsize=18)
    plt.ylabel('True labels', fontsize=12)
    plt.xlabel('Predicted labels', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(outdir+"confusion_matrix.pdf")
    plt.close()
    return


def analyze_predictions(preds, labels, sentences):
    """
    Analyze more deeply the right and wrong predictions of the model on the dev set.
    """
    # Get the wrong predictions.
    indices_wrong = np.where(preds!=labels)[0]
    sentences_wrong = [sentences[i] for i in indices_wrong]
    labels_wrong = [labels[i] for i in indices_wrong]
    preds_wrong = [preds[i] for i in indices_wrong]
    df_wrong = pd.DataFrame(list(zip(sentences_wrong, labels_wrong, preds_wrong)),
                            columns =['Sentence', 'Class_id', 'Prediction_id'])
    
    # Get the right predictions.
    indices_right = np.where(preds==labels)[0]
    sentences_right = [sentences[i] for i in indices_right]
    labels_right = [labels[i] for i in indices_right]
    preds_right = [preds[i] for i in indices_right]
    df_right = pd.DataFrame(list(zip(sentences_right, labels_right, preds_right)),
                            columns =['Sentence', 'Class_id', 'Prediction_id'])
    return df_wrong, df_right
    

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def set_seed(seed):
    """
    Set seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def train(args, model, tokenizer, dataset, tb_writer, categories):
    """
    """
    if args.do_eval and args.eval_filepath is None:
        print("No validation file given: splitting dataset to train/test datasets...\n")
        train_dataset, validation_dataset = split_data(dataset, args.test_percent, args.seed)
    else:
        train_dataset = dataset
    
    print("Creating training dataloader...\n")
    train_data, train_sampler, train_dataloader = create_dataloader(train_dataset, args.batch_size, training_data=True)
                                                       
    # Setting up Optimizer & Learning Rate Scheduler.
    optimizer = AdamW(model.parameters(),
                  lr = args.learning_rate,
                  eps = args.adam_epsilon
                )
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    # Init some useful variables.
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    # For each epoch...
    t = time.time()
    for epoch_i in range(0, args.num_epochs):
        
        # Perform one full pass over the training set.
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.num_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Unpack this training batch from our dataloader. 
            # As we unpack the batch, we'll also copy each tensor to the GPU using the `to` method.
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(args.device)
            b_input_mask = batch[1].to(args.device)
            b_labels = batch[2].to(args.device)

            # Always clear any previously calculated gradients before performing a backward pass. 
            # PyTorch doesn't do this automatically because accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the loss value out of the tuple.
            loss = outputs[0]
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end. 
            # `loss` is a Tensor containing a single value; the `.item()` function just returns the Python value from the tensor.
            tr_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            
            # Update global step.
            global_step += 1
            
            # Progress update every 'logging_steps' batches.
            if args.logging_steps > 0 and step != 0 and step % args.logging_steps == 0:
                
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Compute average training loss over the last 'logging_steps'. Write it to Tensorboard.
                loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                tb_writer.add_scalar('Train/Loss', loss_scalar, global_step)
                logging_loss = tr_loss
                
                # Print the log.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Training loss: {:.2f}'.format(step, len(train_dataloader), elapsed, loss_scalar))

        print("  Training epoch took: {:}\n".format(format_time(time.time() - t0)))
        
        if args.do_eval and args.eval_filepath is None:
            print("Running Validation...")
            # After the completion of each training epoch, measure our performance on our validation set.
            t0 = time.time()
            result, df_wrong, df_right = evaluate(args, model, validation_dataset, categories)
            
            # Write results to tensorboard.
            tb_writer.add_scalar('Test/Accuracy', result['Accuracy'], epoch_i + 1)
            tb_writer.add_scalar('Test/Recall', result['Recall'], epoch_i + 1)
            tb_writer.add_scalar('Test/Precision', result['Precision'], epoch_i + 1)
            tb_writer.add_scalar('Test/F1 score', result['F1 score'], epoch_i + 1)
            tb_writer.add_scalar('Test/MCC', result['MCC'], epoch_i + 1)
            
            # Plot confusion matrix.
            plot_confusion_matrix(result['conf_matrix'], categories, args.output_dir)
            
            # Save dataframes of wrong and right predictions for further analysis.
            df_wrong.to_csv(os.path.join(args.output_dir, 'preds_wrong.csv'))
            df_right.to_csv(os.path.join(args.output_dir, 'preds_right.csv'))
            
            print("  Validation took: {:}\n".format(format_time(time.time() - t0)))
            
    print("Training complete!  Took: {}\n".format(format_time(time.time() - t)))
        
    #print("Saving model to {}...\n.".format(args.output_dir))
    #model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #model_to_save.save_pretrained(args.output_dir)
    #tokenizer.save_pretrained(args.output_dir)
    return model


def evaluate(args, model, validation_dataset, categories):
    """
    """    
    #Creating validation dataloader.
    validation_data, validation_sampler, validation_dataloader = create_dataloader(validation_dataset, args.batch_size, training_data=False)
    
    # Get validation sentences.
    validation_sentences = validation_dataset[3]
    
    # Tracking variables
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    
    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Add batch to GPU.
        b_input_ids, b_input_mask, b_labels = tuple(t.to(args.device) for t in batch)

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            # token_type_ids is the same as the "segment ids", which differentiates sentence 1 and 2 in 2-sentence tasks.
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU and store them.
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = b_labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, b_labels.detach().cpu().numpy(), axis=0)
                
        # Track the number of batches
        nb_eval_steps += 1
            
    # Take the max predicitions.
    preds = np.argmax(preds, axis=1)
    
    # Report results.
    result = compute_metrics(preds, out_label_ids, categories)
    print("  * Accuracy: {0:.6f}".format(result['Accuracy']))
    print("  * Recall: {0:.6f}".format(result['Recall']))
    print("  * Precision: {0:.6f}".format(result['Precision']))
    print("  * F1 score: {0:.6f}".format(result['F1 score']))
    print("  * MCC: {0:.6f}".format(result['MCC']))

    # Get wrong and right predictions.
    df_wrong, df_right = analyze_predictions(preds, out_label_ids, validation_sentences)
    
    return result, df_wrong, df_right


def evaluate_bert_preds(args, model, tokenizer, categories):
    """
    Temporary hard-coded evaluation on predictions from Bert-base.
    """
    # Load queries that Bert-base classified correclty.    
    df_bert_right_preds = pd.read_csv('./output/bert_base_cased/eval_right_preds.csv', delimiter=',', index_col=0)
    df_bert_right_preds['Class_id'] = df_bert_right_preds.apply(lambda row: np.where(categories == row.Class)[0][0], axis=1)
    bert_right_preds_tokenized = tokenize_sentences(tokenizer, df_bert_right_preds)
    bert_right_preds_attention_masks = create_masks(bert_right_preds_tokenized)
    bert_right_preds_dataset = (bert_right_preds_tokenized, df_bert_right_preds.Class_id.values, bert_right_preds_attention_masks, df_bert_right_preds.Sentence.values)
    result, df_wrong, df_right = evaluate(args, model, bert_right_preds_dataset, categories)
    df_wrong.to_csv(os.path.join(args.output_dir, 'bert_right_netbert_wrong.csv'))
    df_right.to_csv(os.path.join(args.output_dir, 'bert_right_netbert_right.csv'))
    with open(os.path.join(args.output_dir, 'scores_bert_right_preds.json'), 'w') as f:
        json.dump(result, f)
        
    # Load queries that Bert-base classified wrongly.
    df_bert_wrong_preds = pd.read_csv('./output/bert_base_cased/eval_wrong_preds.csv', delimiter=',', index_col=0)
    df_bert_wrong_preds['Class_id'] = df_bert_wrong_preds.apply(lambda row: np.where(categories == row.Class)[0][0], axis=1)
    bert_wrong_preds_tokenized = tokenize_sentences(tokenizer, df_bert_wrong_preds)
    bert_wrong_preds_attention_masks = create_masks(bert_wrong_preds_tokenized)
    bert_wrong_preds_dataset = (bert_wrong_preds_tokenized, df_bert_wrong_preds.Class_id.values, bert_wrong_preds_attention_masks, df_bert_wrong_preds.Sentence.values)
    result, df_wrong, df_right = evaluate(args, model, bert_wrong_preds_dataset, categories)
    df_wrong.to_csv(os.path.join(args.output_dir, 'bert_wrong_netbert_wrong.csv'))
    df_right.to_csv(os.path.join(args.output_dir, 'bert_wrong_netbert_right.csv'))
    with open(os.path.join(args.output_dir, 'scores_bert_wrong_preds.json'), 'w') as f:
        json.dump(result, f)
    return


def main(args):
    """
    """
    # Create tensorboard summarywriter.
    tb_writer = SummaryWriter()
    
    # Create output dir if none mentioned.
    if args.output_dir is None:
        model_name = os.path.splitext(os.path.basename(args.model_name_or_path))[0]
        args.output_dir = "./output/" + model_name + '/'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print("\n========================================")
    print('               Load model                 ')
    print("========================================\n")
    print("Loading BertForSequenceClassification model...\n")
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path, # Use the 12-layer BERT model, with an cased vocab.
        num_labels = args.num_labels, # The number of output labels
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        cache_dir = args.cache_dir,
    )
    print('Loading BertTokenizer...\n')
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    
    print("Setting up CUDA & GPU...")
    if torch.cuda.is_available():
        if args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)
            args.n_gpu = 1
            print("-> GPU training available! GPU {} {} will be used (no parallel training).\n".format(torch.cuda.get_device_name(args.gpu_id), args.gpu_id))
        else:
            args.n_gpu = torch.cuda.device_count()
            gpu_ids = list(range(0, args.n_gpu))
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[-1])
            print("-> GPU training available! Training will use GPU(s) {}\n".format(gpu_ids))
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0
        print("-> No GPU available, using the CPU instead.\n")
    model.to(args.device)  # Tell pytorch to run the model on the device.
    
    # Set the seed value all over the place to make this reproducible.
    set_seed(args.seed)
    
    
    print("\n========================================")
    print('               Processing data            ')
    print("========================================\n")
    classes_of_interest = ['Data Sheets',
                            'Configuration (Guides, Examples & TechNotes)',
                            'Install & Upgrade Guides',
                            'Release Notes',
                            'End User Guides']
    df, categories = load_data(args, classes_of_interest)
    
    # Get all sentences, their associated class and class_id.
    sentences = df.Sentence.values
    classes = df.Class.values
    class_ids = df.Class_id.values
    print('  - Number of sentences: {:,}'.format(df.shape[0]))
    print('  - Number of doc types: {:,}'.format(len(categories)))
    for i, cat in enumerate(categories):
        print("     * {} : {}".format(cat, i))
        
    print("Tokenizing sentences...")
    tokenized = tokenize_sentences(tokenizer, df)
    attention_masks = create_masks(tokenized)
    
    dataset = (tokenized, class_ids, attention_masks, sentences)
    if args.do_train:
        print("\n========================================")
        print('            Launching training            ')
        print("========================================\n")
        model = train(args, model, tokenizer, dataset, tb_writer, categories)
        
        # Hard-coded evaluation after training (temporary because loading fine-tuned model gives weird results)
        evaluate_bert_preds(args, model, tokenizer, categories)
        
        # NB: For unknown reason, saving the fine-tuned model, then loading it
        # and running an evaluation on the same test file leads to accuracy of
        # 0.17 while it was 0.88 after training. I suspect the BertForSequenceClassification
        # model not to save properly all its parameters (maybe juste loading the weights of
        # BERT and not the classifier MLP above).
        
    
if __name__=="__main__":
    args = parse_arguments()
    main(args)
