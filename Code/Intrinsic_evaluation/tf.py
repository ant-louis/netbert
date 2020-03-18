import glob
import time
import datetime
import argparse
from tqdm import tqdm

import string
from collections import Counter

import nltk
nltk.download('stopwords')



def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", 
                        type=str, 
                        default='/raid/antoloui/Master-thesis/Data/Cleaned/Tmp/',
                        help="Path of the input directory."
                       )
    parser.add_argument("--output_dir", 
                        type=str, 
                        default='/raid/antoloui/Master-thesis/Data/Cleaned/Tmp/',
                        help="Path of the output directory."
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


def load_sentences(input_dir):
    """
    Given a file of raw sentences, return the list of these sentences.
    """
    sentences = []
    files = glob.glob(input_dir + '*.raw')
    for file in files:
        with open(file) as f:
            sentences.extend(f.readlines())
    return sentences


def get_words(sentences):
    """
    Given a list of sentences, extract the words while removing stopwords
    and punctuations.
    """
    # Stopwords and punctuations.
    stop = nltk.corpus.stopwords.words('english') + list(string.punctuation)
    
    all_words = []
    for sent in tqdm(sentences, desc='  Sentences'):
        words = [word for word in nltk.word_tokenize(sent.lower()) if word not in stop]
        all_words.extend(words)
    #all_words = [word for sent in sentences for word in nltk.word_tokenize(sent.lower()) if word not in stop]
    return all_words
    

def main(args):
    """
    """
    print("===================================================")
    print("Loading corpus...")
    print("===================================================")
    t0 = time.time()
    sentences = load_sentences(args.input_dir)
    print("   {} sentences loaded. -  Took: {:}\n".format(len(sentences), format_time(time.time() - t0)))
    
    print("===================================================")
    print("Removing stopwords and punctuation...")
    print("===================================================")
    t0 = time.time()
    words = get_words(sentences)
    print("   {} words processed. -  Took: {:}\n".format(len(words), format_time(time.time() - t0)))
    
    print("===================================================")
    print("Computing frequency of each word...")
    print("===================================================")
    t0 = time.time()
    word_freq = Counter(words)
    common_words = word_freq.most_common(100)
    print("   Word frequency computed. -  Took: {:}\n".format(format_time(time.time() - t0)))
    
    
    
if __name__=="__main__":
    args = parse_arguments()
    main(args)
