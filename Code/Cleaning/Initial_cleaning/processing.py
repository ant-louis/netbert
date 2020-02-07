from tqdm import tqdm
tqdm.pandas()
import spacy, en_core_web_sm
import numpy as np
import pandas as pd
import json
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')



def create_dataframe(id_file):
    """
    Load all json files and create dataframe.
    """
    rows = []
    file_path = "../../Data/Original/" + str(id_file) + ".json"

    with open(file_path) as f:
        data = json.load(f)  # data is a list of dict of the form: {'text':['...'], 'uri':['...']}

        print("Extracting text from {} documents in file '{}.json'...".format(len(data), id_file))
        for i, doc in enumerate(data):
            text = doc.get('text') # Get the text of the current doc
            if text is not None:
                row_dict = {'Text': text[0], 'Length': len(text[0])}
                rows.append(row_dict)
    return pd.DataFrame(rows)
    

def clean_corpus(df):
    """
    Clean corpus of text for all documents.
    """
    df.Text = df.Text.replace('\s+', ' ', regex=True)  # Remove duplicate spaces
    df.Text = df.Text.str.encode('ascii', 'ignore').str.decode('utf-8')   # Encode in ascii to remove weird characters such as \uf0a7
    #df.Text = df.Text.str.lower()  # Lower case all strings
    return df


def spacy_segmentation(doc_text):
    """
    Given a string, segment it by sentences (performed by Spacy).
    """
    nlp = en_core_web_sm.load()
    nlp.max_length = 2621500  # because larger document has a size of 2621440 char
    doc = nlp(doc_text)
    sentences = list(doc.sents)
    return [sent.text for sent in sentences]


def nltk_segmentation(doc_text):
    """
    Given a string, segment it by sentences (performed by nltk).
    """
    return sent_tokenize(doc_text)


def sent_cleaning(list_sent):
    """
    """
    # Remove sequences of special characters
    spec_char = set(',?;.:/=+%`¨*$€-_())°!§\'\"&@#~®†ºπ‡¬≈©◊~∞µ…÷≠<>^')
    list_sent = [' '.join([x for x in sent.split() if len(x)<=2 or not all(c in spec_char for c in x)]) for sent in list_sent]
    
    # If line begins with a number, remove the number   
    list_sent = [sent.split(maxsplit=1)[1] if (len(sent.split(maxsplit=1))>1 and sent.split(maxsplit=1)[0].isdigit()) else sent for sent in list_sent]
    
    # If line begins with a unique special char, remove that char
    list_sent = [sent.split(maxsplit=1)[1] if (len(sent.split(maxsplit=1))>1 and len(sent.split(maxsplit=1)[0])==1 and sent.split(maxsplit=1)[0] in spec_char) else sent for sent in list_sent]

    # Keep only sentences with more than 2 words and less than 200 words
    list_sent = [sent for sent in list_sent if (len(sent.split())>2 and len(sent.split())<200)]
    return list_sent


def sent_convert(list_sent):
    """
    Given a list of string sentences, return one unique string where
    sentences are separated by newlines.
    """
    return "\n".join(list_sent) 


def process_sentences(df):
    """
    """
    print("Segmenting sentences...")
    df['Text'] = df['Text'].progress_apply(nltk_segmentation)
    print("Cleaning sentences...")
    df['Text'] = df['Text'].progress_apply(sent_cleaning)
    print("Concatenating all sentences...")
    df['Text'] = df['Text'].progress_apply(sent_convert)
    return df


def save_sentences(df, id_file):
    """
    """
    sentences = "\n\n".join(df["Text"])
    output_file = "../../Data/Preprocessed/text_" + str(id_file) + ".txt"
    with open(output_file, "w+") as f:
        f.write(sentences)



if __name__ == "__main__":
    
    for id_file in tqdm(range(1, 14)):
        print("Creating dataframe for file '{}.json'...".format(id_file))
        df = create_dataframe(id_file)
        print("Dataframe created with a total of {} documents.".format(len(df.index)))

        print("Cleaning corpus of text...")
        df = clean_corpus(df)
        print("Done !")

        print("Processing sentences...")
        df = process_sentences(df)
        print("Done !")

        print("Saving sentences to output file...")
        save_sentences(df, id_file)
        print("Done !")
   