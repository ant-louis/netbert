![](https://img.shields.io/github/license/antoiloui/netbert)

# NetBERT 📶

**A BERT-base model pre-trained on a huge corpus of computer networking text (~23Gb)**.

NetBERT demonstrate clear improvements over BERT on the following two representative text mining tasks:

- Computer Networking *Text Classification* (**0.9%** F1 improvement);
- Computer Networking *Information Retrieval* (**12.3%** improvement on a custom information retrieval score).

Additional experiments on *Word Similarity* and *Word Analogy* tend to show that NetBERT capture more meaningful semantic properties and relations between networking concepts than BERT does. For more information, you cand [download](https://matheo.uliege.be/bitstream/2268.2/9060/7/Antoine_Louis_Thesis.pdf) my thesis.

## Table of contents

1. [Usage](#usage)
2. [Data Cleaning](#cleaning)
3. [Pre-training](#pretraining)
4. [Tasks](#tasks)
    1. [Text Classification](#text_classification)
    2. [Information Retrieval](#info_retrieval)
    3. [Word Similarity](#word_similarity)
    4. [Word Analogy](#word_analogy)

## 1. Usage <a name="usage"></a>
You can use NetBERT with [🤗 transformers](https://github.com/huggingface/transformers) library as follows:

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pretrained model and tokenizer
model = BertForMaskedLM.from_pretrained("antoiloui/netbert")
tokenizer = BertTokenizer.from_pretrained("antoiloui/netbert")
```

## 2. Data Cleaning <a name="cleaning"></a>

### Data

The *computer networking* corpus was collected by scraping all the text content from [cisco.com](https://www.cisco.com/). It resulted in about 30GB of uncleaned text, collected from **442,028** web pages in total. The pre-processing of the original corpus results in a cleaned dataset of about **170.7M** sentences, for a total size of **22.7GB**.

The following section describes how to run the cleaning scripts located in '*scripts/data_cleaning/*'.

### (a) Clean dataset

The following command clean a dataset of documents stored in a json file:

```python
python cleanup_dataset.py --data_dir=<data_dir> --infile=<infile>
```

where *--data_dir* indicates the path of the repository containing the json files to clean, and *--infile* indicates the name of the json file to clean. Note that one can clean simultaneously all the json files present in */<data_dir>* by running:

```python
python cleanup_dataset.py --all=True
```

This script will clean the original dataset by:

- applying *fix_text* function from [ftfy](https://ftfy.readthedocs.io/en/latest/);
- replacing two or more spaces with one;
- removing sequences of special characters;
- removing small documents (less than 128 tokens);
- removing non-english documents with [langdetect](https://pypi.org/project/langdetect/).

### (b) Presplit sentences

The following command presplit each document stored a json file into sentences:

```python
python presplit_sentences_json.py --data_dir=<data_dir> --infile=<infile>
```

where *--data_dir* indicates the path of the repository containing the json files to presplit, and *--infile* indicates the name of the json file to presplit. Note that one can presplit simultaneously all the json files present in */<data_dir>* by running:

```python
python presplit_sentences_json.py --all=True
```

This script will pre-split each document in the given json file and perform additional cleaning on the individual sentences, namely:

- If sentence begins with  a number, remove the number;
- If line begins with a unique special char, remove that char;
- Keep only sentences with more than 2 words and less than 200 words.

## (c) Create train/dev/test data

The following command create the train/dev/test data in json form:

```python
python create_train_dev_test_json.py --input_files <in1> <in2> --output_dir=<output_dir> --test_percent <%_train> <%_dev> <%_test>
```

## (d) Export json documents to raw text

The following command convert the json file containing all documents in raw text:

```python
python json2text.py --json_file=<json_file> --output_file=<output_file>
```

## 3. Pre-training <a name="pretraining"></a>

[*coming up...*]

## 4. Tasks <a name="tasks"></a>

### 3.1. Text Classification <a name="text_classification"></a>

[*coming up...*]

### 3.2. Information Retrieval <a name="info_retrieval"></a>

[*coming up...*]

### 3.3. Word Similarity <a name="word_similarity"></a>

[*coming up...*]

### 3.4. Word Analogy <a name="word_analogy"></a>

[*coming up...*]
