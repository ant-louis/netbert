# NetBERT: A Pre-trained Language Representation Model for Computer Networking

This repository provides the code for pre-training and fine-tuning NetBERT, a computer networking language representation model designed for networking text mining tasks. 

## Table of contents
1. [Motivation](#motivation)
2. [Datasets](#datasets)
3. [Pre-training](#pretraining)
4. [Experiments](#experiments)
    4.1. [Text Classification](#text_classification)
    4.2. [Information Retrieval](#info_retrieval)
    4.3. [Word Similarity](#word_similarity)
    4.4. [Word Analogy](#word_analogy)
5. [Search](#search)

## 1. Motivation <a name="intro"></a>
Text mining is becoming increasingly important at Cisco as the number of product documents becomes larger and larger. Being able to retrieve the right information in the sorthest time possible is crucial, as it would increase the productivity of Cisco's employees by taking away the tedious task of searching the information among long technical documents.

While recent advancements in natural language processing (NLP) has allowed major improvements for various text mining tasks, applying them directly to Cisco documents often yields to unsatisfactory results due to a word distribution shift from general domain corpora to Cisco computer networking corpora. 

Therefore, we introduce NetBERT, a domain-specific language representation model pre-trained on large-scale Cisco corpora.


## 2. Datasets <a name="datasets"></a>
The original dataset used for pre-training BERT consists of all text content scrapped from [cisco.com](https://www.cisco.com/), resulting in about 30GB of uncleaned text data. This dataset is further preprocessed before training. The final dataset has the following properties:

|         | Documents  | Sentences  | Words   | Chars | Size   |
|---------|------------|------------|-------- |-------|--------|
|**Train**| 383.9K     | 145.9M     | 3.1B    | 21.7B | 20.4GB |
|**Dev**  | 21.3K      | 8.8M       | 192.3M  | 1.2B  | 1.2GB  |
|**Test** | 21.3K      | 8.4M       | 182.2M  | 1.1B  | 1.1GB  |


## 3. Pre-training <a name="pretraining"></a>
The pre-training of BERT is done using the [transformers](https://github.com/huggingface/transformers) library.
On 8 GPUs NVIDIA Tesla V100-SXM2 32GB, it takes about 36 hours to train the model over one epoch.

*Currently training...*


## 4. Experiments <a name="experiments"></a>

### 4.1. Text Classification <a name="text_classification"></a>
*Coming up...*

### 4.2. Information Retrieval <a name="info_retrieval"></a>
*Coming up...*

### 4.3. Word Similarity <a name="word_similarity"></a>
*Coming up...*

### 4.4. Word Analogy <a name="word_analogy"></a>
*Coming up...*

## 5. Search <a name="search"></a>
*Coming up...*
