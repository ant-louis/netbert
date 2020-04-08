# NetBERT: a pre-trained computer networking language representation model

This repository provides the code for pre-training and fine-tuning NetBERT, a computer networking language representation model designed for networking text mining tasks such as named entity recognition (NER) or document classification. 

## Table of contents
1. [Motivation](#motivation)
2. [Datasets](#datasets)
3. [Pre-training](#pretraining)
4. [Fine-tuning](#finetuning)
5. [About me](#about)

## Motivation <a name="intro"></a>
Text mining is becoming increasingly important at Cisco as the number of product documents becomes larger and larger. Being able to retrieve the right information in the sorthest time possible is crucial, as it would increase the productivity of Cisco's employees by taking away the tedious task of searching the information among long technical documents.

While recent advancements in natural language processing (NLP) has allowed major improvements for various text mining tasks, applying them directly to Cisco documents often yields to unsatisfactory results due to a word distribution shift from general domain corpora to Cisco computer networking corpora. 

Therefore, we introduce NetBERT (Bidirectional Encoder Representations from Transformers for Computer Networking Text Mining), which is a domain-specific language representation model pre-trained on large-scale Cisco corpora.


## Datasets <a name="datasets"></a>
The original dataset used for pre-training BERT consists of all content of [cisco.com](https://www.cisco.com/) resulting in about 30GB of uncleaned text data. This dataset is further preprocessed before training (see [Cleaning](./Code/Cleaning/README.md) for detailed information about data cleaning). The resulting dataset has the following properties:

|         | Documents  | Sentences  | Words   | Chars | Size   |
|---------|------------|------------|-------- |-------|--------|
|**Train**| 383.9K     | 145.9M     | 3.1B    | 21.7B | 20.4GB |
|**Dev**  | 21.3K      | 8.8M       | 192.3M  | 1.2B  | 1.2GB  |
|**Test** | 21.3K      | 8.4M       | 182.2M  | 1.1B  | 1.1GB  |


## Pre-training <a name="pretraining"></a>
The pretraining of BERT is done using the [transformers](https://github.com/huggingface/transformers) library, on 8 GPUs NVIDIA Tesla V100-SXM2 32GB. It takes about 36 hours to train the model over one epoch.

*Currently training...*


## Fine-tuning <a name="pretraining"></a>

### Named Entity Recognition (NER)
*Coming up...*

### Document classification
*Coming up...*

## About <a name="about"></a>
This project takes place as part of my Master Thesis in Data Science & Engineering @University of Liège, Belgium. It is combined with an internship @Cisco.
