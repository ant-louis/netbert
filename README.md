# NetBERT: A Pre-trained Language Representation Model for Computer Networking

Obtaining accurate information about products in a fast and efficient way is becoming increasingly important at Cisco as the related documentation rapidly grows. Thanks to recent progress in natural language processing (NLP), extracting valuable information from general domain documents has gained in popularity, and deep learning has boosted the development of effective text mining systems. However, directly applying the advancements in NLP to domain-specific documentation might yield unsatisfactory results due to a word distribution shift from general domain language to domain-specific language. Hence, this work aims to determine if a large language model pre-trained on domain-specific (computer networking) text corpora improves performance over the same model pre-trained exclusively on general domain text, when evaluated on in-domain text mining tasks.

To this end, we introduce NetBERT (Bidirectional Encoder Representations from Transform-ers for Computer Networking), a domain-specific language representation model based on BERT (Devlin et al., 2018) and pre-trained on large-scale computer networking corpora. Through several extrinsic and intrinsic evaluations, we compare the performance of our novel model against the general-domain BERT. We demonstrate clear improvements over BERT on the following two representative text mining tasks: networking text classification (0.9% F1 improvement) and networking information retrieval (12.3% improvement on a custom retrieval score). Additional experiments on word similarity and word analogy tend to show that NetBERT capture more meaningful semantic properties and relations between networking concepts than BERT does. We conclude that pre-training BERT on computer networking corpora helps it understand more accurately domain-related text.

## Table of contents
1. [Pre-training](#pretraining)
2. [Experiments](#experiments)
    1. [Text Classification](#text_classification)
    2. [Information Retrieval](#info_retrieval)
    3. [Word Similarity](#word_similarity)
    4. [Word Analogy](#word_analogy)
3. [Search](#search)


## 1. Pre-training <a name="pretraining"></a>
The original dataset used for pre-training BERT consists of all text content scrapped from [cisco.com](https://www.cisco.com/), resulting in about 30GB of uncleaned text data. This dataset is further processed and cleaned before pre-training. The final dataset has the following properties:

|         | Documents  | Sentences  | Words   | Chars | Size   |
|---------|------------|------------|-------- |-------|--------|
|**Train**| 383.9K     | 145.9M     | 3.1B    | 21.7B | 20.4GB |
|**Dev**  | 21.3K      | 8.8M       | 192.3M  | 1.2B  | 1.2GB  |
|**Test** | 21.3K      | 8.4M       | 182.2M  | 1.1B  | 1.1GB  |

The pre-training of BERT is done using the 🤗 [Transformers](https://github.com/huggingface/transformers) library.
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
