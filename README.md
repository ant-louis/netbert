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
The domain-specific corpus was collected by scraping all the text content from [cisco.com](https://www.cisco.com/), the Cisco confidential employee website. It resulted in about 30GB of uncleaned text, collected from 442,028 web pages in total. The pre-processing of the original corpus results in a cleaned dataset of about 170.7M sentences, for a total size of 22.7GB. This dataset is further split into train/validation/test sets with a ratio 90\%-5\%-5\% respectively.

<center>
    
|         | Sentences  | Words   | Data size |
|---------|------------|---------|-----------|
|**Train**| 145.9M     | 3.1B    | 20.4GB    |
|**Val**  | 8.8M       | 192.3M  | 1.2GB     |
|**Test** | 8.4M       | 182.2M  | 1.1GB     |

</center>

The model pre-training was performed on one machine with 8×32GB NVIDIA Tesla V100 GPUs and implemnted using the 🤗 [Transformers](https://github.com/huggingface/transformers) library.  The model trained continuously for 20 epochs (i.e., 1.9M training steps) which took a total of 29 days. The resulting perplexities are given below for BERT and NetBERT after 3, 12 and 20 epochs, respectively:

|           | BERT   | NetBERT-3 | NetBERT-12 | NetBERT-20 |
|-----------|--------|-----------|------------|------------|
| **Train** | 34.618 | 1.423     | 1.298      | 1.253      |
| **Val**   | 34.674 | 1.420     | 1.302      | 1.258      |
| **Test**  | 34.456 | 1.416     | 1.302      | 1.259      |


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
