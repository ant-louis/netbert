[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# NetBERT

**NetBERT** is a "small" BERT model pre-trained on a huge corpus of *computer networking* text (~23Gb). NetBERT demonstrate clear improvements over BERT on the following two representative text mining tasks: 
- *Computer Networking Text Classification* (0.9% F1 improvement);
- *Computer Networking information retrieval* (12.3% improvement on a custom retrieval score). 
Additional experiments on *Word Similarity* and *Word Analogy* tend to show that NetBERT capture more meaningful semantic properties and relations between networking concepts than BERT does.

## Table of contents
1. [Using NetBERT](#using_netbert)
2. [Pre-training](#pretraining)
    2.1. [Data](#data)
    2.2. [Hardware and Schedule](#hardware)
    2.3. [Results](#results)
3. [Experiments](#experiments)
    1. [Text Classification](#text_classification)
    2. [Information Retrieval](#info_retrieval)
    3. [Word Similarity](#word_similarity)
    4. [Word Analogy](#word_analogy)
4. [Search](#search)




### 1. Using NetBERT <a name="using_netbert"></a>
You can use NetBERT with [🤗 transformers](https://github.com/huggingface/transformers) library as follows:

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pretrained model and tokenizer
model = BertForMaskedLM.from_pretrained("antoiloui/netbert")
tokenizer = BertTokenizer.from_pretrained("antoiloui/netbert")
```

### 2. Pre-training <a name="pretraining"></a>

#### 2.1. Data <a name="data"></a>
The domain-specific corpus was collected by scraping all the text content from [cisco.com](https://www.cisco.com/). It resulted in about 30GB of uncleaned text, collected from 442,028 web pages in total. The pre-processing of the original corpus results in a cleaned dataset of about 170.7M sentences, for a total size of 22.7GB. For confidentiality reasons, this data is not shared.

#### 2.2. Hardware and Schedule <a name="hardware"></a>
The model pre-training was performed on one machine with 8×32GB NVIDIA Tesla V100 GPUs. The model trained continuously for 20 epochs (i.e., 1.9M training steps) which took a total of 29 days.

#### 2.3. Results <a name="results"></a>
The resulting perplexities are given below for BERT and NetBERT after 3, 12 and 20 epochs, respectively:


|           | BERT   | NetBERT-3 | NetBERT-12 | NetBERT-20 |
|-----------|--------|-----------|------------|------------|
| **Train** | 34.618 | 1.423     | 1.298      | 1.253      |
| **Val**   | 34.674 | 1.420     | 1.302      | 1.258      |
| **Test**  | 34.456 | 1.416     | 1.302      | 1.259      |


### 3. Experiments <a name="experiments"></a>

#### Text Classification <a name="text_classification"></a>
*Coming up...*

#### Information Retrieval <a name="info_retrieval"></a>
*Coming up...*

#### Word Similarity <a name="word_similarity"></a>
*Coming up...*

#### Word Analogy <a name="word_analogy"></a>
*Coming up...*

