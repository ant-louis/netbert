[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# NetBERT

**NetBERT** is a "small" BERT model pre-trained on a huge corpus of *computer networking* text (~23Gb). NetBERT demonstrate clear improvements over BERT on the following two representative text mining tasks: 
- Computer Networking *Text Classification* (**0.9%** F1 improvement);
- Computer Networking *Information Retrieval* (**12.3%** improvement on a custom information retrieval score).

Additional experiments on *Word Similarity* and *Word Analogy* tend to show that NetBERT capture more meaningful semantic properties and relations between networking concepts than BERT does. For more information, you cand [download](https://matheo.uliege.be/bitstream/2268.2/9060/7/Antoine_Louis_Thesis.pdf) my thesis.

## Table of contents
1. [Using NetBERT](#using_netbert)
2. [Pre-training](#pretraining)
    1. [Data](#data)
    2. [Hardware and Schedule](#hardware)
    3. [Results](#results)
3. [Experiments](#experiments)
    1. [Text Classification](#text_classification)
    2. [Information Retrieval](#info_retrieval)
    3. [Word Similarity](#word_similarity)
    4. [Word Analogy](#word_analogy)


## 1. Using NetBERT <a name="using_netbert"></a>
You can use NetBERT with [🤗 transformers](https://github.com/huggingface/transformers) library as follows:

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pretrained model and tokenizer
model = BertForMaskedLM.from_pretrained("antoiloui/netbert")
tokenizer = BertTokenizer.from_pretrained("antoiloui/netbert")
```

## 2. Pre-training <a name="pretraining"></a>

### Data <a name="data"></a>
The *computer networking* corpus was collected by scraping all the text content from [cisco.com](https://www.cisco.com/). It resulted in about 30GB of uncleaned text, collected from **442,028** web pages in total. The pre-processing of the original corpus results in a cleaned dataset of about **170.7M** sentences, for a total size of **22.7**GB. For confidentiality reasons, this data is not shared.

### Hardware and Schedule <a name="hardware"></a>
The model pre-training was performed on one machine with **8×32**GB NVIDIA Tesla V100 GPUs. The model trained continuously for **20** epochs (i.e., 1.9M training steps) which took a total of **29** days.

### Results <a name="results"></a>
The resulting *perplexities* are given below for BERT and NetBERT after 3, 12 and 20 epochs, respectively:


|              | BERT   | NetBERT-3 | NetBERT-12 | NetBERT-20 |
|--------------|--------|-----------|------------|------------|
| *Train*      | 34.618 | 1.423     | 1.298      | **1.253**  |
| *Validation* | 34.674 | 1.420     | 1.302      | **1.258**  |
| *Test*       | 34.456 | 1.416     | 1.302      | **1.259**  |


## 3. Experiments <a name="experiments"></a>

### 3.1. Text Classification <a name="text_classification"></a>
This task aims at comparing the quality of both BERT and NetBERT embeddings by fine-tuning both models on a *computer networking* sentence classification task. Intuitively, if one model is able to predict the class of a sentence better than the other, it means that the word representations that have been learned by the former model capture a more accurate meaning of that sentence than those learned by the other model.

#### Dataset
The dataset used in this experiment was collected by the Cisco One Search team in San Jose, California. They gathered a set of actual search queries from Cisco employees, and labeled them with the type of document in which the information being sought was found. In total, the dataset contains about **48,000** queries labeled with seven different document types (e.g., "Configuration", "Data sheets", etc). The dataset was randomly split into train (80%), validation (10%) and test
(10%) sets.

#### Results
After finding the optimal hyperparameters (i.e., batch size, learning rate and number of epochs) on the validation set, both BERT and NetBERT were eventually fine-tuned on the train and validation sets, and evaluated on the test set. In order to report some notion of variability on the computed metrics, both models were
evaluated on 100 bootstrapped samples drawn from the test set (of the same size as the latter). The mean values as well as the standard deviations are summarized (in percent) in the following table:

| Metrics     | BERT       | NetBERT        |
|-------------|------------|----------------|
| *MCC*       | 88.3 (0.6) | **89.6** (0.6) |
| *Accuracy*  | 93.4 (0.3) | **94.1** (0.3) |
| *F1*        | 91.7 (0.5) | **92.1** (0.5) |
| *Precision* | 90.2 (0.7) | **91.6** (0.6) |
| *Recall*    | 90.9 (0.5) | **91.8** (0.5) |

### 3.2. Information Retrieval <a name="info_retrieval"></a>
[*coming up...*]

### 3.3. Word Similarity <a name="word_similarity"></a>
[*coming up...*]

### 3.4. Word Analogy <a name="word_analogy"></a>
[*coming up...*]

