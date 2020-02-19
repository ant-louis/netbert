# Semantic Textual Similarity (STS)

## Sentences Similarity Search

As a first intrinsic evaluation of the BERT encodings, we perform a sentence similiraty search for each sentence in a text file in such way that for a particular sentence:
- The most dissimilar sentences are selected according to the edit distance (Levenshtein distance);
- Among these sentences, the top *k* most similar sentences are selected according to the cosine similarity.

The goal is to retrieve sentences that look different from the query sentence but seems to have the same content if the encoding is similar.

The sentences are encoded using the famous [bert-as-service](https://github.com/hanxiao/bert-as-service). Therefore, to run the script, follow the next steps:

1. Start bert-as-a-service server:

```
export ZEROMQ_SOCK_TMP_DIR=/tmp/
bert-serving-start -num_worker=1 -max_seq_len=NONE -model_dir <path_of_tf_model> ## bert-serving-start -num_worker=1 -max_seq_len=NONE -model_dir ../models/base_cased
```
Note that the tensorflow checkpoint needs to have a bert_model.ckpt file containing the pre-trained weights (which is actually 3 files), a vocab file (vocab.txt) to map WordPiece to word id, and a config file (bert_config.json) which specifies the hyperparameters of the model.

2. Run the following command:

```
python similarity_search.py
````