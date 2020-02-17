# Semantic Textual Similarity (STS)

## Similarity search

1. Start bert-as-a-service server:

```
bert-serving-start -model_dir ./models/base_cased/ -num_worker=1 -max_seq_len=NONE
```

2. Run the following command:

```
python similarity_search.py
````