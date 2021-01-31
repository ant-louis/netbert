![](https://img.shields.io/github/license/antoiloui/netbert)

# NetBERT 📶

**A BERT model pre-trained on a huge corpus of computer networking text (~23Gb)**.

## Usage
You can use NetBERT with [🤗 transformers](https://github.com/huggingface/transformers):

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pretrained model and tokenizer
model = BertForMaskedLM.from_pretrained("antoiloui/netbert")
tokenizer = BertTokenizer.from_pretrained("antoiloui/netbert")
```

## Documentation

Detailed documentation on the pre-trained model, its implementation, and the data can be found [here](docs/index.md).

## Citation

For attribution in academic contexts, please cite this work as:

```
@mastersthesis{louis2020netbert,
    title={NetBERT: A Pre-trained Language Representation Model for Computer Networking},
    author={Louis, Antoine},
    year={2020},
    school={University of Liege}
}
```
