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
  author      = {Antoine Louis},
  title       = {NetBERT: A Pre-trained Language Representation Model for Computer Networking},
  school      = {University of Liège},
  address     = {Liège, Belgium},
  year        = {2020},
  url         = {http://hdl.handle.net/2268.2/9060}
}
```
