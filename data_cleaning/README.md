# Data cleaning

## 1. Clean up dataset
The following command clean a dataset of documents stored in a json file:
```
python cleanup_dataset.py --data_dir=<data_dir> --infile=<infile>
```
where
*--data_dir* indicates the path of the repository containing the json files to clean, and *--infile* indicates the name of the json file to clean. Note that one can clean simultaneously all the json files present in */<data_dir>* by running:
```
python cleanup_dataset.py --all=True
```

This script will clean the original dataset by:
- applying *fix_text* function from [ftfy](https://ftfy.readthedocs.io/en/latest/);
- replacing two or more spaces with one;
- removing sequences of special characters;
- removing small documents (less than 128 tokens);
- removing non-english documents with [langdetect](https://pypi.org/project/langdetect/).


## 2. Presplit sentences
The following command presplit each document stored a json file into sentences:
```
python presplit_sentences_json.py --data_dir=<data_dir> --infile=<infile>
```
where
*--data_dir* indicates the path of the repository containing the json files to presplit, and *--infile* indicates the name of the json file to presplit. Note that one can presplit simultaneously all the json files present in */<data_dir>* by running:
```
python presplit_sentences_json.py --all=True
```

This script will presplit each document in the given json file and perform additional cleaning on the individual sentences, namely:
- If sentence begins with  a number, remove the number;
- If line begins with a unique special char, remove that char;
- Keep only sentences with more than 2 words and less than 200 words.


## 3. Create train/dev/test data
The following command create the train/dev/test data in json form:
```
python create_train_dev_test_json.py --input_files <in1> <in2> --output_dir=<output_dir> --test_percent <%_train> <%_dev> <%_test>
```

## 4. Export json documents to raw text
The following command convert the json file containing all documents in raw text:
```
python json2text.py --json_file=<json_file> --output_file=<output_file>
```
