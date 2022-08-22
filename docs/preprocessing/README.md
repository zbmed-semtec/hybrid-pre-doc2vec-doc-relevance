# Hybrid approach phrase preprocessing

In this tutorial, we will cover the phrase preprocessing step for the hybrid-dictionary-ner approach. An alternative (and compatible) more general preprocessing can be found in the medline-preprocessing repository in [this tutorial](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/docs/Phrase_Preprocessing_Tutorial/tutorial_phrase_preprocessing.ipynb).



# Prerequisites

1. Retrieve TSV files from TREC or RELISH data sets

    - Use the recommended BioC-approach to generate .tsv files from the data sets.
    
    - Remove structure words, by using the Structure_Words_removal module in [this tutorial](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Structure_Words_removal)..

# Steps

## Step 1: Imports

First, we need to import the libraries from the code folder. To do so, change the `repository_path` variable to indicate the root path of the repository:


```python
import os
import sys

repository_path = os.path.expanduser("~/hybrid-dictionary-ner-doc2vec-doc-relevance")

sys.path.append(f"{repository_path}/code/preprocessing/")
os.chdir(repository_path)

import logging
import preprocess as pp

logging.basicConfig(format='%(asctime)s %(message)s')
```

## Step 2: Load the data and preprocess

After reading the input data, the preprocessing function is executed. The code process is described later in "Code Strategy" section.


```python
input_path = "data/preprocessing/RELISH/RELISH_documents_20220628_ann_swr.tsv"
#input_path = "data/preprocessing/TREC/TREC_documents_20220628_ann_swr.tsv"

#input_path = "../data_full/RELISH/RELISH_documents_20220628_ann_swr.tsv"
#input_path = "../data_full/TREC/TREC_documents_20220628_ann_swr.tsv"

data = pp.read_data(input_path)
data = pp.preprocess_data(data)
```

## Step 3: Save the preprocess data

The output is either stored in `.tsv` or `.npy` format. The `.tsv` is smaller in disk and faster to write.


```python
output_path = "data/preprocessing/RELISH/RELISH_tokens.tsv"
#output_path = "data/preprocessing/TREC/TREC_tokens.tsv"

#output_path = "../data_full/RELISH/RELISH_tokens.tsv"
#output_path = "../data_full/TREC/TREC_tokens.tsv"

pp.save_output(data, output_path, npy_format=False)
#pp.save_output(data, output_path, npy_format=True)
```

# Decision notes

## Code strategy

1. The input file must be in `.tsv` format, containing three columns: "PMID", "title", "abstract". The file is recommended to be pruned of structure words following [this tutorial](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Structure_Words_removal).

2. The code loops through every row of the input `.tsv` and preprocess the title and abstract separately. 

3. The preprocess consists in:
    
    * (Optional) If the function parameter `process_abv` is set to `True` (`False` by default), an experimental abbreviation algorithm is executed to find and combine terms like "E. coli" or "S. aureus" into a single word like "e.coli". This process is not well tested and not recommended unless necessary.

    * Lowercase everything. 

    * Tokenize space-separated words. The text is split by white spaces and only alphanumeric characters and allowed punctuation is kept.

    * Removes all special character except for the hyphens `-` (this can be modified using the function parameter `allowed_punctuation`).

    * (Optional) Saves the results as a three-dimensional numpy array and saves it as a `.npy` file.

    * (Optional) Saves the results as a three column `.tsv` file containing "PMID", "title" and "abstract".

## Decisions

* Instead of using the default phrase preprocessing found in medline-preprocessing ([here](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/docs/Phrase_Preprocessing_Tutorial/tutorial_phrase_preprocessing.ipynb)), the steps produced in here are particular for the hybrid-dictionary-ner approach. The results produced are expectd to be the same, but execution time is greatly improved in this approach. The main difference is to not include the biological tokenizer `en_core_sci_lg` from the sciSpacy module, since its use is not recommended in this approach.

* For the decisions related to the actual preprocess steps followed, please refer to the main documentation in [here](https://github.com/zbmed-semtec/medline-preprocessing#cleaning-for-word-embedding).

## Notes

The time to preprocess each dataset (TREC or RELISH) using 1 core of an Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz with 16GB of RAM running Ubuntu 20.04 LTS and Python 3.8.10 is:

* RELISH (163189 publications): 2min 31s ± 538 ms on average.

* TREC (32604 publications): 26.9 s ± 203 ms on average.

**REMOVE THIS LINE BEFORE FINAL VERSION COMMIT**


```python
#!jupyter nbconvert docs/preprocessing/tutorial_preprocessing.ipynb --to markdown --output README.md
```

    [NbConvertApp] Converting notebook docs/preprocessing/tutorial_preprocessing.ipynb to markdown
    [NbConvertApp] Writing 4914 bytes to docs/preprocessing/README.md

