# Doc2Vec Model and embeddings generation

In this tutorial, we will cover the generation of the Doc2Vec model for the hybrid-dictionary-ner approach. The aim is to produce embeddings for each RELISH and TREC publication.

# Prerequisites

1. Preprocessed tokens in .npy format or .tsv format. They can be generated following the [preprocessing tutorial](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/blob/main/docs/preprocessing/tutorial_preprocessing.ipynb).

# Steps

## Step 1: Imports

First, we need to import the libraries from the code folder. To do so, change the `repository_path` variable to indicate the root path of the repository:


```python
%load_ext autoreload
%autoreload 2

import os
import sys

repository_path = os.path.expanduser("~/hybrid-dictionary-ner-doc2vec-doc-relevance")

sys.path.append(f"{repository_path}/code/embeddings/")
os.chdir(repository_path)

import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import create_model as cm

logging.basicConfig(format='%(asctime)s %(message)s')

```

## Step 2: Loading the data

Next, we need to import the preprocessed tokens. An small sample is provided in the data folder. The `load_tokens()` function returns the title and abstract combined in one document:


```python
tokens_path = "data/embeddings/RELISH/RELISH_tokens.tsv"
pmid, join_text = cm.load_tokens(tokens_path)
```

We need to create the `TaggedDocuments` required by `Doc2Vec` to generate and train the models:


```python
tagged_data = cm.generate_TaggedDocument(pmid, join_text)
```

## Step 3: Creating the model

First, we need to choose the hyperparameters of the model. In this tutorial, we will only consider one combination of hyperparameters (for the hyperparameter optimization, please refer to the [tendency analysis](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/blob/main/docs/tendency_analysis/tutorial_tendency_analysis.ipynb) tutorial). The easiest way to indicate the hyperparameters is to create a dictionary with the [available options](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec):


```python
params_d2v = {
    "dm": 0,
    "vector_size": 200, 
    "window": 5, 
    "min_count": 5, 
    "epochs": 10, 
    "workers": 4}
```

To create the model, use the `generate_doc2vec_model()` function with the tagged data and the model parameters. This function automatically creates the required vocabulary for training:


```python
model = cm.generate_doc2vec_model(tagged_data, params_d2v)
```

## Step 4: Training the model

The function `train_doc2vec_model` is responsible for training the previously generated model. The argument verbose determines the information to receive from the training process:


```python
cm.train_doc2vec_model(model, tagged_data, verbose=1)
```

    2022-08-01 10:24:07,013 --- Time to train: 0.26 seconds


The model can be stored to later be used by `save_doc2vec_model()` function:


```python
ouput_model_path = "data/embeddings/RELISH/RELISH_hybrid_d2v.model"
cm.save_doc2vec_model(model, ouput_model_path)
```

# Decision notes

## Code strategy

## Decisions

# TODO

* Include library dependencies in prerequisites.
