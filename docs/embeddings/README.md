# Doc2Vec Model and embeddings generation

In this tutorial, we will cover the generation of the Doc2Vec model for the hybrid-dictionary-ner approach. The aim is to produce embeddings for each RELISH and TREC publication.

# Prerequisites

1. Preprocessed tokens in .npy format or .tsv format. They can be generated following the [preprocessing tutorial](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/blob/main/docs/preprocessing/tutorial_preprocessing.ipynb).

# Steps

## Step 1: Imports

First, we need to import the libraries from the code folder. To do so, change the `repository_path` variable to indicate the root path of the repository:


```python
#%load_ext autoreload
#%autoreload 2

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
#tokens_path = "data/embeddings/TREC/TREC_tokens.tsv"

#tokens_path = "../data_full/RELISH/RELISH_tokens.tsv"
#tokens_path = "../data_full/TREC/TREC_tokens.tsv"

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
    "vector_size": 256, 
    "window": 7, 
    "min_count": 5, 
    "epochs": 10, 
    "workers": 8}
```

To create the model, use the `generate_doc2vec_model()` function with the tagged data and the model parameters. This function automatically creates the required vocabulary for training:


```python
model = cm.generate_doc2vec_model(tagged_data, params_d2v)
```

## Step 4: Training the model

The function `train_doc2vec_model` is responsible for training the previously generated model. The argument verbose determines the information to receive from the training process:


```python
cm.train_doc2vec_model(model, tagged_data, verbose=2)
```

    2022-08-26 10:54:16,190 	Epoch #0 start
    2022-08-26 10:54:16,220 	Epoch #0 end
    2022-08-26 10:54:16,221 	Epoch #1 start
    2022-08-26 10:54:16,248 	Epoch #1 end
    2022-08-26 10:54:16,248 	Epoch #2 start
    2022-08-26 10:54:16,274 	Epoch #2 end
    2022-08-26 10:54:16,275 	Epoch #3 start
    2022-08-26 10:54:16,305 	Epoch #3 end
    2022-08-26 10:54:16,305 	Epoch #4 start
    2022-08-26 10:54:16,340 	Epoch #4 end
    2022-08-26 10:54:16,340 	Epoch #5 start
    2022-08-26 10:54:16,370 	Epoch #5 end
    2022-08-26 10:54:16,370 	Epoch #6 start
    2022-08-26 10:54:16,408 	Epoch #6 end
    2022-08-26 10:54:16,409 	Epoch #7 start
    2022-08-26 10:54:16,447 	Epoch #7 end
    2022-08-26 10:54:16,448 	Epoch #8 start
    2022-08-26 10:54:16,477 	Epoch #8 end
    2022-08-26 10:54:16,478 	Epoch #9 start
    2022-08-26 10:54:16,508 	Epoch #9 end
    2022-08-26 10:54:16,508 --- Time to train: 0.32 seconds


The model can be stored to later be used by `save_doc2vec_model()` function:


```python
output_model_path = "data/embeddings/RELISH/RELISH_hybrid_d2v.model"
#output_model_path = "data/embeddings/TREC/TREC_hybrid_d2v.model"

#output_model_path = "../data_full/RELISH/RELISH_hybrid_d2v.model"
#output_model_path = "../data_full/TREC/TREC_hybrid_d2v.model"

cm.save_doc2vec_model(model, output_model_path)
```

## Step 5: Store the embeddings

The embeddings can be stored either in the model itself or as a separate entity outside of Doc2Vec (this allows to calculate cosine similarity without the need of Doc2Vec once the embeddings are already generated).

At the same time, the user can choose to store the embeddings into a single file (recommended) or into multiple files using the same `save_doc2vec_embedding()` function:


```python
output_path = "data/embeddings/RELISH/RELISH_document_embeddings.pkl"
#output_path = "data/embeddings/TREC/TREC_document_embeddings.pkl"

#output_path = "../data_full/RELISH/RELISH_document_embeddings.pkl"
#output_path = "../data_full/TREC/TREC_document_embeddings.pkl"

cm.save_doc2vec_embeddings(model, pmid, output_path, one_file=True)
```

# Decision notes

## Code strategy

1. The pipeline accepts either a `.tsv` or a `.npy` format as the input tokens. Usually, `.tsv` format is prefered since its size on disk is smaller. The tokens should have been generated following the [preprocessing tutorial](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/blob/main/docs/preprocessing/tutorial_preprocessing.ipynb).
    
    If using a custom `.tsv` file, three columns are required: "PMID", "title" and "abstract".
    

2. The input of Doc2Vec models should be a list of `TaggedDocument`. In this case, we join the title and the abstract as a single paragraph and set the PMID as the tag. 

3. To decide on the hyperparameters, we performed a literature review of common Doc2Vec hyperparameters and their different possibilities. Results can be consulted [here](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/resources). We opted to use a dictonary of the hyperparameters since this allows for an easy hyperparameter search implementation. Please, refer to the [tendency analysis](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/blob/main/docs/tendency_analysis/tutorial_tendency_analysis.ipynb) study in which the different hyperparamters are tested.

4. To create the model, we just need the input hyperparameters and the tagged data to build a vocabulary. The vocabulary building step is executed in here in order to better separate the model creating from its training, but it could be constructed at the training time without any problem.

5. To train the model, we use the number of epochs selected in the model parameters and the examples provided in the tagged data. Additionally, we provide a logging implementation to obtain information about the training:

    * Warning/Errors (verbose = 0): default information logged by `gensim` if any error occurs during the training.

    * Info (verbose = 1): provides information about the total training time (in seconds).

    * Debug (verbose = 2): shows the time at which every epoch starts and finishes.

    Lastly, once the model is trained, it can be stored in disk to load later.

6. The last step is to generate the embeddings for each publication. This allows to later calculate the cosine similarities for each pair of documents without the need of the `Doc2Vec` model. 

## Decisions

* The parameters are passed to the `generate_doc2vec_model()` function as a dictionary to later facilitate the inclusion of hyperparemeter optimization as well as providing an easy to use and implement feaure.

* In the training process, we employed the `logging` library to provide information about to the end user. The information reported is selected with the `verbose` parameter as explained in the section before.

* **MISSING EMBEDDINGS OUTPUT DECISIONS**

* **MISSING VOCABULARY CREATION DECISIONS (if needed)**

<! ---
The vocabulary is built when the model is created followed the tutorials in their documentation. The vocabulary construction can be executed automatically either at model initiailization and model training, but to provide a clearer pipeline, it is left manua
-->

## Notes

The time to train each dataset (TREC or RELISH) using 8 cores of an Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz with 16GB of RAM running Ubuntu 20.04 LTS and Python 3.8.10 is:

* RELISH (163189 publications): 25 seconds per epoch on average.

* TREC (32604 publications): 5 seconds per epoch on average.

These results will greatly depend on the chosen hyperparameters.

# TODO

* Include library dependencies in prerequisites.

* Finish the decisions.

* Finish how to fill the relevance matrix maybe.

**REMOVE THIS LINE BEFORE FINAL VERSION COMMIT**


```python
!jupyter nbconvert docs/embeddings/tutorial_embeddings.ipynb --to markdown --output README.md
```

    [NbConvertApp] Converting notebook docs/embeddings/tutorial_embeddings.ipynb to markdown
    [NbConvertApp] Writing 9129 bytes to docs/embeddings/README.md

