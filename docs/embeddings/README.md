# Doc2Vec Model and embeddings generation

In this tutorial, we will cover the generation of the Doc2Vec model for the hybrid-dictionary-ner approach. The aim is to produce embeddings for each RELISH and TREC publication.

# Prerequisites

1. Preprocessed tokens in NPY format or TSV format. They can be generated following the [preprocessing tutorial](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/tree/main/docs/preprocessing).

# Model generation

## Steps

### Step 1: Imports

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

### Step 2: Loading the data

Next, we need to import the preprocessed tokens. A small sample is provided in the data folder. The `load_tokens()` function returns the title and abstract combined in one document:


```python
tokens_path = "data/RELISH/RELISH_tokens.tsv"
#tokens_path = "data/TREC/TREC_tokens.tsv"

pmid, join_text = cm.load_tokens(tokens_path)
```

We need to create the `TaggedDocuments` required by `Doc2Vec` to generate and train the models:


```python
tagged_data = cm.generate_TaggedDocument(pmid, join_text)
```

### Step 3: Creating the model

First, we need to choose the hyperparameters of the model. In this tutorial, we will only consider one combination of hyperparameters (for the hyperparameter optimization, please refer to the [tendency analysis](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/blob/main/docs/tendency_analysis/tutorial_tendency_analysis.ipynb) tutorial). The easiest way to indicate the hyperparameters is to create a dictionary with the [available options](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec):


```python
params_d2v = {
    "vector_size": 200, 
    "window": 5, 
    "min_count": 5, 
    "epochs": 5, 
    "workers": 4}
```

To create the model, use the `generate_doc2vec_model()` function with the tagged data and the model parameters. This function automatically creates the required vocabulary for training:


```python
model = cm.generate_doc2vec_model(tagged_data, params_d2v)
```

### Step 4: Training the model

The function `train_doc2vec_model` is responsible for training the previously generated model. The argument verbose determines the information to receive from the training process:


```python
cm.train_doc2vec_model(model, tagged_data, verbose=2)
```

    2022-09-11 10:24:02,639 	Epoch #0 start
    2022-09-11 10:24:02,669 	Epoch #0 end
    2022-09-11 10:24:02,670 	Epoch #1 start
    2022-09-11 10:24:02,699 	Epoch #1 end
    2022-09-11 10:24:02,699 	Epoch #2 start
    2022-09-11 10:24:02,726 	Epoch #2 end
    2022-09-11 10:24:02,727 	Epoch #3 start
    2022-09-11 10:24:02,755 	Epoch #3 end
    2022-09-11 10:24:02,756 	Epoch #4 start
    2022-09-11 10:24:02,780 	Epoch #4 end
    2022-09-11 10:24:02,781 --- Time to train: 0.14 seconds


The model can be stored to later be used by `save_doc2vec_model()` function:


```python
output_model_path = "data/RELISH/RELISH_hybrid.model"
#output_model_path = "data/TREC/TREC_hybrid.model"

cm.save_doc2vec_model(model, output_model_path)
```

### Step 5: Store the embeddings

The embeddings can be stored either in the model itself or as a separate entity outside Doc2Vec (this allows to calculate cosine similarity without the need of Doc2Vec once the embeddings are already generated).

The embeddings are stored into a dataframe with two columns: pmids and embeddings. The output is stored in pickle format.


```python
output_path = "data/RELISH/RELISH_embeddings.pkl"
#output_path = "data/TREC/TREC_embeddings.pkl"

cm.save_doc2vec_embeddings(model, pmid, output_path)
```

## Use `create_model.py` as a script

The code file itself is prepared to work on their own given some parameters. In order to execute the script, run the following command:

```bash
create_model.py [-h] -i INPUT [-o OUTPUT] [--embeddings EMBEDDINGS]
```

You must pass the following argument:

* -i / --input: path to the TSV file with the tokens.

Additionally, other parameters can be specified:

* -o / --output: path to the output model.

* --embeddings: path to the output embeddings.

An example of the command that will generate the model from the tokens in the data folder is:

```bash
python code/embeddings/create_model.py --input data/RELISH/RELISH_tokens.tsv --embeddings data/RELISH/RELISH_embeddings.pkl --output data/RELISH/RELISH_hybrid.model

python code/embeddings/create_model.py --input data/TREC/TREC_tokens.tsv --embeddings data/TREC/TREC_embeddings.pkl --output data/TREC/TREC_hybrid.model
```

## Results

An example of the output PKL file:


```python
import pandas as pd

pd.read_pickle("data/RELISH/RELISH_embeddings.pkl")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pmids</th>
      <th>embeddings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17977838</td>
      <td>[0.03675682470202446, -0.06266345828771591, -0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17997202</td>
      <td>[0.04471859708428383, -0.08651144802570343, -0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18207447</td>
      <td>[0.0328378789126873, -0.04986971616744995, -0....</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18235058</td>
      <td>[0.005275525618344545, -0.0077206785790622234,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18251855</td>
      <td>[0.0480794794857502, -0.07831325381994247, -0....</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>29394417</td>
      <td>[0.035523608326911926, -0.06166147068142891, -...</td>
    </tr>
    <tr>
      <th>146</th>
      <td>29483077</td>
      <td>[0.04499121755361557, -0.08086196333169937, -0...</td>
    </tr>
    <tr>
      <th>147</th>
      <td>29655810</td>
      <td>[0.0457320399582386, -0.08665464073419571, -0....</td>
    </tr>
    <tr>
      <th>148</th>
      <td>29721798</td>
      <td>[0.06336772441864014, -0.09875210374593735, -0...</td>
    </tr>
    <tr>
      <th>149</th>
      <td>29797481</td>
      <td>[0.040957752615213394, -0.06522607058286667, -...</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 2 columns</p>
</div>



# Filling of the relevance matrix

Additionally, this tutorial will also explain how to fill a relevance matrix with the Cosine Similarities calculated using the Doc2vec model. It is recommended to use the [more general approach](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity) to calculate the similarities directly from the embeddings and not from the model. 

## Steps

### Step 1: Load the libraries

First, we load the required library:


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
import fill_relevance_matrix as frm

logging.basicConfig(format='%(asctime)s %(message)s')
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


### Step 2: Load relevance matrix and model

Set the directory path of the relevance matrix and of the model. If you are running this code with the above tutorial, there is no need to load the model again:


```python
input_rm = "data/RELISH/RELISH_relevance_matrix.tsv"
input_model = "data/RELISH/RELISH_hybrid.model"

#input_rm = "data/TREC/TREC_relevance_matrix.tsv"
#input_model = "data/TREC/TREC_hybrid.model"

relevance_matrix = frm.load_relevance_matrix(input_rm)
model = frm.load_d2v_model(input_model)
```

### Step 3: Fill the relevance matrix and save it

In this step, it is recommended to use the multiprocessing pipeline:


```python
#filled_relevance_matrix = frm.fill_relevance_matrix(relevance_matrix, model, verbose=1)
filled_relevance_matrix = frm.fill_relevance_matrix_multiprocess(relevance_matrix, model, verbose=1)

```

    2022-09-11 10:24:04,160 --- Time to fill: 0.30 seconds


You can specify the number of cores with the `num_processess` parameter. By default, it is set to the number of cores of the system.

### Step 4: Save the filled relevance matrix:


```python
output_path = "data/RELISH/RELISH_filled_relevance_matrix.tsv"
#output_path = "data/TREC/TREC_filled_relevance_matrix.tsv"

frm.save_rel_matrix(filled_relevance_matrix, output_path)
```

## Use `fill_relevance_matrix.py` as a script

The code file itself is prepared to work on their own given some parameters. In order to execute the script, run the following command:

```bash
fill_relevance_matrix.py [-h] --input_rm INPUT_RM --input_model INPUT_MODEL --output OUTPUT [--verbose VERBOSE] [--multithread {0,1}] [--num_cores NUM_CORES]
```

You must pass the following argument:

* --input_rm: path to the TSV file with the tokens.

* --input_model: input path to the Doc2Vec model.

* --output: output path to the filled relevance matrix.

Additionally, other parameters can be specified:

* --verbose: the level of information logged in the process.

* --multithread: whether to use multiprocessing or not. It is recommended to set to 1. Optionally, set the number of cores to be used with 'num_cores' argument.

* --num_cores: number of cores to use if multiprocessing is available. By default, leave to 'None' to use all cores.

An example of the command that will fill the relevance matrix from the data folder is:

```bash
python code/embeddings/fill_relevance_matrix.py --input_rm data/RELISH/RELISH_relevance_matrix.tsv --input_model data/RELISH/RELISH_hybrid.model --output data/RELISH/RELISH_filled_relevance_matrix.tsv --verbose 1

python code/embeddings/fill_relevance_matrix.py --input_rm data/TREC/TREC_relevance_matrix.tsv --input_model data/TREC/TREC_hybrid.model --output data/TREC/TREC_filled_relevance_matrix.tsv --verbose 1
```

# Decision notes

## Code strategy

1. The pipeline accepts either a TSV or a NPY format as the input tokens. Usually, TSV format is preferred since its size on disk is smaller. The tokens should have been generated following the [preprocessing tutorial](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/blob/main/docs/preprocessing/tutorial_preprocessing.ipynb).
    
    If using a custom TSV file, three columns are required: "PMID", "title" and "abstract".
    

2. The input of Doc2Vec models should be a list of `TaggedDocument`. In this case, we join the title and the abstract as a single paragraph and set the PMID as the tag. 

3. To decide on the hyperparameters, we performed a literature review of common Doc2Vec hyperparameters and their different possibilities. Results can be consulted [here](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/resources). We opted to use a dictionary of the hyperparameters, since this allows for an easy hyperparameter search implementation. Please, refer to the [distribution analysis](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Distribution_Analysis/hp_optimization) study in which the different hyperparameters are tested.

4. To create the model, we just need the input hyperparameters and the tagged data to build a vocabulary. The vocabulary building step is executed in here in order to better separate the model creating from its training, but it could be constructed at the training time without any problem.

5. To train the model, we use the number of epochs selected in the model parameters and the examples provided in the tagged data. Additionally, we provide a logging implementation to obtain information about the training:

    * Warning/Errors (verbose = 0): default information logged by `gensim` if any error occurs during the training.

    * Info (verbose = 1): provides information about the total training time (in seconds).

    * Debug (verbose = 2): shows the time at which every epoch starts and finishes.

    Lastly, once the model is trained, it can be stored in disk to load later.

6. The next step is to generate the embeddings for each publication. This allows to later calculate the cosine similarities for each pair of documents without the need of the `Doc2Vec` model. 

7. If we wanted to use the model to fill a relevance matrix, the additional file `fill_relevance_matrix.py` reads the given relevance matrix, applies a verification process to check if the provided file matches the requirements and then compares every PMID in the two columns while filling a fourth column containing the cosine similarity.

## Decisions

* The parameters are passed to the `generate_doc2vec_model()` function as a dictionary to later facilitate the inclusion of hyperparameter optimization, as well as providing an easy to use and implement feature.

* In the training process, we employed the `logging` library to provide information about training to the end user. The information reported is selected with the `verbose` parameter as explained in the section before.

* The embeddings are stored into a `pandas` `DataFrame` with two columns: pmids and embeddings. The embeddings' column stores lists of the embeddings for that publication. We decided to store the dataframe in pickle format for its advantages ([resources](https://github.com/zbmed-semtec/bert-embeddings-doc-relevance/blob/main/playground/speed_size_comparision.ipynb)).

* We decided to provide an additional method of filling the relevance matrix (other than [the more general approach](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity)) in case the user prefers to use the model directly and not the intermediary pickle format embeddings.

## Notes

The time to train each dataset (TREC or RELISH) using 8 cores of an Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz with 16GB of RAM running Ubuntu 20.04 LTS and Python 3.8.10 is:

* RELISH (163189 publications): 25 seconds per epoch on average.

* TREC (32604 publications): 5 seconds per epoch on average.

The time to fill each dataset using 16 cores of the same processor with the same configuration is:

* RELISH (196680 comparisons): 35 seconds on average.

* TREC (18466607 comparisons): 2m 15 seconds on average.

These results will greatly depend on the chosen hyperparameters.

<!--
```python
!jupyter nbconvert docs/embeddings/tutorial_embeddings.ipynb --to markdown --output README.md
```

    [NbConvertApp] Converting notebook docs/embeddings/tutorial_embeddings.ipynb to markdown
    [NbConvertApp] Writing 15399 bytes to docs/embeddings/README.md
-->
