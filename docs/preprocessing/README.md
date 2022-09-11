# Hybrid approach phrase preprocessing

In this tutorial, we will cover the phrase preprocessing step for the hybrid-dictionary-ner approach. An alternative (and compatible) more general preprocessing can be found in the medline-preprocessing repository in [this tutorial](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/docs/Phrase_Preprocessing_Tutorial/tutorial_phrase_preprocessing.ipynb).

# Prerequisites

1. Remove structure words from the dataset by using the Structure_Words_removal module in [this tutorial](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Structure_Words_removal).

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
input_path = "data/RELISH/RELISH_documents_20220628_ann_swr.tsv"
#input_path = "data/TREC/TREC_documents_20220628_ann_swr.tsv"

data = pp.read_data(input_path)
data = pp.preprocess_data(data)
```

## Step 3: Save the preprocess data

The output is either stored in TSV or NPY format. The TSV is usually smaller in disk and faster to write.


```python
output_path = "data/RELISH/RELISH_tokens.tsv"
#output_path = "data/TREC/TREC_tokens.tsv"

pp.save_output(data, output_path, npy_format=False)
#pp.save_output(data, output_path, npy_format=True)
```

# Use `preprocess.py` as a script

The code file itself is prepared to work on their own given some parameters. In order to execute the script, run the following command:

```bash
preprocess.py [-h] -i INPUT [-o OUTPUT]
```

You must pass the following argument:

* -i / --input: path to the TSV file to be preprocessed.

Additionally, other parameters can be specified:

* -o / --output: path to the output TSV file after preprocessing.

* --npy_format: whether to save the output in NPY format. By default False

If no output is provided, the text `_preprocessed` will be added to the input file and saved in the same location. If the output path specified is a TSV (NPY) format, but the `npy_format` argument is set to True (False), the file extension will take priority and the output will be stored in TSV (NPY). Note that by using the script directly, you cannot use the abbreviation method.

An example of the command that will preprocess the example dataset in the data folder is:

```bash
python code/preprocessing/preprocess.py --input data/RELISH/RELISH_documents_20220628_ann_swr.tsv --output data/RELISH/RELISH_tokens.tsv

python code/preprocessing/preprocess.py --input data/TREC/TREC_documents_20220628_ann_swr.tsv --output data/TREC/TREC_tokens.tsv
```

# Results

Example of preprocessing:

<table>
<tr>
<th>Abstract Input</th>
<th>Abstract Output</th>
</tr>
<tr>
<td width="50%">
Despite the high MeSHQ000453 of MeSHD010300, the MeSHQ000503 of its gastrointestinal MeSHQ000175 remains poorly understood. to evaluate MeSHD003679 and defecatory MeSHQ000502 in MeSHD010361 with MeSHD010300 and age- and MeSHD012723-matched MeSHQ000517 and to correlate objective MeSHQ000175 with subjective MeSHQ000175.
</td>
<td width="50%">
despite the high meshq000453 of meshd010300 the meshq000503 of its gastrointestinal meshq000175 remains poorly understood to evaluate meshd003679 and defecatory meshq000502 in meshd010361 with meshd010300 and age- and meshd012723-matched meshq000517 and to correlate objective meshq000175 with subjective meshq000175
</td>
</tr>
</table>

# Decision notes

## Code strategy

1. The input file must be in TSV format, containing three columns: "PMID", "title", "abstract". The file is recommended to be pruned of structure words following [this tutorial](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Structure_Words_removal).

2. The code loops through every row of the input TSV and preprocess the title and abstract separately. 

3. The preprocess consists in:
    
    * (Optional) If the function parameter `process_abv` is set to `True` (`False` by default), an experimental abbreviation algorithm is executed to find and combine terms like "E. coli" or "S. aureus" into a single word like "e.coli". This process is not well tested and not recommended unless necessary.

    * Lowercase everything. 

    * Tokenize space-separated words. The text is split by white spaces and only alphanumeric characters and allowed punctuation is kept.

    * Removes all special character except for the hyphens `-` (this can be modified using the function parameter `allowed_punctuation`).

    * (Optional) Saves the results as a three-dimensional numpy array and saves it as a NPY file.

    * (Optional) Saves the results as a three column TSV file containing "PMID", "title" and "abstract".

## Decisions

* Instead of using the default phrase preprocessing found in medline-preprocessing ([here](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/docs/Phrase_Preprocessing_Tutorial/tutorial_phrase_preprocessing.ipynb)), the steps produced in here are particular for the hybrid-dictionary-ner approach. The results produced are expected to be the same, but execution time is greatly improved in this approach. The main difference is to not include the biological tokenizer `en_core_sci_lg` from the sciSpacy module, since its use is not recommended in this approach.

* For the decisions related to the actual preprocess steps followed, please refer to the main documentation in [here](https://github.com/zbmed-semtec/medline-preprocessing#cleaning-for-word-embedding).

## Notes

The time to preprocess each dataset (TREC or RELISH) using 1 core of an Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz with 16GB of RAM running Ubuntu 20.04 LTS and Python 3.8.10 is:

* RELISH (163189 publications): 2min 31s ± 538 ms on average.

* TREC (32604 publications): 26.9 s ± 203 ms on average.

<!-- 
```python
!jupyter nbconvert docs/preprocessing/tutorial_preprocessing.ipynb --to markdown --output README.md
```

    [NbConvertApp] Converting notebook docs/preprocessing/tutorial_preprocessing.ipynb to markdown
    [NbConvertApp] Writing 6597 bytes to docs/preprocessing/README.md
-->