# Hybrid-dictionary-ner-doc2vec-Doc-relevance

In this repository, an approach exploring the combination of a dictionary-based NER using Whatizit and Doc2Vec framework, was developed to generate a document-to-document recommendation system. This approach will explore the transformation of annotated XML files after the Whatizit processing into a plain text dataset, the required preprocessing of the text and the Doc2Vec model generation, training and evaluation.

## Table of Contents

1. [About](#about)
2. [Input Data](#input-data)
3. [Pipeline](#pipeline)
    1. [XML Translation](#xml-translation)
    2. [Data Preprocessing](#data-preprocessing)
        - [Structure words removal](#structure-words-removal)
        - [Text cleaning and tokenization](#text-cleaning-and-tokenization)
    3. [Generate Embeddings](#generate-embeddings)
    4. [Calculate Cosine Similarity](#calculate-cosine-similarity)
    5. [Hyperparameter Optimization](#)
    6. [Evaluation](#evaluation)
        - [Precision@N](#precisionn)
        - [nDCG@N](#ndcgn)
4. [Getting Started](#getting-started)
5. [Tutorial](#tutorial)


## About

The main idea is to use a dictionary-based name entity recognition to group different medical terms into a single entity. To do so, we use the Whatizit text processing system developed by the Rebholz Research Group at [EMBL-EBI](https://www.ebi.ac.uk/). Once the terms are annotated, we replace these  
annotations by their MeSH ID. For example, the dictionary will recognize every entry of the term "mechanism of action" all will add an XML tag around it. Our pipeline will then replace the whole tag by its MeSH ID, specifically for this term: [MeSHQ000494](http://purl.bioontology.org/ontology/MESH/Q000494).

Once we have the title and abstract of a publication with their medical terms substituted by their MeSH ID, we apply a standard Doc2Vec process to generate embeddings. Then, using the cosine similarity between two different publications, we will rank their similarity from 0 to 1.

## Input data

On what is RELISH, please refer to our [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing), where the data explanation and extraction is described. 

For the hybrid approach, the input data are annotated XML files generated using Whatizit/monq. The process can be found in the [repository documentation](https://github.com/zbmed-semtec/whatizit-dictionary-ner/tree/main/docs).

This is an example of the annotated XML files we need:

```XML
<?xml version='1.0' encoding='utf-8'?>
<collection xmlns:z="https://github.com/zbmed-semtec/whatizit-dictionary-ner#">
    <document>
        <id>
            10606228
        </id>
        <passage>
            <infon key="type">
                title
            </infon>
            <offset>
                0
            </offset>
            <text>
                Spontaneous and <z:mesh cui="C0008904, C0026879" id="http://purl.bioontology.org/ontology/MESH/D009153" semantics="http://purl.bioontology.org/ontology/STY/T131">mutagen</z:mesh>-induced transformation of primary <z:mesh cui="C0243103, C0015033, C0220814" id="http://purl.bioontology.org/ontology/MESH/Q000208" semantics="http://purl.bioontology.org/ontology/STY/T080, http://purl.bioontology.org/ontology/STY/T169">cultures</z:mesh> of Msh2-/- p53-/- colonocytes.
            </text>
        </passage>
        <passage>
            <infon key="type">
                abstract
            </infon>
            <offset>
                98
            </offset>
            <text>
                Publication abstract removed for simplicity.
            </text>
        </passage>
    </document>
</collection>
```

## Pipeline

Here, we describe the main process over the RELISH dataset.

## XML translation

[Main documentation](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/tree/main/docs/xml_translate).

The goal is to convert the annotations in the XML files into a MeSH ID. As mentioned before, the input data are annotated XML files.

1. Loop throught every XML file inside the given folder. Optionally, a single file can also be provided. No requisites for the XML file will be specified, since they are expected to be generated using the Whatizit NER approach.

2. Create a `translation_dictionary`, where every term is associated to a MeSH ID.

3. Replace occurrences of a term inside the `translation_dictionary` by their MeSH ID.

4. Store all the translated XML files into a single TSV file with three columns: PMID, title, abstract. Optionally, the files can be individually saved into a TXT.

At the end of this process, we expect to have a TSV file for each dataset (RELISH and TREC) with the already mentioned three columns and the medical terms identified by the Whatizit dictionary replaced by their MeSH ID.

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PMID</th>
      <th>title</th>
      <th>abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18394048</td>
      <td>Effect of MeSHD000077287 on MeSHD000071080: a ...</td>
      <td>BACKGROUND AND PURPOSE: We studied the effect ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18363035</td>
      <td>A MeSHD016678-wide MeSHD046228 reveals anti-in...</td>
      <td>OBJECTIVE: Paeony root has long been used for ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18366698</td>
      <td>The MeSHQ000706 of biomedicine, complementary ...</td>
      <td>BACKGROUND: Studies have shown that a signific...</td>
    </tr>
  </tbody>
</table>
</div>

## Data preprocessing

The next step is to apply some preprocessing to the publications. This process is split in two different steps:

### Structure words removal

[Main documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Structure_Words_removal).

Both the code and the documentation is located in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing) since it is a common step in every document-to-document similarity approach. We defined as "Structure words" those terms that are introduced in the text to better structure the abstract. In the table shown above, we can see that, for example, the words "BACKGROUND: " and "OBJECTIVE: " are written at the start of the abstract. These terms do not provided any meaningful information and are not found in every publication and, since in the end we try to measure the similarity between documents, they can artificially increase the similarity of two publications that are not related otherwise.

These structure words usually follow a pattern: most of them are in capital letters (not always), they all end with a colon and an empty space ": " and they are always located at the beginning of a sentence. Using these rules, we developed an algorithm to identify and eliminate them.

The main idea is:

1. Loop through every publication abstract and match a regular expression to find the structure words. At the same time, count in how many publications each structure word appears.

2. Since we are applying a regular expression, some false positives might be found (certain acronyms for example). Since we don't want to remove relevant terms that may follow the regular expression, we apply a minimum frequency of appearance threshold. The standard is to require a matched structure word to appear in at least 0.01% of all publications. We create a [list of these structure words](https://github.com/zbmed-semtec/relish-preprocessing/blob/main/data/output/structure-words/structure_word_list_pruned.txt). It is possible to modify the minimum frequency of appearance by applying a different threshold. More information can be found in the corresponding documentation.

3. Remove every structure word found in the list.

Example of structure words removal:

<table>
<tr>
<th>Abstract Input</th>
<th>Abstract Output</th>
</tr>
<tr>
<td width="50%">
<mark>OBJECTIVE: </mark>To describe the development of evidence-based electronic prescribing (e-prescribing) triggers and treatment algorithms for potentially inappropriate medications (PIMs) for older adults. <mark>DESIGN: </mark>Literature review, expert panel and focus group. <mark>SETTING: </mark>Primary care with access to e-prescribing systems. <mark>PARTICIPANTS: </mark>Primary care physicians using e-prescribing systems receiving medication history. <mark>INTERVENTIONS: </mark>Standardised treatment algorithms for clinicians attempting to prescribe PIMs for older patients. <mark>MAIN OUTCOME MEASURE: </mark>Development of 15 treatment algorithms suggesting alternative therapies. <mark>RESULTS: </mark>Evidence-based treatment algorithms were well received by primary care physicians. Providing alternatives to PIMs would make it easier for physicians to change decisions at the point of prescribing. <mark>CONCLUSION: </mark>Prospectively identifying older persons receiving PIMs or with adherence issues and providing feasible interventions may prevent adverse drug events.
</td>
<td width="50%">
To describe the development of evidence-based electronic prescribing (e-prescribing) triggers and treatment algorithms for potentially inappropriate medications (PIMs) for older adults. Literature review, expert panel and focus group. Primary care with access to e-prescribing systems. Primary care physicians using e-prescribing systems receiving medication history. Standardised treatment algorithms for clinicians attempting to prescribe PIMs for older patients. Development of 15 treatment algorithms suggesting alternative therapies. Evidence-based treatment algorithms were well received by primary care physicians. Providing alternatives to PIMs would make it easier for physicians to change decisions at the point of prescribing. Prospectively identifying older persons receiving PIMs or with adherence issues and providing feasible interventions may prevent adverse drug events.
</td>
</tr>
</table>

### Text cleaning and tokenization

[Main documentation](https://github.com/zbmed-semtec/hybrid-pre-doc2vec-doc-relevance/tree/main/docs/preprocessing)

The cleaning process is also described in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing) ([main documentation](https://github.com/zbmed-semtec/hybrid-pre-doc2vec-doc-relevance/blob/main/docs/preprocessing/tutorial_preprocessing.ipynb)). However, a specific preprocessing was developed for the hybrid approach ([specific documentation](https://github.com/zbmed-semtec/hybrid-pre-doc2vec-doc-relevance/tree/main/docs/preprocessing)) since it does not have the same requirements as other approaches.

The process consists in:

1. Lower case everything and split by white spaces.

2. Remove every non-alphanumeric character with the exception of the hyphen `-`. The allowed characters can be modified following the documentation.

3. Store the results in a TSV file with the same three columns as before.

An example of the output is:

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

## Generate Embeddings

### Doc2Vec model generation

[Main documentation](/docs/embeddings)

With the text cleaned, we can now start to generate the embeddings. To do so, we will use the [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) framework provided by [Gensim](https://radimrehurek.com/gensim/). The process can be described as follows:

1. Generate the `TaggedDocuments` to match every PMID to a list of words. In this step, the abstract and the title are combined into a single paragraph (or document).

2. Generate the model with the desired hyperparameters (more information in the following section).

3. Train the model.

4. Extract the embeddings of the training publications or generate them for new publications.

## Calculate Cosine Similarity

To assess the similarity between two documents within the RELISH corpus, we employ the Cosine Similarity metric. This process enables the generation of a 4-column matrix containing cosine similarity scores for existing pairs of PMIDs within our corpus. For a more detailed explanation of the process, please refer to this documentation.

## Hyperparameters

One of the most important steps in every machine learning development is to properly choose a set of hyperparameters. The Doc2vec hyperparameters we will consider for this research are the following:

* The training algorithm **dm**. Refers to either using distributed memory or distributed bag of words.
* The **vector_size** of the generated embeddings. It represents the number of dimensions our embeddings will have.
* The **window** size represents the maximum distance between the current and predicted word.
* The number of **epochs** or iterations of the training dataset.
* The minimum number of appearances a word must have to not be ignored by the algorithm is specified with the **min_count** parameter.

The most relevant aspect when trying to optimize for hyperparameters is to choose an appropriate evaluation method. In this research, two different approaches were considered:

* ROC One vs All approach: if we understand our model as a non-relevant vs relevant classifier, we can use the area under the ROC curve (AUC) as a model quality estimator. More details can be found in the [corresponding documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Distribution_Analysis/hp_optimization).


## Evaluation

### Precision@N

In order to evaluate the effectiveness of this approach, we make use of Precision@N. Precision@N measures the precision of retrieved documents at various cutoff points (N).We generate a Precision@N matrix for existing pairs of documents within the RELISH corpus, based on the original RELISH JSON file. The code determines the number of true positives within the top N pairs and computes Precision@N scores. The result is a Precision@N matrix with values at different cutoff points, including average scores. For detailed insights into the algorithm, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Precision%40N_existing_pairs).

### nDCG@N

Another metric used is the nDCG@N (normalized Discounted Cumulative Gain). This ranking metric assesses document retrieval quality by considering both relevance and document ranking. It operates by using a TSV file containing relevance and cosine similarity scores, involving the computation of DCG@N and iDCG@N scores. The result is an nDCG@N matrix for various cutoff values (N) and each PMID in the corpus, with detailed information available in the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation).


## Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/hybrid-pre-doc2vec-doc-relevance.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/hybrid-pre-doc2vec-doc-relevance.git
```

### Step 2: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.6 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 3: Translate XMLs

Code implementation for this step can be found [here](https://github.com/zbmed-semtec/hybrid-pre-doc2vec-doc-relevance/tree/main/docs/xml_translate). 

### Step 4: Data Preprocessing

Code implementation for structure word removal can be found [here](https://github.com/zbmed-semtec/relish-preprocessing/tree/main/docs/structure_words_removal) and for text pre-processing can be found [here](https://github.com/zbmed-semtec/hybrid-pre-doc2vec-doc-relevance/tree/main/docs/preprocessing).

### Step 5: Generate Embeddings

Code implementation for generating embeddings using the Doc2vec models can be found [here](https://github.com/zbmed-semtec/hybrid-pre-doc2vec-doc-relevance/tree/main/docs/embeddings).


The `run_embeddings.py` script uses the RELISH Annotated and Preprocessed Tokenized npy file as input and includes a default parameter dictionary with preset hyperparameters. You can easily adapt it for different values and parameters by modifying the params_dict. Make sure to have the RELISH Tokenized.npy file within the directory under the data folder.

To run this script, please execute the following command:

```
python3 code/embeddings/run_embeddings.py --input "data/RELISH_tokenized.npy"
```

The script will create Doc2Vec models, generate embeddings, and store them in separate directories. You should expect to find a total of 18 files corresponding to the various models, embeddings, and embedding pickle files.


### Step 6: Calculate Cosine Similarity
In order to generate the cosine similarity matrix and execute this [script](/code/evaluation/generate_cosine_existing_pairs.py), run the following command:

```
python3 code/evaluation/generate_cosine_existing_pairs.py [-i INPUT PATH] [-e EMBEDDINGS] [-o OUTPUT PATH]
```

You must pass the following four arguments:

+ -i/ --input : File path to the RELISH relevance matrix in the TSV format.
+ -e/ --embeddings : File path to the embeddings in the pickle file format.
+ -o/ --output : File path for the output 4 column cosine similarity matrix.


For example, if you are running the code from the code folder and have the RELISH relevance matrix in the data folder, run the cosine matrix creation for all hyperparameters as:

```
python3 code/evaluation/generate_cosine_existing_pairs.py -i data/Relevance_matrix/relevance_w2v_blank.tsv -e dataframe/embeddings_pickle_0.tsv -o data/cosine_similarity_0.tsv
```

Note: You would have to run the above command for every hyperparameter configuration.

### Step 7: Precision@N
In order to calculate the Precision@N scores and execute this [script](/code/evaluation/precision.py), run the following command:

```
python3 code/evaluation/precision.py [-c COSINE FILE PATH]  [-o OUTPUT PATH]
```

You must pass the following two arguments:

+ -c/ --cosine_file_path: path to the 4-column cosine similarity existing pairs RELISH file: (tsv file)
+ -o/ --output_path: path to save the generated precision matrix: (tsv file)

For example, if you are running the code from the code folder and have the cosine similarity TSV file in the data folder, run the precision matrix creation for the first hyperparameter as:

```
python3 code/evaluation/precision.py -c data/cosine_similarity_0.tsv -o data/hybrid_precision_0.tsv
```



### Step 8: nDCG@N
In order to calculate nDCG scores and execute this [script](/code/evaluation/calculate_gain.py), run the following command:

```
python3 code/evaluation/calculate_gain.py [-i INPUT]  [-o OUTPUT] [-n NUMBER]
```

You must pass the following two arguments:

+ -i / --input: Path to the 4 column cosine similarity existing pairs RELISH TSV file.
+ -o/ --output: Output path along with the name of the file to save the generated nDCG@N TSV file.
+ -n/ --number: Number for the hyperparameter combination.

For example, if you are running the code from the code folder and have the 4 column RELISH TSV file in the data folder, run the matrix creation for the first hyperparameter as:

```
python3 code/evaluation/calculate_gain.py -i data/cosine_similarity_0.tsv -o data/hybrid_gain_0.tsv -n 0
```

### Step 9: Compile Results

In order to compile the average result values for Precison@ and nDCG@N and generate a single TSV file each, please use this [script](code/evaluation/show_avg.py).

You must pass the following two arguments:

+ -i / --input: Path to the directory consisting of all the precision matrices/gain matrices.
+ -o/ --output: Output path along with the name of the file to save the generated compiled Precision@N / nDCG@N TSV file.


If you are running the code from the code folder, run the compilation script as:

```
python3 code/evaluation/show_avg.py -i data/output/gain_matrices/ -o data/output/results_gain.tsv
```

NOTE: Please do not forget to put a `'/'` at the end of the input file path.

## Tutorials

Tutorials are accessible in the form of Jupyter notebooks for the following processes:

+ [Translating Annotated XMLs](https://github.com/zbmed-semtec/hybrid-doc2vec-doc-relevance-training/tree/main/docs/xml_translate)
+ [Data Preprocessing](https://github.com/zbmed-semtec/hybrid-doc2vec-doc-relevance-training/tree/main/docs/preprocessing) 
+ [Generating Embeddings](https://github.com/zbmed-semtec/hybrid-doc2vec-doc-relevance-training/tree/main/docs/embeddings)