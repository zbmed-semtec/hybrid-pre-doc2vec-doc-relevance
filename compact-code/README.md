## Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using one of the following commands:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/hybrid-pre-doc2vec-doc-relevance.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/hybrid-pre-doc2vec-doc-relevance.git
```

#### Navigate to the `compact-code` Folder:

```
cd hybrid-pre-doc2vec-doc-relevance-training/compact-code
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

### Step 3: Dataset
- Download the dataset from this link: [RELISH Annotated_Dataset]([https://drive.google.com/drive/folders/1Bq_U5207utn7tvSt_HLVdOdYR5QW7MMN](https://drive.google.com/drive/folders/1vsC6VrubXVIAEo5pxR0MvY7hDIDZ1tfc)).

### Step 4: Create Model Pipeline

This pipeline aims to create Doc-2Vec models using given hyperparameter sets, train the models, and evaluate their performance using three-class precision, two-class precision and nDCG score.

#### Pipeline Steps:

- **Model Training**: Trains Doc-2Vec model with the given hyperparameters using the annotated input data.
- **Storing Generation**: Stores the generated embeddings for the the annotated input data.
- **Cosine Similarity Computation**: Calculates cosine similarities for the generated embeddings.
- **Precision@N Calculation**: Computes Precision@N scores, a measure of the relevance of retrieved documents, for the obtained cosine similarities.
- **NDCG Score Calculation**: Computes normalized discounted cumulative gain (NDCG) scores, which assesses the quality of ranked search results based on relevance assessments.

In order to start the pipeline execution use this [script](./code/main.py), and run the following command:

```
python3 code/main.py [-i ANNOTATED_DATA] [-gt GROUND_TRUTH] [-pj PARAMS JSON]
```

You must pass the following four arguments:

+ -i/ --input : File path to input RELISH Annotated dataset (.npy file format).
+ -gt/ --ground_truth : File path for the Test split ground truth (.tsv file format).
+ -pj/ --params_json: File path to doc2vec hyperparameters JSON.

For instance, you may execute the following command:
```
python3 code/main.py --input data/RELISH_Annot_Tokens_Sample.npy --ground_truth data/RELISH_ground_truth.tsv --params_json data/hyperparameters_doc2vec.json
```

All outputs of the [script](./code/main.py) are saved in the folder named `output_of_model`. For hyperparameter set i, the code saves the corresponding trained model, embeddings, and similarities in the following file paths: `output_of_model/model_i/Doc2Vec_model`, `output_of_model/doc_embeddings/embeddings_pickle_i.pkl`, and `output_of_model/evaluation/cosine_similarity_i.tsv`, respectively. Additionally, evaluation results are stored in the folder `output_of_model/evaluation` using the same naming convention.

