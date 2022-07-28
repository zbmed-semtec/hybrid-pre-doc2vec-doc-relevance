import sys
import math

import argparse
import logging
import pandas as pd
import numpy as np

from gensim.models.doc2vec import Doc2Vec

"""
# TODO

1. Manage some edge cases (less rows than the default interval divission).
2. Better error handling.
"""

def load_relevance_matrix(input_path: str) -> pd.DataFrame:
    """
    Loads the relevance matrix into a pandas DataFrame.

    Parameters
    ----------
    input_path : str
        Input path for the .TSV relevance matrix file.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the relevance matrix either for RELISH or TREC.
    """
    return pd.read_csv(input_path, sep = "\t")

def load_d2v_model(d2v_model_path: str) -> Doc2Vec:
    """
    Lodas the doc2vec pretrained model.

    Parameters
    ----------
    d2v_model_path : str
        Input path for the .model file.

    Returns
    -------
    Doc2Vec
        Doc2Vec trained model.
    """
    return Doc2Vec.load(d2v_model_path)

def calculate_cosine_similarity(model: Doc2Vec, pmid_1: int, pmid_2: int) -> float:
    """
    Calculates the cosine similarity between two articles given their PMID. If
    any of the two articles is not found within the model corpus, an empty
    value is returned.

    Parameters
    ----------
    model : Doc2Vec
        Doc2Vec trained model.
    pmid_1 : int
        PMID of the first article.
    pmid_2 : int
        PMID of the second article.

    Returns
    -------
    float
        Cosine similarity between the two articles rounded to two decimal
        places. If one of the PMIDs is missing, an empty string is returned.
    """
    try:
        return round(model.dv.similarity(str(pmid_1), str(pmid_2)), 2)
    except:
        return ""

def fill_relevance_matrix(rel_matrix: pd.DataFrame, model: Doc2Vec, verbose: int = 0) -> pd.DataFrame:
    """
    Fills the relevance matrix with an aditional column named "Cosine
    Similarity". Because of different naming convections, the dataset from
    which the relevance matrix is constructed needs to be indicated.

    UNDER DEVELOPMENT.

    Parameters
    ----------
    rel_matrix: pd.DataFrame
        DataFrame containing the PMIDs to be compared.
    model: Doc2Vec
        Doc2Vec trained model.
    verbose: int, optional
        Whether to log the process of filling the relevance matrix, by default
        0.

    Returns
    -------
    rel_matrix: pd.DataFrame
        DataFrame containing the PMIDs to be compared with an additional column
        containing the Cosine Similarity.
    """
    pmid_1_name = "PMID1"
    pmid_2_name = "PMID2"
    rel_matrix["Cosine Similarity"] = ""

    if verbose:
        total_rows = len(rel_matrix)
        divisions = 20
        intervals = math.floor(total_rows/divisions)

    for i, row in rel_matrix.iterrows():
        pmid_1 = row[pmid_1_name]
        pmid_2 = row[pmid_2_name]

        cosine_similarity = calculate_cosine_similarity(model, pmid_1, pmid_2)

        rel_matrix.at[i, "Cosine Similarity"] = cosine_similarity

        if verbose and i%intervals == 0:
            print("Process at {}%".format(math.floor(i/intervals*100/divisions)))
    
    rel_matrix["Cosine Similarity"] = round(pd.to_numeric(rel_matrix["Cosine Similarity"]), 2)
    return rel_matrix

def fill_relevance_matrix_multiprocess(rel_matrix: pd.DataFrame, model: Doc2Vec, num_processes: int = None) -> pd.DataFrame:
    """
    Fills the relevance matrix with an aditional column named "Cosine
    Similarity". Because of different naming convections, the dataset from
    which the relevance matrix is constructed needs to be indicated. It uses
    multiple threads to accelerate the calculations.

    UNDER DEVELOPMENT.

    Parameters
    ----------
    rel_matrix: pd.DataFrame
        DataFrame containing the PMIDs to be compared.
    model: Doc2Vec
        Doc2Vec trained model.
    num_processes: int, optional
        Number of cores to use, by default the total number of cores of the
        system.

    Returns
    -------
    rel_matrix: pd.DataFrame
        DataFrame containing the PMIDs to be compared with an additional column
        containing the Cosine Similarity.
    """
    import multiprocessing as mp
    from itertools import repeat

    rel_matrix["Cosine Similarity"] = ""

    # We divide the relevance matrix into chunks. One for each core.
    if not num_processes:
        num_processes = mp.cpu_count()
    chunk_size = int(rel_matrix.shape[0]/num_processes)
    chunks = [rel_matrix.iloc[rel_matrix.index[i:i + chunk_size]] for i in range(0, rel_matrix.shape[0], chunk_size)]
    
    with mp.Pool(num_processes) as p:
        results = p.starmap(fill_relevance_matrix, zip(chunks, repeat(model)))

    for i in range(len(results)):
        rel_matrix.at[results[i].index] = results[i]
    
    return rel_matrix

def save_rel_matrix(rel_matrix: pd.DataFrame, output_path: str) -> None:
    """
    Saves the relevance matrix with the additional cosine similarity column.

    Parameters
    ----------
    rel_matrix : pd.DataFrame
        DataFrame containing the compared PMIDs and their Cosine Similarity.
    output_path : str
        Path for output relevance matrix.
    """
    rel_matrix.to_csv(output_path, index=False, sep="\t")

# Currently not working.
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     #group = parser.add_mutually_exclusive_group(required=True)
#     parser.add_argument("-i", "--input", type=str,
#                         help="Path to input relevance matrix .TSV file", required=True)
#     parser.add_argument("-m", "--model", type=str,
#                         help="Path to input model", required = True)
#     parser.add_argument("-d", "--dataset", type=str,
#                         help="Dataset from which to fill the relevance matrix")
#     parser.add_argument("-o", "--output", type=str,
#                         help="Path to output relevance matrix file")

#     args = parser.parse_args()

#     # Process dataset
#     if not args.dataset:
#         if "relish" in args.input.lower():
#             dataset = "RELISH"
#         if "trec" in args.input.lower():
#             dataset = "TREC"
#         else:
#             logging.error("Dataset used was not provided nor could be infered. Please specifiy either TREC or RELISH")
#             sys.exit("Dataset was not provided")
#     else:
#         dataset = args.dataset

#     # Process output file:
#     if not args.output:
#         output_path = args.input.replace(".tsv", "_filled.tsv")
#     else:
#         output_path = args.output


#     rel_matrix = load_relevance_matrix(args.input)
#     model = load_d2v_model(args.model)
    
#     fill_relevance_matrix(rel_matrix, model, dataset, verbose = 1)
#     save_rel_matrix(rel_matrix, output_path)
