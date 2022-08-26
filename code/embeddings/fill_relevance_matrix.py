import sys
import math
import time
import logging

import pandas as pd

from gensim.models.doc2vec import Doc2Vec

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

def verify_matrix_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Verifies if the input relevance matrix has the required column names for
    the program to work. The required fields are:

    * PMID 1: specified with "PMID1" or "PMID Reference".
    * PMID 2: specified with "PMID2" or "PMID Assessed".

    Edit the variables valid_pmid1 and valid_pmid2 to allow for other names,
    but they will be renamed to "PMID1" and "PMID2".

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with the PMIDs to calculate the cosine similarity.
    """
    valid_pmid1 = ["PMID1", "PMID Reference"]
    valid_pmid2 = ["PMID2", "PMID Assessed"]

    if not any(valid_str in data.columns for valid_str in valid_pmid1):
        logger.error('No valid PMID1 column found in the specified file.')
        sys.exit("Invalid relevance matrix")

    if not any(valid_str in data.columns for valid_str in valid_pmid2):
        logger.error('No valid PMID2 column found in the specified file.')
        sys.exit("Invalid relevance matrix")
        

def load_relevance_matrix(input_path: str) -> pd.DataFrame:
    """
    Reads a .TSV file containing the relevance matrix. At least two columns are
    required with the PMIDs to compare.

    If the relevance matrix is used to optimize the model parameters or to find
    the cut points using the distribution analysis, a third column containing
    the relevance (RELISH) or group (TREC) is required.

    The function "verify_matrix_columns" controls if the input .TSV file
    contains the valid columns. This function also renames the valid columns
    "PMID1" and "PMID2".

    Parameters
    ----------
    input_path : str
        Input path for the .TSV relevance matrix file.

    Returns
    -------
    data: pd.DataFrame
        Dataframe with at least two columns: PMID1, PMID2. If used for
        optimization, a third column of relevance (for RELISH) or Group (for
        TREC) is required.
    """
    if not input_path.endswith(".tsv"):
        logger.warning("Input path is not a valid .TSV file. Please provide a .TSV file to avoid possible errors.")

    data = pd.read_csv(input_path, sep = "\t")
    verify_matrix_columns(data)
    data.rename(columns = {"PMID Reference": "PMID1", "PMID Assessed": "PMID2", "Relevance Assessment": "relevance"}, inplace = True)

    return data

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

def fill_relevance_matrix(rel_matrix: pd.DataFrame, model: Doc2Vec, verbose: int = 0, percentage_log: int = 5) -> pd.DataFrame:
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
        Determines the logging level of the training. 
        * 0: to not receive any information.
        * 1: to receive the total filling time.
        * 2: to receive the total filling time and notification percentage.
    percentage_log: int, optional
        If verbose is set to 2, the percentage in which to notify.

    Returns
    -------
    rel_matrix: pd.DataFrame
        DataFrame containing the PMIDs to be compared with an additional column
        containing the Cosine Similarity.
    """
    pmid_1_name = "PMID1"
    pmid_2_name = "PMID2"
    rel_matrix["Cosine Similarity"] = ""

    if verbose == 0:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose == 2:
        logger.setLevel(logging.DEBUG)

    total_rows = len(rel_matrix)
    divisions = 100/percentage_log
    intervals = math.floor(total_rows/divisions)
    if intervals == 0: intervals = 1
        #logger.warning("There are not enough comparisons to properly log the filling process.")

    start_time = time.time()
    for i, row in rel_matrix.iterrows():
        pmid_1 = row[pmid_1_name]
        pmid_2 = row[pmid_2_name]

        cosine_similarity = calculate_cosine_similarity(model, pmid_1, pmid_2)
        rel_matrix.at[i, "Cosine Similarity"] = cosine_similarity

        if verbose == 2 and i%intervals == 0:
            logger.info("Process at {}%".format(math.floor(i/intervals*100/divisions)))
    
    rel_matrix["Cosine Similarity"] = round(pd.to_numeric(rel_matrix["Cosine Similarity"]), 2)
    logger.info("--- Time to fill: {:.2f} seconds".format(time.time() - start_time))
    logger.setLevel(logging.WARNING)
    
    return rel_matrix

def fill_relevance_matrix_multiprocess(rel_matrix: pd.DataFrame, model: Doc2Vec, num_processes: int = None, verbose: int = 0) -> pd.DataFrame:
    """
    Fills the relevance matrix with an aditional column named "Cosine
    Similarity". Because of different naming convections, the dataset from
    which the relevance matrix is constructed needs to be indicated. It uses
    multiple threads to accelerate the calculations.

    Parameters
    ----------
    rel_matrix: pd.DataFrame
        DataFrame containing the PMIDs to be compared.
    model: Doc2Vec
        Doc2Vec trained model.
    num_processes: int, optional
        Number of cores to use, by default the total number of cores of the
        system.
    verbose: int, optional
        Determines the logging level of the training. 
        * 0: to not receive any information.
        * 1: to receive the total filling time.


    Returns
    -------
    rel_matrix: pd.DataFrame
        DataFrame containing the PMIDs to be compared with an additional column
        containing the Cosine Similarity.
    """
    import multiprocessing as mp
    from itertools import repeat

    if verbose == 0:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)

    rel_matrix["Cosine Similarity"] = ""

    # We divide the relevance matrix into chunks. One for each core.
    if not num_processes:
        num_processes = mp.cpu_count()
    chunk_size = int(rel_matrix.shape[0]/num_processes)
    chunks = [rel_matrix.iloc[rel_matrix.index[i:i + chunk_size]] for i in range(0, rel_matrix.shape[0], chunk_size)]
    
    start_time = time.time()
    with mp.Pool(num_processes) as p:
        results = p.starmap(fill_relevance_matrix, zip(chunks, repeat(model)))

    for i in range(len(results)):
        rel_matrix.at[results[i].index] = results[i]
    
    rel_matrix["Cosine Similarity"] = round(pd.to_numeric(rel_matrix["Cosine Similarity"]), 2)
    
    logger.info("--- Time to fill: {:.2f} seconds".format(time.time() - start_time))
    logger.setLevel(logging.WARNING)

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

# data = load_relevance_matrix("../../../data_full/RELISH/RELISH_relevance_matrix.tsv")
# model = load_d2v_model("../../../data_full/RELISH/RELISH_hybrid_d2v.model")
# fill_relevance_matrix_multiprocess(data, model, verbose = 1)
# save_rel_matrix(data, "../../../data_full/RELISH/RELISH_filled_relevance_matrix.tsv")

# data = load_relevance_matrix("../../../data_full/TREC/TREC_simplified_relevance_matrix.tsv")
# model = load_d2v_model("../../../data_full/TREC/TREC_hybrid_d2v.model")
# fill_relevance_matrix_multiprocess(data, model, verbose = 1)
# save_rel_matrix(data, "../../../data_full/TREC/TREC_filled_simplified_relevance_matrix.tsv")

# data = load_relevance_matrix("../../../data_full/TREC/TREC_repurposed_matrix.tsv")
# model = load_d2v_model("../../../data_full/TREC/TREC_hybrid_d2v.model")
# fill_relevance_matrix_multiprocess(data, model, verbose = 1)
# save_rel_matrix(data, "../../../data_full/TREC/TREC_filled_repurposed_matrix.tsv")