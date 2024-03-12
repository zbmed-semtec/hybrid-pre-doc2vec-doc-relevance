import pandas as pd
import numpy as np

def sort_pmids(pmids):
    """
    Sorts the pmids in the TREC or RELISH dataset into ascending
    order.

    Parameters
    ----------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    Returns
    -------
    pmids: List[str]
            A list of all pubmed ids in the corpus sorted 
            in the ascending order.
    """
    # Convert all values in the list "pmids" from type 'str' to 'int'
    pmids = list(map(int, pmids))
    # Sort pmids in ascending order
    pmids.sort()

    return pmids

def get_embeddings(embeddings_path, pmids):
    """
    Get a list of embeddings against the sorted pubmed ids.

    Parameters
    ----------
    embeddings_path: str
            Path to the directory where the embeddings are 
            saved. 
    pmids: List[str]
            A list of all pubmed ids in the corpus sorted 
            in the ascending order.
    Returns
    -------
    embeddings_list: List[ndarray]
            A list of embeddings (against the sorted 
            pmids).
    """
    embeddings_list = []
    for pmid in pmids:
        # Load the embeddings
        file_name = embeddings_path + str(pmid) + ".npy"
        embeddings = np.load(file_name)
        # Append the embeddings to the "embeddings_list"
        embeddings_list.append(embeddings)

    return embeddings_list

def save_dataframe(pmids, embeddings_list, output_file):
    """
    Creates a 2-column Pandas dataframe with the pmids
    and their embeddings. Saves the dataframe in a 
    Pickle Python format.

    Parameters
    ----------
    pmids: List[str]
            A list of all pubmed ids in the corpus sorted 
            in the ascending order.
    embeddings_list: List[ndarray]
            A list of embeddings (against the sorted 
            pmids).
    output_file: File path to save the Pandas dataframe. 
    """
    # Create pandas dataframe with pmids and embeddings
    dict = {'pmids': pmids, 'embeddings': embeddings_list} 
    df = pd.DataFrame(dict)

    # Save dataframe to Python pickle format
    df.to_pickle(output_file)

def load_dataframe(file_path):
    """
    Loads a 2-column Pandas dataframe with the pmids
    and their embeddings, from a Pickle file.

    Parameters
    ----------
    file_path: Path where the Pickle file is saved. 
    """
    # Load dataframe
    df = pd.read_pickle(file_path)
    print (df)
