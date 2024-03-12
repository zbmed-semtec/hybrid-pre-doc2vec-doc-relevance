import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from typing import Union, List

# Retrieves cleaned data from RELISH and TREC npy files


def process_data_from_npy(file_path_in: str = None) -> Union[List[str], List[List[str]], List[List[str]], List[List[str]]]:
    """
    Retrieves cleaned data from RELISH and TREC npy files, separating each column 
    into their own respective list.

    Parameters
    ----------
    filepathIn: str
            The filepath of the RELISH or TREC input npy file.
    Returns
    -------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    titles: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed title.
    abstracts: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed abstract.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    """
    doc = np.load(file_path_in, allow_pickle=True)

    pmids = []
    titles = []
    abstracts = []
    docs = []

    for line in doc:
        if isinstance(line[0], (np.ndarray, np.generic)):
            pmids.append(np.ndarray.tolist(line[0]))
            titles.append(np.ndarray.tolist(line[1]))
            abstracts.append(np.ndarray.tolist(line[2]))
            docs.append(np.ndarray.tolist(
                line[1]) + np.ndarray.tolist(line[2]))
        else:
            pmids.append(line[0])
            titles.append(line[1])
            abstracts.append(line[2])
            docs.append(line[1] + line[2])

    return (pmids, titles, abstracts, docs)

# Create and train the Doc2Vec Model


def createDoc2VecModel(pmids: List[str], docs: List[List[str]], params: dict) -> Doc2Vec:
    """
    Create and train the Doc2Vec model using Gensim for the documents 
    in the corpus.

    Parameters
    ----------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    params: dict
            Dictionary containing the parameters for the Doc2Vec model.
    Returns
    -------
    model: Doc2Vec
            Doc2Vec model.
    """
    tagged_data = [TaggedDocument(words=_d, tags=[str(pmids[i])])
                   for i, _d in enumerate(docs)]

    # model = Doc2Vec(vector_size=200, window=5, min_count=1, epochs=5)
    model = Doc2Vec(**params)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model

# Save the Doc2Vec Model


def saveDoc2VecModel(model: Doc2Vec, output_file: str) -> None:
    """
    Saves the Doc2Vec model.

    Parameters
    ----------
    model: Doc2Vec
            Doc2Vec model.
    output_file: str
            File path of the Doc2Vec model generated.
    """
    model.save(output_file)

# Generate and save the document embeddings


def create_document_embeddings(pmids: List[str], model: Doc2Vec, output_directory: str) -> None:
    """
    Create and save the document embeddings for the documents 
    in the corpus using the Doc2Vec model.

    Parameters
    ----------
    pmids: list of str
            A list of all pubmed ids in the corpus.
    model: Doc2Vec
            Doc2Vec model.
    output_directory: str
            The directory path where the document embeddings 
            will be stored.
    """
    for pmid in pmids:
        np.save(f'{output_directory}/{pmid}', model.docvecs[str(pmid)])