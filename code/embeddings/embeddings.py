import sys
import numpy as np
import logging
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from typing import Union, List

# Retrieves cleaned data from RELISH and TREC npy files


def process_data_from_npy(filepathIn: str = None):
        """
        Retrieves data from RELISH npy files, separating each column into their own respective list.

        Parameters
        ----------
        filepathIn: str
                The filepath of the RELISH or TREC input npy file.

        Returns
        -------
        list of str
                All pubmed ids associated to the paper.
        list of str
                All words within the title.
        list of str
                All words within the abstract.
        """
        if not isinstance(filepathIn, str):
                logging.alert("Wrong parameter type for prepareFromTSV.")
                sys.exit("filepathIn needs to be of type string")
        else:
                doc = np.load(filepathIn, allow_pickle=True)
                pmids = []
                titles = []
                abstracts = []
                docs = []
                for line in doc:
                    pmids.append(int(line[0]))
                    if isinstance(line[1], (np.ndarray, np.generic)):
                        titles.append(np.ndarray.tolist(line[1]))
                        abstracts.append(np.ndarray.tolist(line[2]))
                        docs.append(np.ndarray.tolist(
                            line[1]) + np.ndarray.tolist(line[2]))
                    else:
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