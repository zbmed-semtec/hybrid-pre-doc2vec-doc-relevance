import sys
import time

import logging
import argparse
import numpy as np
import pandas as pd

from typing import Tuple, List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):

    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0


    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def load_numpy_file(input_path: str) -> Tuple[List[int], List[List[str]]]:
    """
    Loads a .NPY file with the preprocessed articles. The files should be
    generated using the preprocess script.

    Parameters
    ----------
    input_path : str
        Input path for the .NPY file.

    Returns
    -------
    pmid: List[int]
        The pmid of the articles in a list.
    join_text: List[str]
        The combination of title + abstract of the articles in a list.
    """
    data = np.load(input_path, allow_pickle=True)

    pmid = [doc[0].item() for doc in data]
    #titles = [i[1].tolist() for i in a]
    #abstract = [i[2].tolist() for i in a]
    join_text = [doc[1].tolist() + doc[2].tolist() for doc in data]

    return pmid, join_text


def load_tsv_file(input_path: str) -> Tuple[List[int], List[List[str]]]:
    """
    Loads a .TSV file with the preprocessed articles. The files should be
    generated using the preprocess script, or any .TSV with three columns:
    PMID, title and abstract.

    Parameters
    ----------
    input_path : str
        Input path for the .TSV file.

    Returns
    -------
    pmid: List[int]
        The pmid of the articles in a list.
    join_text: List[str]
        The combination of title + abstract of the articles in a list.
    """
    data = pd.read_csv(input_path, delimiter="\t", quotechar="`")

    pmid = list(data["PMID"])
    #title = [title.split() for title in data["title"]]
    #abstract = [abstract.split() for abstract in data["abstract"]]
    join_text = [doc.split()
                 for doc in (data["title"] + " " + data["abstract"])]

    return pmid, join_text


def load_tokens(input_path: str) -> Tuple[List[int], List[List[str]]]:
    """
    Read from either a .TSV or a .NPY file the data.

    Parameters
    ----------
    input_path : str
        Input path for the file.

    Returns
    -------
    Tuple[List[int], List[List[str]]]
        The output from any of load_tsv_file() or load_numpy_file()
    """
    if input_path.endswith(".tsv"):
        return load_tsv_file(input_path)
    elif input_path.endswith(".npy"):
        return load_numpy_file(input_path)
    else:
        logging.error("Input must be a .TSV or .NPY file.")
        sys.exit("Unrecognize input file extension.")


def generate_TaggedDocument(pmid: List[int], join_text: List[List[str]]) -> TaggedDocument:
    """
    Generates the tagged documents required for Doc2Vec.

    Parameters
    ----------
    pmid : List[int]
        List of PMIDs extracted from the data.
    join_text : List[List[str]]
        List containing to tokens from each article. Title and abstract are
        combined in one text by default.

    Returns
    -------
    tagged_data: TaggedDocument
        TaggedDocument where every PMID is tagged into an element of the token
        list.
    """
    tagged_dict = dict(zip(pmid, join_text))
    tagged_data = [TaggedDocument(words=value, tags=[str(key)])
                   for key, value in tagged_dict.items()]

    return tagged_data


def generate_doc2vec_model(
    tagged_data: TaggedDocument, 
    params: dict = {
        "vector_size": 200, 
        "window": 5, 
        "min_count": 5, 
        "epochs": 5, 
        "workers": 4}
    ) -> Doc2Vec:
    """
    Generates de Doc2Vec model and builds the vocabulary from the
    taggedDocuments.

    Parameters
    ----------
    tagged_data : TaggedDocument
        TaggedDocument where every PMID is tagged into an element of the token
        list.
    params : dict, optional
        Dictionary containing the parameters to create the Doc2Vec model, by
        default { "vector_size": 200, "window": 5, "min_count": 5, "epochs": 5,
        "workers": 4}

    Returns
    -------
    model: Doc2Vec
        Doc2Vec model.
    """

    #model = Doc2Vec(vector_size=200, window=5, min_count=1, epochs=5, workers = 4)
    model = Doc2Vec(**params)
    model.build_vocab(tagged_data)

    return model


def train_doc2vec_model(model: Doc2Vec, tagged_data: TaggedDocument, time_train: bool = False) -> None:
    """
    Trains the model using the taggedDocuments. Since the model had the
    vocabulary built already, we only need to call for the train() function.

    Parameters
    ----------
    model : Doc2Vec
        Doc2Vec model.
    tagged_data : TaggedDocument
        TaggedDocument where every PMID is tagged into an element of the token
        list.
    """
    epoch_logger = EpochLogger()
    if time_train: 
        start_time = time.time()
    callbacks = [epoch_logger] if time_train else []

    model.train(tagged_data, total_examples=model.corpus_count,
                epochs=model.epochs, callbacks=callbacks)

    if time_train: print("--- Time to train: {:.2f} seconds".format(time.time() - start_time))

def save_doc2vec_model(model: Doc2Vec, output_path: str) -> None:
    """
    Saves the Doc2Vec model into the desired path.

    Parameters
    ----------
    model : Doc2Vec
        Doc2Vec model.
    output_path : str
        Path for output model.
    """
    model.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-i", "--input", type=str,
                        help="Path to input .NPY or .TSV file", required=True)
    parser.add_argument("-o", "--output", type=str,
                        help="Path to output Doc2Vec model")
    args = parser.parse_args()

    # Process output file
    if args.output:
        output_path = args.output
        if not output_path.endswith(".model"):
            output_path = output_path + ".model"
    else:
        output_path = "hybrid_d2v.model"

    pmid, join_text = load_tokens(args.input)
    tagged_data = generate_TaggedDocument(pmid, join_text)
    
    # Modify these model parameters.
    params_d2v = {
        "vector_size": 200, 
        "window": 5, 
        "min_count": 5, 
        "epochs": 5, 
        "workers": 4}

    model = generate_doc2vec_model(tagged_data, params_d2v)
    train_doc2vec_model(model, tagged_data, time_train = True)
    save_doc2vec_model(model, output_path)
    