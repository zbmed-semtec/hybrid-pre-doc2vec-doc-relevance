import tqdm
import gensim
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
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
    article_docs = []
    
    for line in range(len(doc)):
        pmids.append(int(doc[line][0]))
        
        # Check if the element is a NumPy array before using tolist
        if isinstance(doc[line][1], np.ndarray):
            article_docs.append(doc[line][1].tolist())
        else:
            article_docs.append(doc[line][1])
        
        # Check if the element is a NumPy array before using tolist
        if isinstance(doc[line][2], np.ndarray):
            article_docs[line].extend(doc[line][2].tolist())
        else:
            article_docs[line].extend(doc[line][2])
    return (pmids, article_docs)

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

def calculate_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def get_similarity_scores(input_relevance_matrix, embeddings_df):
    
    # Read Relevance matrix
    column_names = ["PID1", "PID2", "Value"]
    relevance_matrix_df = pd.read_csv(input_relevance_matrix, sep="\t", names = column_names, skiprows=1)

    # Create an empty list to store rows of PMID-pairs and their relevance- and similarity-scores
    rows_with_similarities = []
    
    # Create a list of ref and assessed PMID-pairs and their relevance-scores
    pmid_pairs_relevance = list(zip(relevance_matrix_df["PID1"], relevance_matrix_df["PID2"], relevance_matrix_df["Value"]))

    # Create a dictionary to store embeddings
    embeddings_dict = {pmid: embedding for pmid, embedding in zip(embeddings_df['PID'], embeddings_df['Embedding'])}


    for ref_pmid, assessed_pmid, rel_value in tqdm.tqdm(pmid_pairs_relevance, total=len(pmid_pairs_relevance),
                                                        desc="Calculating Cosine Similarities"):
        try:
            ref_pmid_vector = embeddings_dict[ref_pmid]
            assessed_pmid_vector = embeddings_dict[assessed_pmid]
            if ref_pmid_vector is not None and assessed_pmid_vector is not None:
                cosine_similarity = round(calculate_cosine_similarity(ref_pmid_vector, assessed_pmid_vector), 4)
                rows_with_similarities.append([ref_pmid, assessed_pmid, rel_value, cosine_similarity])
            else:
                continue
                #print(f"One of the vectors is None for ({ref_pmid}, {assessed_pmid})")
        except KeyError as e:
            #print(f"\nKeyError: {e}, ref_pmid: {ref_pmid}, assessed_pmid: {assessed_pmid}")
            continue

    # Create a DataFrame with columns "PID1", "PID2", "Value", "Cosine Similarity"
    similarity_df = pd.DataFrame(rows_with_similarities, columns=["PID1", "PID2", "Value", "Cosine Similarity"])
    
    return similarity_df      

def save_similarity_scores(input_relevance_df, output_matrix_name):
    # Saves the updated relavance+similarity matrix 
    input_relevance_df.to_csv(output_matrix_name, index=False, sep="\t")
    print('Saved relavance+similarity matrix')

def get_and_save_similarity_scores(input_relevance_matrix, embeddings, output_matrix_name):
    # Read Embeddings
    embeddings_df = pd.read_pickle(embeddings)
    
    # Read Relevance matrix
    column_names = ["PID1", "PID2", "Value"]
    relevance_matrix_df = pd.read_csv(input_relevance_matrix, sep="\t", names = column_names, skiprows=1)

    # Adds empty columns to the file to store similarity scores
    relevance_matrix_df["Cosine Similarity"] = ""

    #print(relevance_matrix_df)

    # Create a dictionary to store embeddings
    embeddings_dict = {pmid: embedding for pmid, embedding in zip(embeddings_df['PID'], embeddings_df['Embedding'])}

    # Create a list of ref and assessed PMID pairs
    pmid_pairs = list(zip(relevance_matrix_df["PID1"], relevance_matrix_df["PID2"]))

    for ref_pmid, assessed_pmid in tqdm.tqdm(pmid_pairs, total=len(pmid_pairs), desc="Calculating Similarities"):
        try:
            ref_pmid_vector = embeddings_dict[ref_pmid]
            assessed_pmid_vector = embeddings_dict[assessed_pmid]
            if ref_pmid_vector is not None and assessed_pmid_vector is not None:
                cosine_similarity = round(calculate_cosine_similarity(ref_pmid_vector, assessed_pmid_vector), 4)
                relevance_matrix_df.loc[(relevance_matrix_df['PID1'] == ref_pmid) & (relevance_matrix_df['PID2'] == assessed_pmid), 
                                        'Cosine Similarity'] = cosine_similarity
            else:
                continue
                #print(f"One of the vectors is None for ({ref_pmid}, {assessed_pmid})")
        except KeyError as e:
            #print(f"\nKeyError: {e}, ref_pmid: {ref_pmid}, assessed_pmid: {assessed_pmid}")
            continue
    
    # Saves the updated matrix 
    relevance_matrix_df.to_csv(output_matrix_name, index=False, sep="\t")
    print('Saved matrix')

def generate_embeddings(model: Doc2Vec, pmids: List[str], docs: List[List[str]]):
    document_embeddings = []
    
    for pmid in pmids:
        # Accessing embedding for each pmid
        embedding_vector = model.docvecs[str(pmid)]
        document_embeddings.append(embedding_vector)  
        
    #save_embeddings_to_pickle(pmids, embeddings_list, output_file)
    data = {"PID": pmids, "Embedding": document_embeddings}
    embeddings_df = pd.DataFrame(data)
    embeddings_df = embeddings_df.sort_values("PID")
    return embeddings_df

def save_embeddings_to_pickle(df, output_file):
    #data = {"PID": pmids, "Embedding": embeddings_list}
    #df = pd.DataFrame(data)
    df.to_pickle(output_file)