import os
import argparse
import itertools
import logging
import embeddings as em
import embeddings_dataframe as ed

log_file = "hybrid/doc2vec.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def run(params_dict: dict, input_file: str):
    """
    Wrapper function to create Tagged Documents, generate & train Doc2Vec models, generate embeddings and
    store embeddings in a pickle file.
    Parameters
    ----------
    params_dict : dict
        Parameter dictionary consisting of all hyperparameters, combinations of which are used to generate Doc2Vec models.
    input_file : str
        Input RELISH tokenized npy file.
    """
    all_combinations = list(itertools.product(*params_dict.values()))

    param_combinations = [{key: value for key, value in zip(params_dict.keys(), combination)} for combination in all_combinations]

    # Retrieves cleaned data from .npy file 
    pmids, titles, abstracts, docs = em.process_data_from_npy(input_file)
    logging.info("Retrieved RELISH Cleaned Data")

    for idx, params in enumerate(param_combinations):
        logging.info(f"Combination {idx + 1}/{len(all_combinations)}")
        # Create and train Doc2Vec model
        model = em.createDoc2VecModel(pmids, docs, params)
        logging.info(f"RELISH Doc2Vec Model {idx} Generated")

        # Define a directory for storing models
        models_directory = f"models/"

        # Ensure the directory exists or create it
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)

        # Save the model generated
        em.saveDoc2VecModel(model, f"models/relish_doc2vec_{idx}.model")
        logging.info(f"RELISH Doc2Vec Model {idx} Saved")

        # Define a directory for storing embeddings
        embeddings_directory = f"embeddings/embeddings_doc2vec_{idx}/"

        # Ensure the directory exists or create it
        if not os.path.exists(embeddings_directory):
            os.makedirs(embeddings_directory)
            
        # Generate the embeddings
        em.create_document_embeddings(pmids, model, embeddings_directory)
        logging.info("RELISH Embeddings Generated")

        # Define a directory for storing embeddings pickle file
        embeddings_dataframe_directory = f"dataframe/"

        # Ensure the directory exists or create it
        if not os.path.exists(embeddings_dataframe_directory):
            os.makedirs(embeddings_dataframe_directory)

        # Generate embeddings dataframe pickle file
        pmids = ed.sort_pmids(pmids)
        embeddings_list = ed.get_embeddings(embeddings_directory, pmids)
        ed.save_dataframe(pmids, embeddings_list, f"dataframe/embeddings_pickle_{idx}.pkl")
        logging.info(f'Generated {idx} pickle dataframe')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                       help="Path to input RELISH annotated and preprocessed tokenized .npy file")                 
    args = parser.parse_args()

    params_dict = {
    "dm": [0, 1],
    "vector_size": [200, 300, 400],
    "window": [5, 6, 7],
    "min_count": [5],
    "epochs": [15],
    "workers": [8]
    }

    run(params_dict, args.input)