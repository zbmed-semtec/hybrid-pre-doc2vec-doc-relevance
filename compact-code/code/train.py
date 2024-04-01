# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py
# This file includes the modifications to the source codes according to this project!

import os
import time
import argparse
import utilities as utilities

def run(best_params, args, iteration, save_model=False):
    start = time.time()
    
    # Loading data
    test_pmids, test_docs = utilities.process_data_from_npy(args.input)
    print(f"Retrieved RELISH Cleaned Data for hyperparameter-set {iteration}.")

    # Train the model with 80% of the data (i.e. training data) and best parameters
    model = utilities.createDoc2VecModel(test_pmids, test_docs, best_params)

    print(f"RELISH Hybrid Dord2Vec Model Generated for hyperparameter-set {iteration}.")

    # Define the file path for Storing test Embeddings
    embeddings_file = f"output_of_model/doc_embeddings/embeddings_pickle_{iteration}.pkl"
    # Generate the embeddings
    embeddings_df = utilities.generate_embeddings(model, test_pmids, test_docs)
    # Store the embeddings
    utilities.save_embeddings_to_pickle(embeddings_df, embeddings_file)
    print(f"RELISH Embeddings Pickle File Saved for hyperparameter-set {iteration}.")
    # Define the file path for Storing cosine similarity matrix
    similarity_file = f"output_of_model/evaluation/cosine_similarity_{iteration}.tsv"
    # Generate and save the cosine similarity matrix
    utilities.get_and_save_similarity_scores(args.ground_truth, embeddings_file, similarity_file)
    print(f"RELISH Cosine Similarity Matrix Saved for hyperparameter-set {iteration}.")
    '''        
    if save_model:
        
        # Define the file path for saving the model
        model_file = f"output_of_model/model/{iteration}/Doc2Vec_model"
        # Save the model
        utilities.saveDoc2VecModel(model, model_file)
    ''' 
    end = time.time()
    print(f"Time Taken for Creation and Evaluation of the Model for hyperparameter-set {iteration}: {end - start} seconds.")
    
    return similarity_file