# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/main.py
# This file includes the modifications to the source codes according to this project!

import os
import argparse
import json
from train import run
import precision
import precision_two_class
import calculate_gain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to input (Annotated data) .npy file")
    parser.add_argument("-gt", "--ground_truth", type=str, help="Path to ground truth .tsv file")
    parser.add_argument("-pj", "--params_json", type=str, help="File location of word2vec parameter list.")

    args = parser.parse_args()
    
    # Extract model hyperparameters
    params = []
    with open(args.params_json, "r") as openfile:
        params = json.load(openfile)
    
    # Define the directory for storing pipeline outputs
    output_directory = "output_of_model"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Define the directory for saving the model
    model_directory = "output_of_model/model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # Define the Directory for Storing Embeddings
    embeddings_directory = "output_of_model/doc_embeddings"
    if not os.path.exists(embeddings_directory):
        os.makedirs(embeddings_directory)
    # Define the directory for storing evaluation results
    results_directory = "output_of_model/evaluation"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
       
    for iteration in range(len(params)):
        print(f'start for parameter-set {iteration}')
        
        #os.makedirs(f"output_of_model/model/{iteration}", exist_ok=True)
    
        similarity_file = run(params[iteration], args, iteration, save_model=False)
    
        precision_file = os.path.join(results_directory, f"precision_three_class_{iteration}.tsv")
        precision_file_two_class = os.path.join(results_directory, f"precision_two_class_{iteration}.tsv")
        dcg_file = os.path.join(results_directory, f"dcg_{iteration}.tsv")
        idcg_file = os.path.join(results_directory, f"idcg_{iteration}.tsv")
        ndcg_file = os.path.join(results_directory, f"ndcg_{iteration}.tsv")

        # Generate and save the three-class precision matrix
        ref_pmids, data = precision.read_file(similarity_file)
        matrix = precision.generate_matrix(ref_pmids, data)
        precision.write_to_tsv(ref_pmids, matrix, precision_file)
        print(f"Three-class Precision matrix saved for parameter-set {iteration}")
    
        # Generate and save the two-class precision matrix
        ref_pmids, data = precision_two_class.read_file(similarity_file)
        matrix = precision_two_class.generate_matrix(ref_pmids, data)
        precision_two_class.write_to_tsv(ref_pmids, matrix, precision_file_two_class)
        print(f"Two-class Precision matrix saved for parameter-set {iteration}")

        # Generate and save the DCG and IDCG matrices
        sim_matrix = calculate_gain.load_cosine_sim_matrix(similarity_file)
        calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
        calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
        all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(dcg_file, idcg_file)
        calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
        
        print(f"DCG, IDCG, and NDCG matrices saved for parameter-set {iteration}")
        
        # Delete dcg and idcg files whose results are summarized in ndcg
        try:
            os.remove(os.path.join(results_directory, f"dcg_{iteration}.tsv"))
            os.remove(os.path.join(results_directory, f"idcg_{iteration}.tsv"))
        except FileNotFoundError:
            continue



