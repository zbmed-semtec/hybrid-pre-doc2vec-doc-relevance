"""
THIS FILE IS DEPRECATED AND WILL BE REMOVED SHORTLY.

PLEASE GO TO code/tendency_analysis
"""

import os
import sys

sys.path.append(os.path.expanduser("~/hybrid-dictionary-ner-doc2vec-doc-relevance/code/embeddings/"))
sys.path.append(os.path.expanduser("~/hybrid-dictionary-ner-doc2vec-doc-relevance/code/counting_table/"))
sys.path.append(os.path.expanduser("~/hybrid-dictionary-ner-doc2vec-doc-relevance/testing/ROC"))

import create_model as cm
import calculate_sim as cs
import generate_counting_table as ct
import ROC as ROC

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def generate_hyperparameters(params):
    try:
        from sklearn.model_selection import ParameterGrid

        param_grid = ParameterGrid(params)
        df_hp = pd.DataFrame.from_dict(param_grid)

        return df_hp
    except:
        import itertools

        keys, values = zip(*params.items())
        df_hp = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return df_hp

def execute_optimization(hp_df: pd.DataFrame, 
        tagged_data: TaggedDocument, 
        relevance_matrix: pd.DataFrame, 
        dataset: str, 
        output_dir: str, 
        load_model: bool = False,
        save_model: bool = False, 
        load_rel_matrix: bool = False,
        save_rel_matrix: bool = False,
        load_counting_table: bool = False,
        save_counting_table: bool = False,
        custom_relish = False):

    output_dir = f"testing/hyperparameter_optimization/runs/{output_dir}"
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    hp_df_path = output_dir + "/hp_df.tsv"
    hp_df.to_csv(hp_df_path, sep = "\t")

    hp_df["AUC"] = 0
    for i, row in hp_df.iterrows():
        print(f"Starting row {i}:")
        params = row.to_dict()
        del params["AUC"]

        relevance_matrix_row = relevance_matrix.copy(deep = True)

        # Models
        print(f"\tGenerate and train the model.")
        model_path = output_dir + f"/model_{i}.model"
        if load_model and os.path.exists(model_path):
            model = cs.load_d2v_model(model_path)
        else:
            model = cm.generate_doc2vec_model(tagged_data, params)
            cm.train_doc2vec_model(model, tagged_data, time_train = True)

            # We shouldn't save each model, it would take too much disk space
            if save_model:
                print(f"\tSave the model.")
                cm.save_doc2vec_model(model, output_dir + f"/model_{i}.model")

        # Relevance Matrix
        print(f"\tFill the relevance matrix.")
        rel_matrix_path = output_dir + f"/relevance_matrix_{i}.tsv"
        if load_rel_matrix and os.path.exists(rel_matrix_path):
            relevance_matrix_row = cm.load_relevance_matrix(rel_matrix_path)
        else:
            cs.fill_relevance_matrix_multiprocess(relevance_matrix_row, model)
            if save_rel_matrix:
                print(f"\tSaving the relevance matrix.")
                cs.save_rel_matrix(relevance_matrix_row, rel_matrix_path)

        # Counting Table
        print(f"\tGenerating the counting table.")
        counting_table_path = output_dir + f"/counting_table_{i}.tsv"
        if load_counting_table and os.path.exists(counting_table_path):
            counting_table = pd.read_csv(counting_table_path, sep = "\t")
        else:
            counting_table = ct.hp_create_counting_table(relevance_matrix_row, dataset = dataset)
            if save_counting_table:
                print(f"\tSaving the counting table.")
                ct.save_table(counting_table, counting_table_path)

        # Measure and save ROC
        print(f"\tPlotting the ROC curve.")
        ROC_path = output_dir + f"/ROC_{i}.png"
        ROC.generate_roc_values(counting_table, dataset = dataset)
        ROC.draw_roc_curve(counting_table, draw_auc = True, show_figure = False, output_path = ROC_path)

        hp_df.at[i, "AUC"] = ROC.calculate_auc(counting_table)
        hp_df.to_csv(hp_df_path, sep = "\t")
    
    return hp_df





        



