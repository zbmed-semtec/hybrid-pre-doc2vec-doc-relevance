import os
import sys

os.chdir("../../")
sys.path.append('code/embeddings/')
sys.path.append("code/counting_table")

import create_model as cm
import calculate_sim as cs
import generate_counting_table as ct

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

def execute_optimization(hp_df: pd.DataFrame, tagged_data: TaggedDocument, relevance_matrix: pd.DataFrame, dataset: str, output_dir: str, load_table: bool = False, save_model: bool = False, custom_relish = False):
    output_dir = f"testing/hyperparameter_optimization/runs/{output_dir}"
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    hp_df.to_csv(output_dir + "/hp_df.tsv", sep = "\t")

    for i, row in hp_df.iterrows():
        print(f"Starting row {i}:")
        params = row.to_dict()
        relevance_matrix_row = relevance_matrix.copy(deep = True)

        print(f"\tGenerate and train the model.")
        model = cm.generate_doc2vec_model(tagged_data, params)
        cm.train_doc2vec_model(model, tagged_data, time_train = True)

        # We shouldn't save each model, it would take too much disk space
        if save_model:
            print(f"\tSave the model.")
            cm.save_doc2vec_model(model, output_dir + f"/model_{i}.model")

        print(f"\tFill the relevance matrix.")
        cs.fill_relevance_matrix(relevance_matrix_row, model, dataset, verbose = 1)
        #cs.save_rel_matrix(relevance_matrix_row, output_dir + f"/relevance_matrix_{i}.tsv")

        output_counting_table = output_dir + f"/counting_table_{i}.tsv"
        if load_table and os.path.exists(output_counting_table):
            counting_table = pd.read_csv(output_counting_table, sep = "\t")
        else:
            if not custom_relish:
                counting_table = ct.create_counting_table(relevance_matrix_row, dataset = dataset)
            else:
                counting_table = ct.custom_counting_table_relish(relevance_matrix_row, dataset = dataset)

            ct.save_table(counting_table, output_counting_table)

        if not custom_relish:
            ct.plot_graph(counting_table, normalize = True, output_path = output_dir + f"/counting_diagram_{i}.png", dataset = dataset)
        else:
            ct.custom_plot_graph_relish(counting_table, normalize = True, output_path = output_dir + f"/counting_diagram_{i}.png", dataset = dataset)

        



