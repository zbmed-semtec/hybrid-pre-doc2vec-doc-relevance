import create_model as cm
import calculate_sim as cs

params_d2v = {
    "dm": 0,
    "vector_size": 200, 
    "window": 5, 
    "min_count": 5, 
    "epochs": 10, 
    "workers": 4}

# RELISH
def create_relish_model():
    input_tokens_relish = "../../../data_full/RELISH_tokens.tsv"
    output_model_relish = "models/hybrid_d2v_relish.model"

    pmid_relish, join_text_relish = cm.load_tokens(input_tokens_relish)
    tagged_data_relish = cm.generate_TaggedDocument(pmid_relish, join_text_relish)

    model_relish = cm.generate_doc2vec_model(tagged_data_relish, params_d2v)
    cm.train_doc2vec_model(model_relish, tagged_data_relish, time_train = True)
    cm.save_doc2vec_model(model_relish, output_model_relish)

def fill_relish_relevance_matrix():
    input_relevance_matrix_relish = "../../../data_full/relish_relevance_matrix.csv"
    input_model_relish = "models/hybrid_d2v_relish.model"
    output_relevance_matrix_relish = "../../../data_full/relish_relevance_matrix_filled.tsv"

    rel_matrix_relish = cs.load_relevance_matrix(input_relevance_matrix_relish)
    model_relish = cs.load_d2v_model(input_model_relish)

    cs.fill_relevance_matrix(rel_matrix_relish, model_relish, verbose = 1)
    cs.save_rel_matrix(rel_matrix_relish, output_relevance_matrix_relish)

# TREC
def create_trec_model():
    input_tokens_trec = "../../../data_full/TREC_tokens.tsv"
    output_model_trec = "models/hybrid_d2v_trec.model"
    pmid_trec, join_text_trec = cm.load_tokens(input_tokens_trec)
    tagged_data_trec = cm.generate_TaggedDocument(pmid_trec, join_text_trec)

    model_trec = cm.generate_doc2vec_model(tagged_data_trec, params_d2v)
    cm.train_doc2vec_model(model_trec, tagged_data_trec, time_train = True)
    cm.save_doc2vec_model(model_trec, output_model_trec)

def fill_trec_relevance_matrix():
    input_relevance_matrix_trec = "../../../data_full/trec_relevance_matrix.csv"
    input_model_trec = "models/hybrid_d2v_trec.model"
    output_relevance_matrix_trec = "../../../data_full/trec_relevance_matrix_filled.tsv"

    rel_matrix_trec = cs.load_relevance_matrix(input_relevance_matrix_trec)
    model_trec = cs.load_d2v_model(input_model_trec)

    cs.fill_relevance_matrix(rel_matrix_trec, model_trec, verbose = 1)
    cs.save_rel_matrix(rel_matrix_trec, output_relevance_matrix_trec)