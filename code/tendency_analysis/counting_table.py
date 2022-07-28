import math
import sys

import numpy as np
import pandas as pd
import logging

from matplotlib import pyplot as plt 

def load_relevance_matrix(input_path: str) -> pd.DataFrame:
    """
    Reads a .TSV file containing the relevance matrix. Three columns are
    needed: PMID 1, PMID 2 and relevance (for RELISH) or group (for TREC).

    Parameters
    ----------
    input_path: str
        File path to the Relevance Matrix. It is used to populate the forth
        column. 

    Returns
    -------
    data: pd.DataFrame
        Dataframe with at least three columns: PMID 1, PMID 2, and either
        relevance (for RELISH) or group (for TREC).
    """
    if not input_path.endswith(".tsv"):
        logging.warning("A tab separated value (.TSV) file is recommended.")

    data = pd.read_csv(input_path, sep = "\t")
    return data

def count_entries(data: pd.DataFrame, interval: float, dataset: str = "RELISH") -> dict:
    """
    Counts the number of Relevance Assessments or Groups for a given value of
    Cosine Similarity.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe with 4 columns: PMID 1, PMID 2, relevance/group and
        Cosine Similarity.
    interval: float
        Value of Cosine Similarity to count the entries.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".

    Returns
    -------
    counter: dict
        Dictionary containing the counts for each relevance/group.
    """
    if dataset not in ["TREC", "RELISH"]:
        logging.error("The dataset must be either TREC or RELISH in order to properly process the relevance matrix.")
        sys.exit("Invalid dataset provided for counting the entries.")

    if dataset == "RELISH":
        filtered_df = data[data["Cosine Similarity"] == interval]["relevance"]
        counter = {0: sum(filtered_df == 0), 1: sum(filtered_df == 1), 2: sum(filtered_df == 2)}
    elif dataset == "TREC":
        filtered_df = data[data["Cosine Similarity"] == interval]["Group"]
        counter = {'A': sum(filtered_df == 'A'), 'B': sum(filtered_df == 'B'), 'C': sum(filtered_df == 'C')}

    return counter

def create_counting_table(data: pd.DataFrame, dataset: str = "RELISH") -> pd.DataFrame:
    """
    Creates the "counting table" from a given Relevance matrix.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe with 4 columns: PMID 1, PMID 2, relevance/group and
        Cosine Similarity.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".

    Returns
    -------
    counting_df: pd.DataFrame
        DataFrame of the counting table generated.
    """
    if dataset not in ["TREC", "RELISH"]:
        logging.error("The dataset must be either TREC or RELISH in order to properly process the relevance matrix.")
        sys.exit("Invalid dataset provided to generate the counting table.")

    if dataset == "RELISH":
        counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(), "2s": 0, "1s": 0, "0s": 0})

        for i, row in counting_df.iterrows():
            interval = row["Cosine Interval"]
            interval_counts = count_entries(data, interval, dataset = dataset)

            counting_df.at[i, "2s"] = interval_counts[2]
            counting_df.at[i, "1s"] = interval_counts[1]
            counting_df.at[i, "0s"] = interval_counts[0]

    elif dataset == "TREC":
        counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(),
                                    "As": 0, "Bs": 0, "Cs": 0})

        for i, row in counting_df.iterrows():
            interval = row["Cosine Interval"]
            interval_counts = count_entries(data, interval, dataset = dataset)

            counting_df.at[i, "As"] = interval_counts['A']
            counting_df.at[i, "Bs"] = interval_counts['B']
            counting_df.at[i, "Cs"] = interval_counts['C']
        
    return counting_df

def hp_create_counting_table(data: pd.DataFrame, dataset: str = "RELISH") -> pd.DataFrame:
    """
    Creates the "counting table" from a given Relevance matrix in the
    hyperparameter optimization process. The main difference from
    create_counting_table() function is that the relevant groups (either 2s and
    1s or As and Bs) are joined toguether to discriminate between relevant and
    non-relevant publications.

    UNDER DEVELOPMENT.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe with 4 columns: PMID 1, PMID 2, relevance/group and
        Cosine Similarity.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".

    Returns
    -------
    counting_df: pd.DataFrame
        DataFrame of the counting table generated.
    """
    if dataset not in ["TREC", "RELISH"]:
        logging.error("The dataset must be either TREC or RELISH in order to properly process the relevance matrix.")
        sys.exit("Invalid dataset provided to generate the counting table.")

    if dataset == "RELISH":
        counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(), "2s": 0, "0s": 0})

        for i, row in counting_df.iterrows():
            interval = row["Cosine Interval"]
            interval_counts = count_entries(data, interval, dataset = dataset)

            counting_df.at[i, "2s"] = interval_counts[2] + interval_counts[1]
            counting_df.at[i, "0s"] = interval_counts[0]

    elif dataset == "TREC":
        counting_df = pd.DataFrame({"Cosine Interval":  np.round(np.linspace(0, 1, 101), 2).tolist(),
                                    "As": 0, "Cs": 0})

        for i, row in counting_df.iterrows():
            interval = row["Cosine Interval"]
            interval_counts = count_entries(data, interval, dataset = dataset)

            counting_df.at[i, "As"] = interval_counts['A'] + interval_counts['B']
            counting_df.at[i, "Cs"] = interval_counts['C']
        
    return counting_df


def plot_graph(data: pd.DataFrame, dataset: str = "RELISH", normalize: bool = False, show_figure: bool = True, output_path: str = None) -> None:
    """
    Plots the graph of "Relevance counting" against the "Cosine intervals" for
    the number the different relevances/groups found in the input counting
    table.
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the counting table.
    dataset: str, optional
        String to determine the dataset. Must be either RELISH or TREC, by
        default "RELISH".
    normalize: bool, optional
        Boolean to determine whether to normalize the plotted histograms so
        that the sum adds up to one, by default False.
    show_figure: bool, optional
        Boolean to determine whether to print the pyplot figure, by default
        True.
    output_path : str, optional
        If an output path is given, the figure will be saved, by default None.
    """
    if dataset not in ["TREC", "RELISH"]:
        logging.error("The dataset must be either TREC or RELISH in order to properly process the relevance matrix.")
        sys.exit("Invalid dataset provided to plot the counting table")

    intervals = data["Cosine Interval"].values.tolist()

    if dataset == "RELISH":    
        two_points = data["2s"].values.tolist()
        zero_points = data["0s"].values.tolist()

        if normalize:
            two_points = [i/sum(two_points) for i in two_points]
            zero_points = [i/sum(zero_points) for i in zero_points]
        
        plt.plot(intervals, two_points, 'r', label='2 counts')  
        plt.plot(intervals, zero_points, 'g', label='0 counts')

        if "1s" in data.columns:
            one_points = data["1s"].values.tolist()
            if normalize:
                one_points = [i/sum(one_points) for i in one_points]
            plt.plot(intervals, one_points, 'g', label='0 counts')
    elif dataset == "TREC":
        two_points = data["As"].values.tolist()
        zero_points = data["Cs"].values.tolist()

        if normalize:
            two_points = [i/sum(two_points) for i in two_points]
            zero_points = [i/sum(zero_points) for i in zero_points]
        
        plt.plot(intervals, two_points, 'r', label='A counts')  
        plt.plot(intervals, zero_points, 'g', label='C counts')

        if "1s" in data.columns:
            one_points = data["Bs"].values.tolist()
            if normalize:
                one_points = [i/sum(one_points) for i in one_points]
            plt.plot(intervals, one_points, 'g', label='B counts')

    plt.xlabel("Cosine intervals")
    plt.ylabel("Relevance counting")

    plt.legend()
    if output_path != "none":
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor="white")
    
    if show_figure:
        plt.show()
    plt.close()

def save_table(counting_df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the counting table into .TSV format.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the counting table.
    output_path : str
        Output path to where the counting table will be saved.
    """
    counting_df.to_csv(output_path, index=False, sep = "\t")