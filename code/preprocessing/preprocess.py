"""
This file contains the necessary functions to run the text preprocessing
required for the hybrid approach.

Example
-------
To execute the script, you can run the following command:

    $ python code/preprocessing/preprocess.py --input data/RELISH/RELISH_documents_20220628_ann_swr.tsv --output data/RELISH/RELISH_tokens.tsv


Notes
-----
A more detailed tutorial can be found in
[`docs/xml_translate`](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/tree/main/docs/preprocessing)
"""
import sys

import argparse
import logging

import re
import numpy as np
import pandas as pd
from string import punctuation
from typing import List

__version__ = "0.1.2"
__author__ = "Guillermo Rocamora Pérez"

def read_data(input_path: str) -> pd.DataFrame:
    """
    Reads the input TSV files.

    Parameters
    ----------
    input_path : str
        Input path for the TSV files.

    Returns
    -------
    pd.DataFrame
        DataFrame with three columns: PMID, title and abstract
    """
    return pd.read_csv(input_path, delimiter="\t", quotechar="`")


def preprocess_text(text: str, alphanumeric_pattern: str = r"[a-zA-Z\d]", allowed_punctuation: List[str] = ["-"], process_abv: bool = False) -> str:
    """
    Preprocess a given string. The steps are: 
        1. (Optional) Convert abbreviations such as "E. Coli" to a single word.
        2. Set all the characters to lowercase.
        3. Remove the desired punctuations from text.
        4. Split the text by white spaces.

    Parameters
    ----------
    text : str
        Text to be preprocessed. It must a be a single string.
    alphanumeric_pattern: str, optional
        String containing the REGEX pattern to identify alphanumeric
        characters.
    allowed_punctuation: str, optional
        List of allowed punctuation characters, by default only "-" will be
        kept.
    process_abv : bool, optional
        Whether modify abbreviations such as "E. coli", where it is transformed
        as ecoli, by default False.

    Returns
    -------
    text: str
        The preprocessed text file
    """
    if process_abv:
        abv_pattern = r"((?:^|\s)[a-zA-Z]\.)(\s?)([a-z]\w+)"
        text = re.sub(abv_pattern, r"\1\3", text)

    text_list = []
    for word in text.lower().split():
        word_list = []
        for character in word:
            if re.match(alphanumeric_pattern, character) or character in allowed_punctuation:
                word_list.append(character)
        word = "".join(word_list)
        text_list.append(word)
    text = " ".join(text_list).strip()

    return text


def preprocess_data(data: pd.DataFrame, alphanumeric_pattern: str = r"[a-zA-Z\d]", allowed_punctuation: List[str] = ["-"], process_abv: bool = False) -> pd.DataFrame:
    """
    Main pipeline to read the input DataFrame and preprocess both the title and
    abstract columns.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with three columns: PMID, title and abstract.
    alphanumeric_pattern: str, optional
        String containing the REGEX pattern to identify alphanumeric
        characters.
    allowed_punctuation: str, optional
        List of allowed punctuation characters, by default only "-" will be
        kept.
    process_abv : bool, optional
        Whether modify abbreviations such as "E. coli", where it is transformed
        as ecoli, by default False.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with three columns: PMID, title and abstract
    """
    for i, row in data.iterrows():
        title = row["title"]
        abstract = row["abstract"]
        #title = "MeSHD003643 signal-induced localization of p53 MeSHD011506 to MeSHD008928. A potential MeSHD012380 in apoptotic signaling. The element's H(2)O that we've described. The study of E. coli bacteria in E.coli sickness."

        title = preprocess_text(title, alphanumeric_pattern, allowed_punctuation, process_abv)
        abstract = preprocess_text(abstract, alphanumeric_pattern, allowed_punctuation, process_abv)

        data.at[i, "title"] = title
        data.at[i, "abstract"] = abstract

    return(data)


def save_output(data: pd.DataFrame, output_path: str, npy_format: bool = False) -> None:
    """
    Saves the dataframe into a TSV file

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed dataframe to be saved.
    output_path : str
        Path for output file.
    npy_format : bool, optional
        Whether to save the output file in numpy format, by default False. It
        is not recommended to do so, since a TSV file is usually faster and
        takes less disk usage.
    """
    if npy_format:
        if output_path.endswith(".tsv"):
            output_path = output_path.replace(".tsv", ".npy")
        output_data = []
        for i, row in data.iterrows():
            output_data.append([np.asanyarray(data.at[i, "PMID"]), 
                np.asanyarray(data.at[i, "title"].split(), dtype=object), 
                np.asanyarray(data.at[i, "abstract"].split(), dtype=object)])

        np.save(output_path, np.asanyarray(output_data, dtype = object))
    else:
        if not output_path.endswith(".tsv"):
            output_path = output_path + ".tsv"
        data.to_csv(output_path, sep="\t", index=False, quotechar="`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Path to input TSV file", required=True)
    parser.add_argument("-o", "--output", type=str,
                        help="Path to output TSV file")
    parser.add_argument("--npy_format", help="Whether to save the output in numpy format.", type=int, default=0)
    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        logging.error(
            f"The input file must be a .tsv, or tab separated values", exc_info=False)
        sys.exit("No valid input file.")

    if not args.output:
        if args.npy_format:
            output_path = args.input.replace(".tsv", "_preprocessed.npy")
        else:
            output_path = args.input.replace(".tsv", "_preprocessed.tsv")
        logging.warning(
            f"No output path specified, defaulting to {output_path}")
    else:
        output_path = args.output

    if output_path.endswith(".npy") and not args.npy_format:
        logging.warning("The output path is a NPY format, but the argument 'npy_format' is set to False. The results will be save in numpy format nontheless.")
        args.npy_format = 1
    elif output_path.endswith(".tsv") and args.npy_format:
        logging.warning("The output path is a TSV format, but the argument 'npy_format' is set to True. The results will be save in TSV format nontheless.")
        args.npy_format = 0
    abv_pattern = r"((?:^|\s)[a-zA-Z]\.)(\s?)([a-z]\w+)"
    alphanumeric_pattern = r"[a-zA-Z\d]"
    allowed_punctuation = ["-"]
    
    data = read_data(args.input)
    data = preprocess_data(data, alphanumeric_pattern, allowed_punctuation)
    
    if not args.npy_format:
        save_output(data, output_path, npy_format=False)
    else:
        save_output(data, output_path, npy_format=True)
