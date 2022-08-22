import sys

import argparse
import logging

import re
import numpy as np
import pandas as pd
from string import punctuation
from typing import List


def read_data(input_path: str) -> pd.DataFrame:
    """
    Reads the input .TSV files

    Parameters
    ----------
    input_path : str
        Input path for the .TSV files.

    Returns
    -------
    pd.DataFrame
        DataFrame with three columns: PMID, title and abstract
    """
    return pd.read_csv(input_path, delimiter="\t", quotechar="`")


def preprocess_text(text: str, alphanumeric_pattern: str = r"[a-zA-Z\d]", allowed_punctuation: List[str] = ["-"], process_abv: bool = False) -> str:
    """
    Preprocess a given string. The steps are: 
        1. Convert abbreviations such as "E. Coli" to a single word.
        2. Set all the characters to lowercase.
        3. Remove the desired punctuations from text.

    Parameters
    ----------
    text : str
        Text to be preprocessed. It must a be a single string.
    process_abv : bool, optional
        Whether modify abbreviations such as "E. coli", where it is transformed
        as ecoli, by default False.
    alphanumeric_pattern: str, optional
        String containing the REGEX pattern to identify alphanumeric
        characters.
    allowed_punctuation: str, optional
        List of allowed punctuation characters, by default only "-" will be
        kept.

    Returns
    -------
    text: str
        The preprocecessed text file
    """
    if process_abv:
        abv_pattern = r"((?:^|\s)[a-zA-Z]\.)(\s?)([a-z]\w+)"
        text = re.sub(abv_pattern, r"\1\3", text)

    text_list = []
    for word in text.lower().split():
        word_list = []
        for character in word:
            # print(character)
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
        Dataframe with three columns: PMID, title and abstract
    process_abv : bool, optional
        Whether modify abbreviations such as "E. coli", where it is transformed
        as ecoli, by default False.
    alphanumeric_pattern: str, optional
        String containing the REGEX pattern to identify alphanumeric
        characters.
    allowed_punctuation: str, optional
        List of allowed punctuation characters, by default only "-" will be
        kept.

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
    Saves the dataframe into a .tsv file

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed dataframe to be saved.
    output_path : str
        Path for output file.
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
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-i", "--input", type=str,
                        help="Path to input TSV file", required=True)
    parser.add_argument("-o", "--output", type=str,
                        help="Path to output TSV file")
    args = parser.parse_args()

    if not args.input.endswith(".tsv"):
        logging.error(
            f"The input file must be a .tsv, or tab separated values", exc_info=False)
        sys.exit("No valid input file.")

    if not args.output:
        output_path = args.input.replace(".tsv", "_preprocessed.tsv")
        logging.warning(
            f"No output path specified, defaulting to {output_path}")
    else:
        output_path = args.output

    abv_pattern = r"((?:^|\s)[a-zA-Z]\.)(\s?)([a-z]\w+)"
    alphanumeric_pattern = r"[a-zA-Z\d]"
    #alphanumeric_pattern = r".*[a-zA-Z\d\-].*"
    allowed_punctuation = ["-"]
    
    data = read_data(args.input)
    data = preprocess_data(data, alphanumeric_pattern, allowed_punctuation)
    save_output(data, output_path, npy_format=False)
    save_output(data, output_path, npy_format=True)
