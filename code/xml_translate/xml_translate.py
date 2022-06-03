"""
This file aims to transform an annotated XML files obtained from
[Whatizit](https://github.com/zbmed-semtec/whatizit-dictionary-ner) to plain
text. The translation converts the tagged concepts by the dictionary to its
corresponding MeSH ID.

Example
-------
To execute the script, you can run the following command:

    $ python xml_translate.py --indir ../../data/sample_annotated_xml

Notes
-----
A more detailed tutorial can be found in
[`docs/xml_translate`](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/blob/main/docs/xml_translate/tutorial_xml_translate.ipynb)
"""
__version__ = "0.3.1"
__author__ = "Guillermo Rocamora Pérez"

import os
import sys

import argparse
import glob
import re
import logging
import pandas as pd

from xml.etree import ElementTree as ET
from typing import Union, List


# logging.getLogger().setLevel(logging.INFO)


def argument_parser(args: argparse.ArgumentParser) -> Union[List[str], List[str]]:
    """
    Reads the arguments and creates the necesary variables for the program
    operate properly.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Arguments from argparse.

    Returns
    -------
    files_in: list[str]
        List of input files.
    files_out: list[str]
        List of output files.
    """
    if args.indir:
        if args.indir.endswith(".xml"):
            logging.error(
                f"Your input is not a directory, please use --input instead.", exc_info=False)
            sys.exit("No valid input directory.")

        indir = args.indir.rstrip("/")
        files_in = glob.glob(indir + "/*.xml")

        outdir = args.output if args.output else indir + "_translated"
        files_out = list(map(lambda x: x.replace(indir, outdir), files_in))

        if not files_in:
            logging.error(
                f"No XML files located in the input directory.", exc_info=False)
            sys.exit("No valid input directory.")

        logging.info(f"Files from {indir} will be trasnlated into {outdir}")

        if not os.path.exists(outdir) and not args.tsv:
            logging.info(f"Output directory created at {outdir}")
            os.mkdir(outdir)
    elif args.input:
        files_in = [args.input]
        if args.output:
            files_out = [args.output]
        else:
            files_out = [args.input.replace(".xml", "_translated.xml")]

    return files_in, files_out


class XMLtrans:
    """
    Class to handle every aspect of the translation algorithm. Each object
    should correspond to a given XML file to be modified.

    Attributes
    ----------
    root: xml.etree.ElementTree.Element
        Root object of xml.etree after parsing the document.
    mesh_id_pattern: str
        Regular expression used to match the MeSH ID.
    namespace: dict
        Dictionary containing the namespace for the XML.
    trans_dict: dict
        Dictionary containing the translation between concept an MeSH ID.
    tagged_texts: dict
        Dictionary containing the tagged text from the XML file split in title
        and abstract.
    mod_text: dict
        Dictionary containing the modified text split in title and abstract.
    """

    def __init__(self, input_file: str, verify_integrity: bool = False) -> None:
        """
        Initialize an object from XMLtrans class.

        Parameters
        ----------
        input_file : str
            Path for input file.
        verify_integrity : bool, optional
            Whether to verify that all tagged concepts share the same MeSH ID,
            by default False.
        """
        try:
            self.tree = ET.parse(input_file)
            self.root = self.tree.getroot()
        except Exception:
            logging.error(
                f"Input file ({input_file}) is not a valid XML file.", exc_info=True)
            sys.exit("Input file must have a valid XML format.")

        self.mesh_id_pattern = r"\/MESH\/(.*)"
        self.namespace = {
            "z": "https://github.com/zbmed-semtec/whatizit-dictionary-ner/"}

        self.pmid = self.locate_pmid()
        self.trans_dict = self.create_dict(verify_integrity)

    def locate_pmid(self) -> str:
        """
        Locates the ID tag inside the XML file and returns it.

        Returns
        -------
        str
            PMID of the publication.
        """
        if self.root.find("document/id") is None:
            logging.warning("No PMID was found in the document. Defaults to 0")
            return 0

        return self.root.find("document/id").text.strip()

    def extract_mesh_id(self, tag: ET.Element) -> str:
        """
        From a matched z:mesh tag, it extracts the correspondent MeSH ID if the
        field "id" is found.

        Parameters
        ----------
        tag : ET.Element
            Object correspondent of a <z:mesh></z:mesh> tag.

        Returns
        -------
        mesh_id:
            Text with thethe mesh ID.
        """
        if tag.attrib.get("id"):
            mesh_id = "MeSH" + \
                re.search(self.mesh_id_pattern, tag.attrib.get("id")).group(1)
        else:
            mesh_id = tag.text.strip()
        return mesh_id

    def create_dict(self, verify_integrity: bool = False) -> dict:
        """
        Creates the translation dictionary.

        Parameters
        ----------
        verify_integrity : bool, optional
            Whether to verify that all tagged concepts share the same MeSH ID,
            by default False.

        Returns
        -------
        trans_dict: dict
            The translation dictionary that matches between text and MeSH ID.
        """
        trans_dict = {}

        for tagged in self.root.findall("document/passage/text/z:mesh", self.namespace):
            mesh_id = self.extract_mesh_id(tagged)
            if not tagged.text in trans_dict.keys():
                trans_dict[tagged.text] = mesh_id
            else:
                if verify_integrity and trans_dict[tagged.text] != mesh_id:
                    logging.error("There is no integrity in the MeSH IDs." +
                                  "Two same concepts have different MeSH IDs." +
                                  f" The problem is in: {trans_dict[tagged.text]}.")
        return trans_dict

    def translate(self) -> None:
        """
        Translates the XML file. It looks for the "passage" tag, which should
        contain both a description of the text under "infon" and the text
        itself. Using a translation dictionary, the tagged words with <z:mesh>
        are replaced by their MeSH ID.

        The translated text is stored in a dictionary according to the names
        found in the "infon" tag.
        """
        self.tagged_texts = {}
        self.mod_text = {}

        for passage in self.root.findall("document/passage"):
            # TODO: Error handling if no "infon" tag is found
            description = passage.find("infon").text.strip()
            text = passage.find("text")
            self.tagged_texts[description] = text

            local_text = []
            for concept in text.itertext():
                local_text.append(self.trans_dict.get(
                    concept.strip(), concept))
            self.mod_text[description] = "".join(local_text).strip()

    def save_xml(self, file_out: str) -> None:
        """
        Saves the modified output xml into the given path.

        Parameters
        ----------
        file_out : str
            Path for output file.
        """
        for passage in self.root.findall("document/passage"):
            description = passage.find("infon").text.strip()
            for text in passage.iter("text"):
                for child in list(text):
                    text.remove(child)
                text.text = "\n" + self.mod_text[description] + "\n"

        with open(file_out, "wb") as file:
            self.tree.write(file, encoding="utf-8", xml_declaration=True)

    def output_text(self, split: bool = False) -> str:
        """
        Returns the modified text all joined or divided in title/abstract.
        This functions is necessary to run the unittests.

        Parameters
        ----------
        split : int, optional
            Whether to split the output text it title/abstract, by default
            False.

        Returns
        -------
        out_text: str
            The modified output text.
        """
        if split == 0:
            out_text = " ".join(self.mod_text.values())
        else:
            out_text = ""
            for key, value in self.mod_text.items():
                out_text += key + ": " + value + "\n"
        return out_text

    def save_txt(self, file_out: str, split: bool = False) -> None:
        """
        Saves the modified output text into the given path.

        Parameters
        ----------
        file_out : str
            Path for output file.
        split : int, optional
            Whether to split the output text it title/abstract, by default
            False.
        """
        if not file_out.endswith(".txt"): file_out += ".txt"

        out_text = self.output_text(split)

        with open(file_out, "w+") as file:
            file.write(out_text)
        logging.info(f"Successfully created file at {file_out}.")


def translate_pipeline(files_in: List[str], output_file: str) -> pd.DataFrame:
    """
    Pipeline to translate all input files and to output a tab separated value
    file.

    Parameters
    ----------
    files_in : List[str]
        List of input files.
    output_file : str
        Path for output file.

    Returns
    -------
    publications_df: pd.DataFrame
        Generated pandas dataframe containing three columns: "PMID", "title"
        and "abstract".
    """
    if not output_file:
        # "../../data/sample_annotated_xml_translated/18394048_annotated.xml"
        output_file = files_in[0][0:files_in[0].rfind("/")]

    if not output_file.endswith(".tsv"):
        output_file = output_file + ".tsv"
    publications_list = []
    for _, file in enumerate(files_in):
        xml_dict = {}
        xml_translation = XMLtrans(file)
        xml_translation.translate()
        xml_dict["PMID"] = xml_translation.pmid
        xml_dict["title"] = xml_translation.mod_text["title"]
        xml_dict["abstract"] = xml_translation.mod_text["abstract"]
        publications_list.append(xml_dict)

    publications_df = pd.DataFrame(publications_list)
    publications_df.to_csv(output_file, sep="\t", index=False, quotechar="`")

    return publications_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", type=str,
                       help="Path to input XML file")
    group.add_argument("-d", "--indir", type=str,
                       help="Path to input folder with XML files")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to output text file/dir")
    parser.add_argument("--tsv", type=int, default=0,
                        help="Whether to output to a .TSV file, write 1 to do so")
    args = parser.parse_args()

    if not args.tsv and args.output and args.output.endswith(".tsv"):
        logging.warning(
            "Your output files ends with .TSV, but --tsv argument is set to 0." +
            "If you wish to output to a .TSV file, please change it to 1.")

    files_in, files_out = argument_parser(args)

    if args.tsv:
        translate_pipeline(files_in, args.output)
    else:
        for i, file in enumerate(files_in):
            logging.info(f"File {file} open.")
            xml_translation = XMLtrans(file)
            xml_translation.translate()
            xml_translation.save_xml(files_out[i])
    #file = files_in[0]
    #xml_translation = XMLtrans(file)
    # xml_translation.translate()
