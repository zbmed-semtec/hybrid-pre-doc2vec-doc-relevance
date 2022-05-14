import argparse
import glob
import re
import sys
import os
from xml.etree import ElementTree as ET
import logging

#logging.getLogger().setLevel(logging.INFO)


# TODO: move commented code to its own file.
#import nltk
# def remove_stopwords(text: str) -> str:
#     """
#     Remove the english stopwords from the text.

#     @type text:     string
#     @param text:    text to remove the stopwords from

#     @rtype:         string
#     @returns:       the text without stopwords
#     """
#     nltk.download("stopwords")
#     stopwords = set(nltk.corpus.stopwords.words("english"))

#     output = " ".join( [word for word in text.split(" ") if word not in stopwords])
#     return output


def argument_parser(args: argparse.ArgumentParser):
    """
    Reads the arguments and creates the necesary variables for the program
    operate properly.

    By default, if no output is provided, it will add "_translated" either to
    the file path or the folder path provided.

    @type args:         argparse.ArgumentParser  
    @param args:        arguments from argparse

    @rtype files_in:    str
    @return files_in:   list of input files
    @rtype files_out:   str
    @return files_out:  list of output files
    """
    if args.indir:
        indir = args.indir.rstrip("/")
        files_in = glob.glob(indir + "/*.xml")

        outdir = args.output if args.output else indir + "_translated"
        files_out = list(map(lambda x: x.replace(indir, outdir), files_in))

        logging.info(f"Files from {indir} will be trasnlated into {outdir}")

        if not os.path.exists(outdir):
            logging.info(f"Output directory created at {outdir}")
            os.mkdir(outdir)
    else:
        files_in = [args.input]
        if args.output:
            files_out = [args.output]
        else:
            files_out = [args.input.replace(".xml", "_translated.xml")]

    return files_in, files_out


class XMLtrans:
    def __init__(self, input_file: str, verify_integrity: bool = False) -> None:
        """
        Initialize an object from XMLtrans class.

        @type input_file:           string
        @param input_file:          path for input file
        @type verify_integrity:     bool
        @param verify_integrity:    whether to verify that all tagged concepts
                                    share the same MeSH ID
        """
        try:
            self.root = ET.parse(input_file).getroot()
        except Exception:
            logging.error(
                f"Input file ({input_file}) is not a valid XML file.", exc_info=True)
            sys.exit("Input file must have a valid XML format.")

        self.mesh_id_pattern = r"\/MESH\/(.*)"
        self.namespace = {
            "z": "https://github.com/zbmed-semtec/whatizit-dictionary-ner/"}

        self.trans_dict = self.create_dict(verify_integrity)

    def extract_mesh_id(self, tag: ET.Element) -> str:
        """
        From a matched z:mesh tag, it extracts the correspondent MeSH ID if the
        field "id" is found.

        @type tag:  ElementTree.Element
        @param tag: object correspondent of a <z:mesh></z:mesh> tag

        @rtype:     string
        @returns:   text with thethe mesh ID
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

        @type verify_integrity:     bool
        @param verify_integrity:    whether to verify that all tagged concepts share the same MeSH ID

        @rtype:                     dict
        @returns:                   the translation dictionary that matches between text and MeSH ID
        """
        trans_dict = {}

        for tagged in self.root.findall("document/passage/text/z:mesh", self.namespace):
            mesh_id = self.extract_mesh_id(tagged)
            if not tagged.text in trans_dict.keys():
                trans_dict[tagged.text] = mesh_id
            else:
                if verify_integrity and trans_dict[tagged.text] != mesh_id:
                    # TODO: Better error handling
                    print("ERROR")
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

    def output_text(self, split: int = 0) -> str:
        """
        Returns the modified text all joined or divided in title/abstract

        @type split:     bool
        @param split:    whether to split the output text it title/abstract

        @rtype:          str
        @returns:        the modified output text
        """
        if split == 0:
            out_text = " ".join(self.mod_text.values())
        else:
            out_text = ""
            for key, value in self.mod_text.items():
                out_text += key + ": " + value + "\n"
        return out_text

    def save_output(self, file_out: str, split: int = 0) -> None:
        """
        Saves the modified output text into the given path

        @type file_out:     str
        @param file_out:    path for output file
        @type split:        bool
        @param split:       whether to split the output text it title/abstract
        """
        out_text = self.output_text(split)

        with open(file_out, "w+") as file:
            file.write(out_text)
        logging.info(f"Successfully created file at {file_out}.")

    # TODO: move commented code to its own file.
    # def process_text(self) -> None:
    #     """
    #     Applies all the processes to clean the text for doc2vec algorithms
    #     """
    #     for key, value in self.mod_text.items():
    #         # To Lowercase
    #         value = value.lower()

    #         # Remove stopwords
    #         value = remove_stopwords(value)
    #         self.mod_text[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", type=str,
                       help="Path to input XML file")
    group.add_argument("-d", "--indir", type=str,
                       help="Path to input folder with XML files")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to output text file/dir")
    args = parser.parse_args()

    files_in, files_out = argument_parser(args)

    for i, file in enumerate(files_in):
        logging.info(f"File {file} open.")
        xml_translation = XMLtrans(file)
        xml_translation.translate()
        xml_translation.save_output(files_out[i])
