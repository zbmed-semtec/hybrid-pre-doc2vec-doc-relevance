import argparse
import re
import os
from xml.etree import ElementTree as ET
import nltk

def remove_stopwords(text: str) -> str:
    nltk.download('stopwords')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    output = " ".join([word for word in text.split(" ") if word not in stopwords])
    return output

class XMLtrans:
    def __init__(self, input_file: str, verify_integrity: bool = False) -> None:
        self.root = ET.parse(input_file).getroot()
        
        self.mesh_id_pattern = r'\/MESH\/(.*)'
        self.namespace = {"z": "https://github.com/zbmed-semtec/whatizit-dictionary-ner/"}

        self.trans_dict = self.create_dict(verify_integrity)

    def extract_mesh_id(self, tag: ET.Element) -> str:
        if tag.attrib.get("id"):
            mesh_id = "MeSH" + re.search(self.mesh_id_pattern, tag.attrib.get("id")).group(1)
        else:
            mesh_id = tag.text.strip()
        return mesh_id

    def create_dict(self, verify_integrity: bool = False) -> dict:
        trans_dict = {}

        for tagged in self.root.findall("document/passage/text/z:mesh", self.namespace):
            mesh_id = self.extract_mesh_id(tagged)
            if not tagged.text in trans_dict.keys():
                trans_dict[tagged.text] = mesh_id
            else:
                if verify_integrity and trans_dict[tagged.text] != mesh_id:
                    print("ERROR")
        return trans_dict

    def translate(self) -> None:
        self.tagged_texts = {}
        self.mod_text = {}

        for passage in self.root.findall("document/passage"):
            description = passage.find("infon").text.strip()
            text = passage.find("text")
            self.tagged_texts[description] = text

            local_text = []
            for concept in text.itertext():
                local_text.append(self.trans_dict.get(concept.strip(), concept))
            self.mod_text[description] = "".join(local_text).strip()

    def output_text(self, split: int = 0) -> str:
        if split == 0:
            out_text = " ".join(self.mod_text.values())
        else:
            out_text = ""
            for key, value in self.mod_text.items():
                out_text += (key + ": " + value + "\n")
        return out_text

    def save_output(self, file_out: str, split: int = 0) -> None:
        out_text = self.output_text(split)

        with open(file_out, "w+") as file:
            file.write(out_text)
    
    def preprocess_text(self) -> None:
        for key, value in self.mod_text.items():
            # To Lowercase
            value = value.lower()
            
            # Remove stopwords
            value = remove_stopwords(value)
            self.mod_text[key] = value
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", type = str, help="Path to input XML file")
    group.add_argument("-d", "--indir", type = str, help = "Path to input folder with XML files")
    parser.add_argument("-o", "--output", type = str, help = "Path to output text file/dir")

    args = parser.parse_args()

    # Code to extract the input files and output files considering all scenarios.
    # By default, it adds "_translated" to both output directories and files.
    if args.indir:
        file_list = [os.path.join(args.indir, file) for file in os.listdir(args.indir) if file.endswith("xml")]

        file_out = [file[:file.find(".xml")] + "_translated.xml" for file in os.listdir(args.indir) if file.endswith("xml")]
        directory = args.output if args.output else args.indir.strip("/") + "_translated_v2"
        file_out = [os.path.join(directory, file) for file in file_out]
        
        if not os.path.exists(directory) : os.mkdir(directory)
    else:
        file_list = [args.input]
        file_out = [args.output] if args.output else [args.input[:args.input.rfind(".xml")] + "_translated.xml"]

    for i, file in enumerate(file_list):
        print(file)
        xml_translation = XMLtrans(file)
        xml_translation.translate()
        xml_translation.save_output(file_out[i])