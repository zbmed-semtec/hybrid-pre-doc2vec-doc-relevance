import argparse
import re
import os

class XMLtrans:
    def __init__(self, input_file: str,) -> None:
        self.input_file = input_file
        
        self.text_pattern = r"<text>(.*?)</text>"
        self.mesh_pattern = r"<z:mesh.*?>(.*?)</z:mesh>"
        self.id_pattern = r'\/MESH\/(.*?)\"'    

        self.read_file()

    def read_file(self) -> None:
        """
        Reads the file input
        """
        self.raw_text = ""

        with open(self.input_file) as file:
            self.raw_text = "".join([line.strip() for line in file.read().splitlines()])

    def extract_mesh_id(self, tag: re.Match) -> str:
        """
        Translates the tag into its MeSH Id. If no MeSH Id is found, 
        it returns the tagged word without any modification.

        @type  tag: re.Match
        @param tag: the matched pattern.

        @rtype:     str
        @return:    the MeSH id or the word itself
        """
        mesh_id = re.search(self.id_pattern, tag.group(0))
        if(mesh_id):
            return("MeSH" + mesh_id.group(1))
        else:
            return(tag.group(1))

    def translate(self) -> None: 
        """
        Translates the XML file. It loops through the content between "text" tag
        in the XML file and extract both the tagged words and the untagged.
        
        The translated text is stored in "self.mod_text".
        """
        self.mod_text = ""

        text_splitted = re.findall(self.text_pattern, self.raw_text)
        for ind_text in text_splitted:
            tag_text = re.finditer(self.mesh_pattern, ind_text)

            last_end_pos = 0
            for tag in tag_text:
                self.mod_text += (ind_text[last_end_pos:tag.start()])
                self.mod_text += self.extract_mesh_id(tag)
                last_end_pos = tag.end()
            self.mod_text += (ind_text[last_end_pos:])
            self.mod_text += " "
        self.mod_text = self.mod_text.strip()

    def save_output(self, file_out: str) -> None:
        """
        Saves the translated file in the desired output file.

        @type   file_out: str
        @param  file_out: path for output text file
        """
        with open(file_out, "w+") as file:
            file.write(self.mod_text)

    def split_text(self) -> None:
        """
        Separates the whole text into title and abstract. Since the translation
        algorithm joins every text found, it uses the "BACKGROUND" keyword as the
        separator.
        """
        abstract_marker = "BACKGROUND: "

        self.title = self.mod_text[:self.mod_text.find(abstract_marker)]
        self.abstract = self.mod_text[self.mod_text.find(abstract_marker) + len(abstract_marker):]
        
        #conclusion_marker = "CONCLUSIONS: "
        #results_maker = "RESULTS: "
        #self.results = self.mod_text[self.mod_text.find(results_maker) + len(results_maker):self.mod_text.find(conclusion_marker)]
        #self.conclusions = self.mod_text[self.mod_text.find(conclusion_marker) + len(conclusion_marker):]


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
        directory = args.output if args.output else args.indir.strip("/") + "_translated"
        file_out = [os.path.join(directory, file) for file in file_out]
        
        if not os.path.exists(directory) : os.mkdir(directory)
    else:
        file_list = [args.input]
        file_out = [args.output] if args.output else [args.input[:args.input.rfind(".xml")] + "_translated.xml"]

    for i, file in enumerate(file_list):
        xml_translation = XMLtrans(file)
        xml_translation.translate()
        xml_translation.save_output(file_out[i])