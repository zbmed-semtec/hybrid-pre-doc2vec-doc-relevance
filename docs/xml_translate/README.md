# Annotated XML files to plain text

This tutorial aims to explain how to translate from annotated XML files obtained from [Whatizit](https://github.com/zbmed-semtec/whatizit-dictionary-ner) to plain text. The translation converts the tagged concepts by the dictionary to its corresponding MeSH ID.

# Prerequisites

1. Annotated XML files. On how to obtain them, please follow instructuions [here](https://github.com/zbmed-semtec/whatizit-dictionary-ner/tree/main/docs).

# Steps

## Step 1:

Import the necessary class to process the files:


```python
import sys
sys.path.append('../../code/xml_translate/')

from xml_translate import XMLtrans
```

## Step 2:

Define the input and output paths. Creates the output directory if needed:


```python
import os
input_path = "../../data/sample_annotated_xml"
output_path = "../../data/sample_annotated_xml_translated"

if not os.path.isdir(output_path): os.mkdir(output_path)

```

Given the directory path, two lists containing the input and output files are created.


```python
import glob

input_files = glob.glob(input_path + "/*.xml")
output_files = list(map(lambda x: x.replace(input_path, output_path), input_files))
```

## Step 3:

Loop through the input files. For each input, it creates a `XMLtrans` object and executes the `translate` and `save_output` methods.


```python
for i, file in enumerate(input_files):
    xml_translation = XMLtrans(file)
    xml_translation.translate()
    xml_translation.save_output(output_files[i], split = True)
```

# Other options

## Translate specific file

In order to translate one file, just create the `XMLtrans` object with the input file path, execute the required methods, and use the `save_output` method with the output file path as a parameter.


```python
%%script false --no-raise-error
input_file = "../../data/sample_annotated_xml/10605405_annotated.xml"
output_file = "../../data/sample_annotated_xml/10605405_annotated_translated.xml"

xml_translation = XMLtrans(input_file)
xml_translation.translate()
xml_translation.save_output(output_file)
```

## Use `xml_trasnlate.py` file directly

The code file itself is prepared to work on their own given some parameters. In order to execute the script, run the following command:

```bash
python xml_translate.py [-h] (-i INPUT | -d INDIR) [-o OUTPUT]
```

You must pass one of the following arguments:

* -i / --input: path to XML file to be translated.

* -d / --indir: path to directory containing the XML files to be translated.

Optionally, the output file/directory argument can be used to specify where should the translated files be stored. By default:

* If only one file is provided, it will be saved in the same directory with the text `_translated` added to it.

* If a directory is provided, it will create a new directory with the same name but adding `_translated` to it.

An example of the command that will create the direcortory `sample_annotated_xml_translated`:

```bash
python xml_translate.py -d sample_annotated_xml
```

In `code/tests` folder, a file containing unittest can be found.

# Results

Example of translation:

<table>
<tr>
<th>XML Input</th>
<th>Output file</th>
</tr>
<tr>
<td>

```xml
<?xml version='1.0' encoding='utf-8'?>
<collection xmlns:z="https://github.com/zbmed-semtec/whatizit-dictionary-ner/">
<document>
   <id>
      18366698
   </id>
   <passage>
      <infon key="type">
         title
      </infon>
      <offset>
         0
      </offset>
      <text>
         The <z:mesh cui="C0243174, C0042153, C0038209"
         id="http://purl.bioontology.org/ontology/MESH/Q000706"
         semantics="http://purl.bioontology.org/ontology/STY/T169,
         http://purl.bioontology.org/ontology/STY/T081">use</z:mesh>
         of biomedicine, complementary and <z:mesh
         cui="C1148474, C0949216, C0002346, C0936077"
         id="http://purl.bioontology.org/ontology/MESH/D000529"
         semantics="http://purl.bioontology.org/ontology/STY/T091,
         http://purl.bioontology.org/ontology/STY/T061">alternative
         medicine</z:mesh>, and <z:mesh cui="C0025130,
         C0242390, C0025127, C0025131, C0016419, C0015034,
         C0242389"
         id="http://purl.bioontology.org/ontology/MESH/D008519"
         semantics="http://purl.bioontology.org/ontology/STY/T058,
         http://purl.bioontology.org/ontology/STY/T091,
         http://purl.bioontology.org/ontology/STY/T061">ethnomedicine</z:mesh>
         for the <z:mesh cui="C0039798"
         id="http://purl.bioontology.org/ontology/MESH/Q000628"
         semantics="http://purl.bioontology.org/ontology/STY/T169">treatment</z:mesh>
         of <z:mesh cui="C0014544, C0751111, C0236018,
         C0086237"
         id="http://purl.bioontology.org/ontology/MESH/D004827"
         semantics="http://purl.bioontology.org/ontology/STY/T047,
         http://purl.bioontology.org/ontology/STY/T033">epilepsy</z:mesh>
         among people of South <z:mesh cui="C0078988,
         C1553322, C1553323, C1556095, C1257906, C1561452,
         C0152035, C0337910, C0337892, C1556094"
         id="http://purl.bioontology.org/ontology/MESH/D044466"
         semantics="http://purl.bioontology.org/ontology/STY/T098">Asian</z:mesh>
         origin in the UK.
      </text>
   </passage>
</document>
</collection>
```

</td>
<td width="50%">
The MeSHQ000706 of biomedicine, complementary and MeSHD000529, and MeSHD008519 for the MeSHQ000628 of MeSHD004827 among people of South MeSHD044466 origin in the UK.
</td>
</tr>
</table>

# Decision notes

## Code strategy:

1. Using `xml.etree` library, parse the input XML file to an object of that library.

2. Find all the `z:mesh` tags inside the document and creates a dictionary that matches the tagged word to its MeSH ID (extracted from the tag using regular expressions). The objective is to have a dictionary to later translate all read texts.

3. Find all `passage` tags in the document. For each tag:

    3.1 Read the description found in `infon` tag.
    
    3.2 Find the tag `text` and iter through every text found in them. It checks if the text is in the dictionary previously created and, if matched, replace it by the MeSH ID.

    3.3 Creates a named list with the description as key and the translated text as value.
    
4. Joins all the translated `passage` found and saves them in plain text. It can also be separated by the descriptions found. 

## Decisions:

* I use the `xml.etree` library as it is one the most used XML parsers and provided all the functionality required. I also tried using regular expression directly on the text itself, but it proved to be less efficient and harder to handle errors.

* Why to create a dictionary? Mainly for two reasons:

    1. To validate the annotated XML files since every "concept" should have the same MeSH ID. By having a dictionary, I can ensure that the `z:mesh` tags are consistent with each other. 
    
    2. To facilitate the text translation. To avoid using regular expressions in the text to find the tags, I combined the functionality of the method `itertext()` from `xml.etree` and the dictionary to find a less prone to error solutions.

* The code does not assume the number of `passage` tags found in them, but it assumes that they all have an `infon` tag (to describe the passage) and a `text` tag which contains the text itself. 

# TODO 

* Improve error handling process. At the moment, if the XML file does not match the necessary structure, there is no indication to where the error is or how to deal with it.

* Consider adding multiprocessing. There are around 200.000 documents, which takes around a minute and a half to execute. Multiprocessing functionality could be useful since the translation process is done one file at a time and it should be easily parallelizable.

**REMOVE THIS LINE BEFORE FINAL VERSION COMMIT**


```python
!jupyter nbconvert tutorial_xml_translate.ipynb --to markdown --output README.md
```

    [NbConvertApp] Converting notebook tutorial_xml_translate.ipynb to markdown
    [NbConvertApp] Writing 16672 bytes to README.md

