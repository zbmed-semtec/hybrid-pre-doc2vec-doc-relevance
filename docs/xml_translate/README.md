# Annotated XML files to plain text

This tutorial aims to explain how to translate from annotated XML files obtained from [Whatizit](https://github.com/zbmed-semtec/whatizit-dictionary-ner) to plain text. The translation converts the tagged concepts by the dictionary to its corresponding MeSH ID.

# Prerequisites

1. Annotated XML files. On how to obtain them, please follow instructions [here](https://github.com/zbmed-semtec/whatizit-dictionary-ner/tree/main/docs).

# Steps

## Step 1:

Import the necessary class to process the files:


```python
import sys
sys.path.append('../../code/xml_translate/')

from xml_translate import XMLtrans, translate_pipeline
```

## Step 2:

Define the input and output paths. Creates the output directory if needed. We will also export the articles to a .TSV file, so indicate the path as well.


```python
import os
input_path = "../../data/sample_annotated_xml"
output_path = "../../data/sample_annotated_xml_translated"
output_tsv = "../../data/sample_annotated_xml.tsv"

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
    xml_translation.save_xml(output_files[i])
```

To export every XML file to a single TSV, run the pipeline function:


```python
translate_pipeline(input_files, output_tsv)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PMID</th>
      <th>title</th>
      <th>abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18394048</td>
      <td>Effect of MeSHD000077287 on MeSHD000071080: a ...</td>
      <td>BACKGROUND AND PURPOSE: We studied the effect ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18363035</td>
      <td>A MeSHD016678-wide MeSHD046228 reveals anti-in...</td>
      <td>OBJECTIVE: Paeony root has long been used for ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18366698</td>
      <td>The MeSHQ000706 of biomedicine, complementary ...</td>
      <td>BACKGROUND: Studies have shown that a signific...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10627593</td>
      <td>Presynaptic Ca(2+) influx at a MeSHD051379 cen...</td>
      <td>Genetic alterations in Ca(2+) channel subunits...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10642555</td>
      <td>Rad6-dependent MeSHD054875 of MeSHD006657 H2B ...</td>
      <td>Although ubiquitinated MeSHD006657 are present...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10609551</td>
      <td>Individual variation in the expression profile...</td>
      <td>MeSHD009538 evokes dose-dependent and often va...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>18401010</td>
      <td>MeSHD006224 MeSHD013075: MeSHD012380 of MeSHD0...</td>
      <td>Recently, we demonstrated that MeSHD019289 MeS...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>18370348</td>
      <td>[MeSHD009369 search with MeSHD004058-weighted ...</td>
      <td>PURPOSE: Assessment of fat-suppressing MeSHD00...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10605405</td>
      <td>Ki-67, oestrogen receptor, and MeSHD011980 MeS...</td>
      <td>AIM: To examine proliferative activity using t...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10629618</td>
      <td>MeSHD006360 associated with MeSHD004260 MeSHD0...</td>
      <td>Two MeSHD004798 of MeSHD004260 (BER), MeSHD051...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>18369322</td>
      <td>Absence of retroviral vector-mediated transfor...</td>
      <td>There is considerable concern regarding the tr...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10624815</td>
      <td>MeSHD001053 E polymorphisms and MeSHD010300.</td>
      <td>The MeSHD001053 E (APOE) MeSHD011110 has been ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10650943</td>
      <td>Potential involvement of FRS2 in MeSHD007328 s...</td>
      <td>Shp-2 is implicated in several MeSHD020794 sig...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>18393066</td>
      <td>Producing MeSHD056804 using whole-MeSHD002477 ...</td>
      <td>This MeSHD012106 examined the effect of using ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>18371110</td>
      <td>MeSHD001853 trephine MeSHQ000033 and immunohis...</td>
      <td>Chronic myelomonocytic leukaemia (CMML) is a c...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10648631</td>
      <td>Mutational MeSHQ000032 of the highly conserved...</td>
      <td>We have suggested previously that both the neg...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10606228</td>
      <td>Spontaneous and MeSHD009153-induced transforma...</td>
      <td>Loss of MeSHQ000502 of MeSHD053843 (MMR) MeSHD...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10671540</td>
      <td>The regulatory beta subunit of MeSHD047390 med...</td>
      <td>MeSHD047390 is a tetrameric MeSHD004798 compos...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18366613</td>
      <td>A MeSHD002678 MeSHD000069550-based MeSHD008722...</td>
      <td>BACKGROUND: The MeSHD011379 for many MeSHD0093...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>18381570</td>
      <td>BRAFV600E MeSHD009154 is associated with prefe...</td>
      <td>CONTEXT: Mutually exclusive MeSHD009154 of RET...</td>
    </tr>
  </tbody>
</table>
</div>



# Other options

## Translate specific file

In order to translate one file, just create the `XMLtrans` object with the input file path, execute the required methods, and use the `save_xml` method with the output file path as a parameter. It is also possible to export to a plain text file by executing the `save_txt` method.


```python
%%script false --no-raise-error
input_file = "../../data/sample_annotated_xml/10605405_annotated.xml"
output_file = "../../data/sample_annotated_xml/10605405_annotated_translated.xml"

xml_translation = XMLtrans(input_file)
xml_translation.translate()
xml_translation.save_xml(output_file)
xml_translation.save_txt(output_file, split = True) # The split argument indicates whether to differentiate title and abstract.
```

## Use `xml_translatepython.py` file directly

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

If you wish to output to a .TSV file, set the following argument to 1:

* --tsv: default 0

An example of the command that will create the directory  `sample_annotated_xml_translated`:

```bash
python xml_translate.py -d sample_annotated_xml
```

In the folder `code/tests`, a file containing unittest can be found.

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

* I use the `xml.etree` library as it is one of the most used XML parsers and provided all the functionality required. I also tried using regular expression directly on the text itself, but it proved to be less efficient and harder to handle errors.

* Why to create a dictionary? Mainly for two reasons:

    1. To validate the annotated XML files, since every "concept" should have the same MeSH ID. By having a dictionary, I can ensure that the `z:mesh` tags are consistent with each other. 
    
    2. To facilitate the text translation. To avoid using regular expressions in the text to find the tags, I combined the functionality of the method `itertext()` from `xml.etree` and the dictionary to find a less prone to error solutions.

* The code does not assume the number of `passage` tags found in them, but it assumes that they all have an `infon` tag (to describe the passage) and a `text` tag which contains the text itself. 

# TODO 

* Improve error handling process. At the moment, if the XML file does not match the necessary structure, there is no indication to where the error is or how to deal with it.

* Consider adding multiprocessing. There are around 200.000 documents, which takes around a minute and a half to execute. Multiprocessing functionality could be useful since the translation process is done one file at a time, and it should be easily parallelizable.

**REMOVE THIS LINE BEFORE FINAL VERSION COMMIT**


```python
!jupyter nbconvert tutorial_xml_translate.ipynb --to markdown --output README.md
```

    [NbConvertApp] Converting notebook tutorial_xml_translate.ipynb to markdown
    [NbConvertApp] Writing 13102 bytes to README.md

