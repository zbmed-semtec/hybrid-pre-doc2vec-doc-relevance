# XML Translation tool

## What it does

This file aims to transform an annotated XML files obtained from [Whatizit](https://github.com/zbmed-semtec/whatizit-dictionary-ner) to plain text. The translation converts the tagged concepts by the dictionary to its corresponding MeSH ID.

A more detailed tutorial can be found in [`docs/xml_translate`](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/blob/main/docs/xml_translate/tutorial_xml_translate.ipynb)

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

## How to use

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
python xml_translate.py --indir ../../data/sample_annotated_xml
```

In `code/tests` folder, a file containing unittest can be found.
