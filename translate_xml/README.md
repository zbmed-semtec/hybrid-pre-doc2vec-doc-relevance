# XML Translation tool

## What it does

It transforms an annotated XML file produced by [link to **[whatizit-dictionary-ner](https://github.com/zbmed-semtec/whatizit-dictionary-ner)** github] into plain text, where the words tagged by the dictionary are substituted (or "translated") to a corresponding MeSH Id.

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
<collection>
 <document>
  <id>
   19694997
  </id>
  <passage>
   <infon key="type">
    title
   </infon>
   <offset>
    0
   </offset>
   <text> 
<z:mesh cui="C0935625, C0344335, C0001527" 
id="http://purl.bioontology.org/ontology/MESH/D000273" 
semantics="http://purl.bioontology.org/ontology/STY/T201, 
http://purl.bioontology.org/ontology/STY/T024">Adipose tissue
</z:mesh> <z:mesh cui="C1640380, C4551909, C3494180, C4552764, 
C3178845, C4704950, C4704951, C4704952, C1257975, C3178844" 
id="http://purl.bioontology.org/ontology/MESH/D059630" 
semantics="http://purl.bioontology.org/ontology/STY/T025">
mesenchymal stem cell</z:mesh> expansion in 
<z:mesh cui="" id="http://purl.bioontology.org/ontology/STY/T008" 
semantics="">animal</z:mesh> <z:mesh cui="C0229671" 
id="http://purl.bioontology.org/ontology/MESH/D044967" 
semantics="http://purl.bioontology.org/ontology/STY/T031">
serum</z:mesh>-free medium supplemented with autologous 
<z:mesh cui="C0086418" 
id="http://purl.bioontology.org/ontology/MESH/D006801" 
semantics="http://purl.bioontology.org/ontology/STY/T016">human
</z:mesh> <z:mesh cui="C0005821" 
id="http://purl.bioontology.org/ontology/MESH/D001792" 
semantics="http://purl.bioontology.org/ontology/STY/T025">platelet
</z:mesh> lysate.
   </text>
  </passage>
 </document>
</collection>
```

</td>
<td width="35%">
MeSHD000273 MeSHD059630 expansion in animal MeSHD044967-free medium supplemented with autologous MeSHD006801 MeSHD001792 lysate.
</td>
</tr>
</table>

## How to use

In order to execute the script, run the following command:

```bash
python xml_translate.py [-h] (-i INPUT | -d INDIR) [-o OUTPUT]
```

You must pass one of the following arguments:

* -i / --input: path to XML file to be translated.

* -d / --indir: path to directory containing the XML files to be translated.

Optionally, the output file/directory argument can be used to specify where should the translated files be saved. By default:

* If only one file is provided, it will be saved in the same directory with the text `_translated` added to it.

* If a directory is provided, it will create a new directory with the same name but adding `_translated` to both the directory and the files within.

An example of the command that will create the direcortory `sample_annotated_xml_translated`:

```bash
python xml_translate.py -d sample_annotated_xml
```

The file `xml_translation_test.py` contains unittest to ensure that the translation work as expected.
