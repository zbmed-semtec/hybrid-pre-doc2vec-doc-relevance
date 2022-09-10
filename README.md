# hybrid-dictionary-ner-doc2vec-doc-relevance

An approach exploring the combination of a dictionary-based NER approach using Whatizit and Doc2Vec to develop a document-to-document recommendation system. This approach will explore the transformation of annotated XML files after the Whatizit processing into a plain text dataset, the required preprocessing of the text and the Doc2Vec model generation, training and evaluation.

The main idea is to use a dictionary-based name entity recognition to group different medical terms in one individual term. To do so, we use the Whatizit text processing system developed by the Rebholz Research Group at [EMBL-EBI](https://www.ebi.ac.uk/). Once the terms are annotated, we replace these  
annotations by their MeSH ID. For example, the dictionary will recognize every entry of the term "mechanism of action" all will make add an XML tag around it. Our pipeline will then replace the whole tag by its MeSH ID, specifically [MeSHQ000494](http://purl.bioontology.org/ontology/MESH/Q000494).

Once we have the title and abstract of a publication with their medical terms substituted by their MeSH ID, we apply a standard Doc2Vec process to generate embeddings. Then, using the cosine similarity between two different publications, we will rank their similarity from 0 to 1.

## Input data

On what is RELISH and TREC, please refer to our [medline-preprocessing repository](https://github.com/zbmed-semtec/medline-preprocessing), where the data explanation and extraction is described. 

For the hybrid approach, the input data are annotated XML files generated using Whatizit/monq. The process can be found in the [repository documentation](https://github.com/zbmed-semtec/whatizit-dictionary-ner/tree/main/docs).

This is an example of the annotated XML files we need:

```XML
<?xml version='1.0' encoding='utf-8'?>
<collection xmlns:z="https://github.com/zbmed-semtec/whatizit-dictionary-ner#">
    <document>
        <id>
            10606228
        </id>
        <passage>
            <infon key="type">
                title
            </infon>
            <offset>
                0
            </offset>
            <text>
                Spontaneous and <z:mesh cui="C0008904, C0026879" id="http://purl.bioontology.org/ontology/MESH/D009153" semantics="http://purl.bioontology.org/ontology/STY/T131">mutagen</z:mesh>-induced transformation of primary <z:mesh cui="C0243103, C0015033, C0220814" id="http://purl.bioontology.org/ontology/MESH/Q000208" semantics="http://purl.bioontology.org/ontology/STY/T080, http://purl.bioontology.org/ontology/STY/T169">cultures</z:mesh> of Msh2-/- p53-/- colonocytes.
            </text>
        </passage>
        <passage>
            <infon key="type">
                abstract
            </infon>
            <offset>
                98
            </offset>
            <text>
                Publication abstract removed for simplicity.
            </text>
        </passage>
    </document>
</collection>
```

# Process

Here, we describe the main process over both the RELISH and TREC datasets.

## XML translation

[Main documentation](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/tree/main/docs/xml_translate).

The goal is to convert the annotations in the XML files into a MeSH ID. As mentioned before, the input data are annotated XML files.

1. Loop throught every XML file inside the given folder. Optionally, a single file can also be provided. No requisites for the XML file will be specified, since they are expected to be generated using the Whatizit NER approach.

2. Create a `translation_dictionary`, where every term is associated to a MeSH ID.

3. Replace ocurrence of a term inside the `translation_dictionary` by their MeSH ID.

4. Store all the translated XML files into a single TSV file with three columns: PMID, title, abstract. Optionally, the files can be individually saved into a TXT.

At the end of this process, we expect to have a TSV file for each dataset (RELISH and TREC) with the already mentioned three columns and the medical terms identified by the Whatizit dictionary replaced by their MeSH ID.

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
  </tbody>
</table>
</div>

## Data preprocessing

The next step is to apply some preprocessing to the publications. This process is split in two different steps:

### Structure words removal

[Main documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Structure_Words_removal).

Both the code and the documentation is located in the [medline-preprocessing repository](https://github.com/zbmed-semtec/medline-preprocessing) since it is a common step in every document-to-document similarity approach. We defined as "Structure words" those terms that are introduced in the text to better structure the abstract. In the table shown above, we can see that for example the words "BACKGROUND: " and "OBJECTIVE: " are written at the start of the abstract. These terms do not provided any meaningful information and are not found in every publication and, since in the end we try to measure the similarity between documents, they can artificially increase the similarity of two publications that are not related otherwise.

These structure words usually follow a pattern: most of them are in capital letters (not always), they all end with a colon and an empty space ": " and they are always located at the beginning of a sentence. Using these rules, we developed an algorithm to identify and eliminate them.

The main idea is:

1. Loop throught every publication title and abstract and match a regular expression to find the structure words. At the same time, count in how many publications each structure word appears.

2. Since we are applying a regular expression, some false positives might be found (certain acronyms for example). Since we don't want to remove relevant terms that may follow the regular expression, we apply a minimum frequency of appearance threshold. The standard is to require a matched structure word to appear in at least 0.01% of all publications. We create a [list of these structure words](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/data/Structure_Words_removal/structure_word_list_pruned.txt). It possible to modify the minimum frequency of appearance by applying a different threshold. More information can be found in the corresponding documentation.

3. Remove every structure word found in the list.

Example of structure words removal:

<table>
<tr>
<th>Abstract Input</th>
<th>Abstract Output</th>
</tr>
<tr>
<td width="50%">
<mark>OBJECTIVE: </mark>To describe the development of evidence-based electronic prescribing (e-prescribing) triggers and treatment algorithms for potentially inappropriate medications (PIMs) for older adults. <mark>DESIGN: </mark>Literature review, expert panel and focus group. <mark>SETTING: </mark>Primary care with access to e-prescribing systems. <mark>PARTICIPANTS: </mark>Primary care physicians using e-prescribing systems receiving medication history. <mark>INTERVENTIONS: </mark>Standardised treatment algorithms for clinicians attempting to prescribe PIMs for older patients. <mark>MAIN OUTCOME MEASURE: </mark>Development of 15 treatment algorithms suggesting alternative therapies. <mark>RESULTS: </mark>Evidence-based treatment algorithms were well received by primary care physicians. Providing alternatives to PIMs would make it easier for physicians to change decisions at the point of prescribing. <mark>CONCLUSION: </mark>Prospectively identifying older persons receiving PIMs or with adherence issues and providing feasible interventions may prevent adverse drug events.
</td>
<td width="50%">
To describe the development of evidence-based electronic prescribing (e-prescribing) triggers and treatment algorithms for potentially inappropriate medications (PIMs) for older adults. Literature review, expert panel and focus group. Primary care with access to e-prescribing systems. Primary care physicians using e-prescribing systems receiving medication history. Standardised treatment algorithms for clinicians attempting to prescribe PIMs for older patients. Development of 15 treatment algorithms suggesting alternative therapies. Evidence-based treatment algorithms were well received by primary care physicians. Providing alternatives to PIMs would make it easier for physicians to change decisions at the point of prescribing. Prospectively identifying older persons receiving PIMs or with adherence issues and providing feasible interventions may prevent adverse drug events.
</td>
</tr>
</table>

### Text cleaning and tokenization

The cleaning process is also described in the [medline-preprocessing repository](https://github.com/zbmed-semtec/medline-preprocessing) ([main documentation](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/docs/Phrase_Preprocessing_Tutorial/tutorial_phrase_preprocessing.ipynb)). However, a specific preprocessing was develop for the hybrid approach ([specific documentation](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/tree/main/docs/preprocessing)) since it does not have the same requirements as other approaches.

The process consists in:

1. Lower case everything and split by white spaces.

2. Remove every non-alphanumeric character with the exception of the hyphen `-`. The allowed characters can be modified following the documentation.

3. Store the results in a TSV file with the same three columns as before.

An example of the output is:

<table>
<tr>
<th>Abstract Input</th>
<th>Abstract Output</th>
</tr>
<tr>
<td width="50%">
Despite the high MeSHQ000453 of MeSHD010300, the MeSHQ000503 of its gastrointestinal MeSHQ000175 remains poorly understood. to evaluate MeSHD003679 and defecatory MeSHQ000502 in MeSHD010361 with MeSHD010300 and age- and MeSHD012723-matched MeSHQ000517 and to correlate objective MeSHQ000175 with subjective MeSHQ000175.
</td>
<td width="50%">
despite the high meshq000453 of meshd010300 the meshq000503 of its gastrointestinal meshq000175 remains poorly understood to evaluate meshd003679 and defecatory meshq000502 in meshd010361 with meshd010300 and age- and meshd012723-matched meshq000517 and to correlate objective meshq000175 with subjective meshq000175
</td>
</tr>
</table>

## Doc2Vec model generation

[Main documentation](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/tree/main/docs/embeddings)

With the text cleaned, we can now start to generate the embeddings. To do so, we will the [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) framework provided by [Gensim](https://radimrehurek.com/gensim/). The process can be described as follows:

1. Generate the `TaggedDocuments` to match every PMID to a list of words. In this step, the abstract and the title are combined into a single paragraph (or document).

2. Generate the model with the desired hyperparameters (more information in the following section).

3. Train the model.

4. Extract the embeddings of the training publications or generate them for new publications.

## Hyperparameter search and model evaluation

One of the most important steps in every machine learning development is to properly choose a set of hyperparameters. The Doc2vec hyperparamters we will consider for this research are the following:

* The training algorithm **dm**. Refers to either using distributed memory or distributed bag of words.
* The **vector_size** of the genrated embeddings. It represents the number of dimensions our embeddings will have.
* The **window** size represents the maximum distance between the current and predicted word.
* The number of **epochs** or iterations of the training dataset.
* The minimum nombre of appearances a word must have to not be ignored by the algorithm is specified with the **min_count** parameter.

The most relevant aspect when trying to optimize for hyperparameters is to choose an appropiate evaluation method. In this research, two different approaches were considered:

* ROC One vs All approach: if we understand our model as a non-relevant vs relevant classifier, we can use the area under the ROC curve (AUC) as a model quality estimator. More details can be found in the [corresponding documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Distribution_Analysis/hp_optimization).

* Precision and nDCG:  **Work In Progress**.

# Results

**Work In Progress**