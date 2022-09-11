# File Requirements:

To begin the process, you need the XML annotated files that Rohitha uploaded to drive:

* [RELISH](https://drive.google.com/drive/u/0/folders/1qfHIWN2ncfboqtigF3DKOFiQFFYnTWI0)
* [TREC](https://drive.google.com/drive/u/0/folders/1wQ_ys557E3E3opQUuSWSPmtltyqLALm4)

In my home directory, I have a folder called `data_full`, and two subfolders called `RELISH` and `TREC`. I unzipped both datasets in their corresponding folder, so that I had the RELISH files in `~/data_full/RELISH/RELISH_annotated_xmls` and TREC in `~/data_full/TREC/TREC_annotated_xmls`.

I also deleted the file `26740972_annotated.xml` in RELISH and 8914767_annotated.xml in `TREC`, since they used to have an abstract, but it later runs they are marked as missing PMIDs. 

I also recommend that you have both the `hybrid-dictionary-ner-doc2vec-doc-relevance` and `medline-preprocessing` repositories downloaded to your home directory.

# Process

## XML Translation
First, you need to run the `XML_translation` script. To do so, either follow the [documentation](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/tree/main/docs/xml_translate) to execute step by step, or run the main script with the following commands:

```bash
python code/xml_translate/xml_translate.py --indir ../data_full/RELISH/RELISH_annotated_xmls --output ../data_full/RELISH/RELISH_documents_ann.tsv

python code/xml_translate/xml_translate.py --indir ../data_full/TREC/TREC_annotated_xmls --output ../data_full/TREC/TREC_documents_ann.tsv
```

Both commands are executed from the root directory of the hybrid-dictionary-ner-doc2vec-doc-relevance repository, located in `~/hybrid-dictionary-ner-doc2vec-doc-relevance/`.

## Preprocessing
Then, we need to remove the structure words. Again, you can follow the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/docs/Structure_Words_removal) from medline-preprocessing, or you can run the following commands:

```bash
python code/Structure_Words_removal/structurewords_remover.py --input ../data_full/RELISH/RELISH_documents_ann.tsv --output ../data_full/RELISH/RELISH_documents_ann_swr.tsv --list data/Structure_Words_removal/structure_word_list_pruned.txt 

python code/Structure_Words_removal/structurewords_remover.py --input ../data_full/TREC/TREC_documents_ann.tsv --output ../data_full/TREC/TREC_documents_ann_swr.tsv --list data/Structure_Words_removal/structure_word_list_pruned.txt 
```
These commands need to be run in the root directory of the medline-preprocessing repository.

The next step is to execute the usual preprocessing steps, common to all approaches. Instead of using the medline-preprocessing [Phrase_Preprocessing_Tutorial](https://github.com/zbmed-semtec/medline-preprocessing/blob/main/docs/Phrase_Preprocessing_Tutorial/tutorial_phrase_preprocessing.ipynb), I opted to create a more efficient preprocessing pipeline for this approach. I am not sure if this approach would work for other approaches, but it is much faster than using the `sciSpacy` tokenizer.

You can follow the [documentation](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/tree/main/docs/preprocessing) or run the following commands:

```bash
python code/preprocessing/preprocess.py --input ../data_full/RELISH/RELISH_documents_ann_swr.tsv --output ../data_full/RELISH/RELISH_tokens.tsv
	
python code/preprocessing/preprocess.py --input ../data_full/TREC/TREC_documents_ann_swr.tsv --output ../data_full/TREC/TREC_tokens.tsv
```

You can directly download these files from [the drive shared folder](https://drive.google.com/drive/u/0/folders/1QF_QzIrv-SqagckVultltR5X1nwsP2Fh).

## Doc2vec model generation and training

In this case, I do recommend you to follow the documentation, since it allows to modify the training parameters much easily than running the command. It can be found [here](https://github.com/zbmed-semtec/hybrid-dictionary-ner-doc2vec-doc-relevance/tree/main/docs/embeddings), or you can run the following commands:

```bash
python code/embeddings/create_model.py --input ../data_full/TREC/TREC_tokens.tsv --embeddings ../data_full/RELISH/RELISH_embeddings.pkl --output ../data_full/RELISH/RELISH_hybrid.model

python code/embeddings/create_model.py --input ../data_full/TREC/TREC_tokens.tsv --embeddings ../data_full/TREC/TREC_embeddings.pkl --output ../data_full/TREC/TREC_hybrid.model
```
If you want to save the model too, add the argument `--output [model path]`.

## Fill relevance matrix

You can also find a `fill_relevance_matrix.py` inside the `code/embeddings` folder. It is primarily used to fill a relevance matrix given a model. Another approach is the [script](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity) uploaded by Rohitha that uses the output embeddings, instead of the model.

Either way, you can run this script with the following command:

```bash
python code/embeddings/fill_relevance_matrix.py --input_rm ../data_full/RELISH/RELISH_relevance_matrix.tsv --input_model ../data_full/RELISH/RELISH_hybrid.model --output ../data_full/RELISH/RELISH_filled_relevance_matrix.tsv --verbose 1

python code/embeddings/fill_relevance_matrix.py --input_rm ../data_full/TREC/TREC_simplified_relevance_matrix.tsv --input_model ../data_full/TREC/TREC_hybrid.model --output ../data_full/TREC/TREC_simplified_filled_relevance_matrix.tsv --verbose 1

python code/embeddings/fill_relevance_matrix.py --input_rm ../data_full/TREC/TREC_repurposed_relevance_matrix.tsv --input_model ../data_full/TREC/TREC_hybrid.model --output ../data_full/TREC/TREC_repurposed_filled_relevance_matrix.tsv --verbose 1
```

In this file, I use multiprocessing to efficiently fill the relevance matrix. I recommend you to check it, since maybe it can also be implemented elsewhere, and it cut the execution time a reasonable amount (25 minutes to fill TREC relevance matrix to just 2 minutes).

```python
num_processes = mp.cpu_count()

# Split the relevance matrix into chunks
chunk_size = int(rel_matrix.shape[0]/num_processes)
chunks = [rel_matrix.iloc[rel_matrix.index[i:i + chunk_size]] for i in range(0, rel_matrix.shape[0], chunk_size)]

# Fill each chunk in parallel
with mp.Pool(num_processes) as p:
    results = p.starmap(fill_relevance_matrix, zip(chunks, repeat(model)))

# Combine the chunks
for i in range(len(results)):
    rel_matrix.at[results[i].index] = results[i]
```

# Additional Notes

* Every code file I have written is thought to be use both in a jupyter notebook (like in the documentations/tutorials), or as a script that can be run from the terminal itselft. Just type `python [file_name] --help` to obtain more information on how to use it.

* One thing you will notice from my code is that I really like to give options to the end user. My functions tend to have a lot of arguments to modify the behaviour to suit the user need. Unfortunately, that sometimes means that my functions are hard to understand to other coders. 

* Something that can be improved from my code are the error handling problems, where sometimes it is hard to understand where an error is.

If for whatever reason anyone need some help to understand, modify or use this code/repository, please don't hesitate to contact me via email at: guillermorocamora@gmail.com
