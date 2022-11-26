# Evaluation of the hybrid-dictionary-ner-doc2vec-doc-relevance with the distribution-based approach

The following tables show the results of the [distribution-based](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Distribution_Analysis) evaluation approach, when applied on to "hybrid-dictionary-ner-doc2vec-doc-relevance" technique. 
These results are calculated for the different hyper-parameters settings of the Doc2Vec approach to obtain the optimal combination for each dataset used in this work.

Each of the below tables contains seven columns. The first six of these columns represent the hyper-parameters of the Doc2Vec model:
- **dm:** Defines the training algorithm that is used. If dm=0, 'distributed bag of words' (PV-DBOW) is used, else if dm=1, 'distributed memory' (PV-DM) is used.
- **epochs:** Defines the number of iterations over the corpus.
- **min_count:** Ignores those words that have the total frequency less than this number.
- **vector_size:** Defines the dimensionality of the feature vector.
- **window:** Defines the maximum distance between the current and predicted word within a sentence.
- **workers:** Defines the working threads used to train the model.
- **AUC:** Defines the "area under the curve", as a result of this evaluation approach.

**RELISH:** The table below shows the results from the distribution-based approach for the RELISH dataset.

| dm    | epochs      | min_count     | vector_size | window  | workers | AUC    |
|:-----:|:-----------:|:-------------:|:-----------:|:-------:|:-------:|:------:|
| 0     | 15          | 5             | 200         | 5       | 8       | 0.5794 |
| 0     | 15          | 5             | 200         | 6       | 8       | 0.581  |
| 0     | 15          | 5             | 200         | 7       | 8       | 0.5798 |
| 0     | 15          | 5             | 300         | 5       | 8       | 0.5813 |
| 0     | 15          | 5             | 300         | 6       | 8       | 0.5799 |
| 0     | 15          | 5             | 300         | 7       | 8       | 0.5813 |
| 0     | 15          | 5             | 400         | 5       | 8       | 0.5808 |
| 0     | 15          | 5             | 400         | 6       | 8       | 0.5818 |
| 0     | 15          | 5             | 400         | 7       | 8       | 0.5818 |
| 1     | 15          | 5             | 200         | 5       | 8       | 0.5975 |
| 1     | 15          | 5             | 200         | 6       | 8       | 0.599  |
| 1     | 15          | 5             | 200         | 7       | 8       | 0.5959 |
| 1     | 15          | 5             | 300         | 5       | 8       | 0.5968 |
| 1     | 15          | 5             | 300         | 6       | 8       | 0.5954 |
| 1     | 15          | 5             | 300         | 7       | 8       | 0.594  |
| 1     | 15          | 5             | 400         | 5       | 8       | 0.5938 |
| 1     | 15          | 5             | 400         | 6       | 8       | 0.5947 |
| 1     | 15          | 5             | 400         | 7       | 8       | 0.5933 |

**TREC-simplified:** The table below shows the results from the distribution-based approach for the "TREC-simplified" variant of the TREC dataset.

| dm    | epochs      | min_count     | vector_size | window  | workers | AUC    |
|:-----:|:-----------:|:-------------:|:-----------:|:-------:|:-------:|:------:|
| 0     | 15          | 5             | 200         | 5       | 8       | 0.6529 |
| 0     | 15          | 5             | 200         | 6       | 8       | 0.6531 |
| 0     | 15          | 5             | 200         | 7       | 8       | 0.6531 |
| 0     | 15          | 5             | 300         | 5       | 8       | 0.6532 |
| 0     | 15          | 5             | 300         | 6       | 8       | 0.653  |
| 0     | 15          | 5             | 300         | 7       | 8       | 0.6528 |
| 0     | 15          | 5             | 400         | 5       | 8       | 0.6526 |
| 0     | 15          | 5             | 400         | 6       | 8       | 0.6533 |
| 0     | 15          | 5             | 400         | 7       | 8       | 0.6524 |
| 1     | 15          | 5             | 200         | 5       | 8       | 0.6403 |
| 1     | 15          | 5             | 200         | 6       | 8       | 0.6383 |
| 1     | 15          | 5             | 200         | 7       | 8       | 0.6359 |
| 1     | 15          | 5             | 300         | 5       | 8       | 0.6395 |
| 1     | 15          | 5             | 300         | 6       | 8       | 0.639  |
| 1     | 15          | 5             | 300         | 7       | 8       | 0.6368 |
| 1     | 15          | 5             | 400         | 5       | 8       | 0.64   |
| 1     | 15          | 5             | 400         | 6       | 8       | 0.6385 |
| 1     | 15          | 5             | 400         | 7       | 8       | 0.6372 |

**TREC-repurposed:** The table below shows the results from the distribution-based approach for the "TREC-repurposed" variant of the TREC dataset.

| dm    | epochs      | min_count     | vector_size | window  | workers | AUC    |
|:-----:|:-----------:|:-------------:|:-----------:|:-------:|:-------:|:------:|
| 0     | 15          | 5             | 200         | 5       | 8       | 0.7716 |
| 0     | 15          | 5             | 200         | 6       | 8       | 0.7717 |
| 0     | 15          | 5             | 200         | 7       | 8       | 0.7715 |
| 0     | 15          | 5             | 300         | 5       | 8       | 0.7716 |
| 0     | 15          | 5             | 300         | 6       | 8       | 0.7726 |
| 0     | 15          | 5             | 300         | 7       | 8       | 0.771  |
| 0     | 15          | 5             | 400         | 5       | 8       | 0.7718 |
| 0     | 15          | 5             | 400         | 6       | 8       | 0.7716 |
| 0     | 15          | 5             | 400         | 7       | 8       | 0.7703 |
| 1     | 15          | 5             | 200         | 5       | 8       | 0.7433 |
| 1     | 15          | 5             | 200         | 6       | 8       | 0.7424 |
| 1     | 15          | 5             | 200         | 7       | 8       | 0.7392 |
| 1     | 15          | 5             | 300         | 5       | 8       | 0.743  |
| 1     | 15          | 5             | 300         | 6       | 8       | 0.7399 |
| 1     | 15          | 5             | 300         | 7       | 8       | 0.7384 |
| 1     | 15          | 5             | 400         | 5       | 8       | 0.7428 |
| 1     | 15          | 5             | 400         | 6       | 8       | 0.7418 |
| 1     | 15          | 5             | 400         | 7       | 8       | 0.738  |

