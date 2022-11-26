# Evaluation of the hybrid-dictionary-ner-doc2vec-doc-relevance using nDCG@N approach

The following tables show the results of the [nDCG@N](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation) evaluation approach, when applied on to "hybrid-dictionary-ner-doc2vec-doc-relevance" technique. 
These results are calculated for the different hyper-parameters settings of the Doc2Vec approach to obtain the optimal combination for each dataset used in this work.

Each of the below tables contains seven columns. The first six of these columns represent the hyper-parameters of the Doc2Vec model. The remaining six columns represent the average nDCG scores at different values of N:
- **dm:** Defines the training algorithm that is used. If dm=0, 'distributed bag of words' (PV-DBOW) is used, else if dm=1, 'distributed memory' (PV-DM) is used.
- **epochs:** Defines the number of iterations over the corpus.
- **min_count:** Ignores those words that have the total frequency less than this number.
- **vector_size:** Defines the dimensionality of the feature vector.
- **window:** Defines the maximum distance between the current and predicted word within a sentence.
- **workers:** Defines the working threads used to train the model.
- **nDCG@5 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 5 articles retrieved.
- **nDCG@10 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 10 articles retrieved.
- **nDCG@15 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 15 articles retrieved.
- **nDCG@20 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 20 articles retrieved.
- **nDCG@25 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 25 articles retrieved.
- **nDCG@50 (AVG):** Normalized Discounted Cumulative Gain (nDCG) score for the top 50 articles retrieved.

**RELISH:** The table below shows the results from the nDCG@N approach for the RELISH dataset.

| dm  | epochs  | min_count  | vector_size | window  | workers | nDCG@5 (AVG) | nDCG@10 (AVG) | nDCG@15 (AVG) | nDCG@20 (AVG) | nDCG@25 (AVG) | nDCG@50 (AVG) |
|:---:|:-------:|:----------:|:-----------:|:-------:|:-------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| 0   | 15      | 5          | 200         | 5       | 8       | 0.658        |	0.6463        |	0.6497        |	0.6596        |	0.6752        |	0.7865        |
| 0   | 15      | 5          | 200         | 6       | 8       | 0.6545       |	0.6432        |	0.6473        |	0.6579        |	0.6731        |	0.7855        |
| 0   | 15      | 5          | 200         | 7       | 8       | 0.6597       |	0.6466        |	0.6498        |	0.6606        |	0.6762        |	0.7876        |
| 0   | 15      | 5          | 300         | 5       | 8       | 0.6628       |	0.6498        |	0.6526        |	0.663         |	0.6777        |	0.789         |
| 0   | 15      | 5          | 300         | 6       | 8       | 0.6598       |	0.6479        |	0.6508        |	0.6617        |	0.6768        |	0.7881        |
| 0   | 15      | 5          | 300         | 7       | 8       | 0.6597       |	0.6477        |	0.6501        |	0.6613        |	0.6758        |	0.7883        |
| 0   | 15      | 5          | 400         | 5       | 8       | 0.663        |	0.6493        |	0.6536        |	0.664         |	0.6784        |	0.7898        |
| 0   | 15      | 5          | 400         | 6       | 8       | 0.6633       |	0.6511        |	0.6549        |	0.6647        |	0.6798        |	0.7907        |
| 0   | 15      | 5          | 400         | 7       | 8       | 0.6634       |	0.6498        |	0.6526        |	0.664         |	0.6789        |	0.7897        |
| 1   | 15      | 5          | 200         | 5       | 8       | 0.6591       |	0.649         |	0.652         |	0.6615        |	0.6771        |	0.7881        |
| 1   | 15      | 5          | 200         | 6       | 8       | 0.6551       |	0.6463        |	0.6502        |	0.6608        |	0.6762        |	0.7872        |
| 1   | 15      | 5          | 200         | 7       | 8       | 0.6586       |	0.646         |	0.6483        |	0.6599        |	0.6756        |	0.7873        |
| 1   | 15      | 5          | 300         | 5       | 8       | 0.6597       |	0.6494        |	0.6525        |	0.6622        |	0.6778        |	0.7882        |
| 1   | 15      | 5          | 300         | 6       | 8       | 0.6579       |	0.6464        |	0.6501        |	0.6606        |	0.6756        |	0.7876        |
| 1   | 15      | 5          | 300         | 7       | 8       | 0.6553       |	0.644         |	0.6483        |	0.6591        |	0.6743        |	0.7865        |
| 1   | 15      | 5          | 400         | 5       | 8       | 0.6611       |	0.6485        |	0.652         |	0.6613        |	0.677         |	0.7878        |
| 1   | 15      | 5          | 400         | 6       | 8       | 0.6586       |	0.6465        |	0.6495        |	0.6604        |	0.676         |	0.7873        |
| 1   | 15      | 5          | 400         | 7       | 8       | 0.6567       |	0.6449        |	0.6491        |	0.6597        |	0.675         |	0.7865        |

**TREC-repurposed:** The table below shows the results from the nDCG@N approach for the "TREC-repurposed" variant of the TREC dataset.

| dm  | epochs  | min_count  | vector_size | window  | workers | nDCG@5 (AVG) | nDCG@10 (AVG) | nDCG@15 (AVG) | nDCG@20 (AVG) | nDCG@25 (AVG) | nDCG@50 (AVG) |
|:---:|:-------:|:----------:|:-----------:|:-------:|:-------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| 0   | 15      | 5          | 200         | 5       | 8       | 0.4798       |	0.4737        |	0.4722        |	0.4714        |	0.4721        |	0.4849        |
| 0   | 15      | 5          | 200         | 6       | 8       | 0.4773       |	0.4715        |	0.4685        |	0.4693        |	0.47          |	0.4831        |
| 0   | 15      | 5          | 200         | 7       | 8       | 0.4763       |	0.4724        |	0.47          |	0.4695        |	0.4706        |	0.4831        |
| 0   | 15      | 5          | 300         | 5       | 8       | 0.4788       |	0.4737        |	0.4718        |	0.4717        |	0.4735        |	0.4849        |
| 0   | 15      | 5          | 300         | 6       | 8       | 0.4765       | 0.4706        |	0.4701        |	0.4703        |	0.4710        |	0.4837        |
| 0   | 15      | 5          | 300         | 7       | 8       | 0.4755       |	0.4717        |	0.4707        |	0.4702        |	0.4713        |	0.4842        |
| 0   | 15      | 5          | 400         | 5       | 8       | 0.479        |	0.4731        |	0.4708        |	0.4711        |	0.4728        |	0.4851        |
| 0   | 15      | 5          | 400         | 6       | 8       | 0.4767       |	0.4711        |	0.4703        |	0.4702        |	0.4720        |	0.4842        |
| 0   | 15      | 5          | 400         | 7       | 8       | 0.4789       |	0.4731        |	0.4712        |	0.4711        |	0.4728        |	0.485         |
| 1   | 15      | 5          | 200         | 5       | 8       | 0.4637       |	0.461         |	0.4596        |	0.4592        |	0.4595        |	0.4720        |
| 1   | 15      | 5          | 200         | 6       | 8       | 0.4665       |	0.461         |	0.4604        |	0.4593        |	0.46          |	0.4719        |
| 1   | 15      | 5          | 200         | 7       | 8       | 0.4575       |	0.452         |	0.4508        |	0.4507        |	0.4515        |	0.4649        |
| 1   | 15      | 5          | 300         | 5       | 8       | 0.4609       |	0.457         |	0.4557        |	0.456         |	0.4572        |	0.4698        |
| 1   | 15      | 5          | 300         | 6       | 8       | 0.4616       |	0.4569        |	0.4551        |	0.455         |	0.4566        |	0.4686        |
| 1   | 15      | 5          | 300         | 7       | 8       | 0.4592       |	0.4537        |	0.4525        | 0.4527        |	0.4542        |	0.4668        |
| 1   | 15      | 5          | 400         | 5       | 8       | 0.4636       |	0.4581        |	0.4573        |	0.4580        |	0.4584        |	0.4707        |
| 1   | 15      | 5          | 400         | 6       | 8       | 0.4603       |	0.4562        |	0.4545        |	0.4544        |	0.4552        |	0.4692        |
| 1   | 15      | 5          | 400         | 7       | 8       | 0.4540       |	0.4482        |	0.4479        |	0.4481        |	0.4502        |	0.4640        |

