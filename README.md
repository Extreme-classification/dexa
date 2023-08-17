# NGAME

Code for _DEXA: Deep Encoders with Auxiliary Parameters for Extreme Classification_ [1]

---

## Setting up

---

### Expected directory structure

```txt
+-- <work_dir>
|  +-- programs
|  |  +-- dexa
|  |    +-- dexa
|  +-- data
|    +-- <dataset>
|  +-- models
|  +-- results
```

### Download data for DEXA

```txt
* Download the (zipped file) raw data from The XML repository [5].  
* Extract the zipped file into data directory. 
* The following files should be available in <work_dir>/data/<dataset> (create empty filter files if unavailable):
    - trn.json.gz
    - tst.json.gz
    - lbl.json.gz
    - filter_labels_text.txt
    - filter_labels_train.txt
```

## Example use cases

---

### A single learner

Extract and tokenize data as follows.

```bash
./prepare_data.sh LF-AmazonTitles-131K 32
```

The algorithm can be run as follows. A json file (e.g., config/NGAME/LF-AmazonTitles-131K.json) is used to specify architecture and other arguments. Please refer to the full documentation below for more details.

```bash
./run_main.sh 0 NGAME LF-AmazonTitles-131K 0 108
```

## Full Documentation

### Tokenize the data

```txt
./prepare_data.sh <dataset> <seq-len>

* dataset
  - Name of the dataset.
  - Tokenizer expects the following files in <work_dir>/data/<dataset>
    - trn.json.gz
    - tst.json.gz
    - lbl.json.gz
  - it'll dump the following six tokenized files 
    - trn_doc_input_ids.npy
    - trn_doc_attention_mask.npy
    - tst_doc_input_ids.npy
    - tst_doc_attention_mask.npy
    - lbl_input_ids.npy
    - lbl_attention_mask.npy

* seq-len
  - sequence length of text to consider while tokenizing
  - 32 for titles dataset
  - 256 for Wikipedia
  - 128 for other full-text datasets
```

### Run DEXA

```txt
./run_main.sh <gpu_id> <type> <dataset> <version> <seed>

* gpu_id: Run the program on this GPU.

* type
  DEXA builds upon NGAME[2], SiameseXML [3] and DeepXML[4] for training. An encoder is trained in M1 and the classifier is trained in M-IV.
  - DEXA: The intermediate representation is not fine-tuned while training the classifier (more scalable; suitable for large datasets).

* dataset
  - Name of the dataset.
  - DEXA expects the following files in <work_dir>/data/<dataset>
    - trn_doc_input_ids.npy
    - trn_doc_attention_mask.npy
    - trn_X_Y.txt
    - tst_doc_input_ids.npy
    - tst_doc_attention_mask.npy
    - tst_X_Y.txt
    - lbl_input_ids.npy
    - lbl_attention_mask.npy
    - filter_labels_test.txt (put empty file or set as null in config when unavailable)

* version
  - different runs could be managed by version and seed.
  - models and results are stored with this argument.

* seed
  - seed value as used by numpy and PyTorch.
```

## Cite as

```bib
@InProceedings{Dahiya23b,
    author = "Dahiya, K. and Yadav, S. and Sondhi, S. and Saini, D. and Mehta, S. and Jiao, J. and Agarwal, S. and Kar, P. and Varma, M.",
    title = "Deep encoders with auxiliary parameters for extreme classification",
    booktitle = "KDD",
    month = "August",
    year = "2023"
}
```

## YOU MAY ALSO LIKE

- [NGAME: Negative mining-aware mini-batching for extreme classification](https://github.com/Extreme-classification/ngame)
- [SiameseXML: Siamese networks meet extreme classifiers with 100M labels](https://github.com/Extreme-classification/siamesexml)
- [DeepXML: A Deep Extreme Multi-Label Learning Framework Applied to Short Text Documents](https://github.com/Extreme-classification/deepxml)
- [DECAF: Deep Extreme Classification with Label Features](https://github.com/Extreme-classification/DECAF)
- [ECLARE: Extreme Classification with Label Graph Correlations](https://github.com/Extreme-classification/ECLARE)
- [GalaXC: Graph Neural Networks with Labelwise Attention for Extreme Classification](https://github.com/Extreme-classification/GalaXC)

## References

---
[1] K. Dahiya, S. Yadav, S. Sondhi, D. Saini, S. Mehta, J. Jiao, S. Agarwal, P. Kar and M. Varma. Deep encoders with auxiliary parameters for extreme classification. In KDD, Long Beach (CA), August 2023.

[2] K. Dahiya, N. Gupta, D. Saini, A. Soni, Y. Wang, K. Dave, J. Jiao, K. Gururaj, P. Dey, A. Singh, D. Hada, V. Jain, B. Paliwal, A. Mittal, S. Mehta, R. Ramjee, S. Agarwal, P. Kar and M. Varma. NGAME: Negative mining-aware mini-batching for extreme classification. In WSDM, Singapore, March 2023.

[2] K. Dahiya, A. Agarwal, D. Saini, K. Gururaj, J. Jiao, A. Singh, S. Agarwal, P. Kar and M. Varma. SiameseXML: Siamese networks meet extreme classifiers with 100M labels. In ICML, July 2021

[3] K. Dahiya, D. Saini, A. Mittal, A. Shaw, K. Dave, A. Soni, H. Jain, S. Agarwal, and M. Varma. Deepxml:  A deep extreme multi-label learning framework applied to short text documents. In WSDM, 2021.

[4] pyxclib: <https://github.com/kunaldahiya/pyxclib>

[5] The Extreme Classification Repository: <http://manikvarma.org/downloads/XC/XMLRepository.html>
