# ERSTG

code for "Syntactic type-aware graph attention network for biomedical overlapping entity relation extraction"  
our ERSTG based on SpERT framework,
###  Examples Instructions
(1) Train ADE:
```
python ./enRelTsg.py train --config configs/ade_train.conf
```

(2) Evaluate the BioNLP11EPI model on test dataset:
```
python ./enRelTsg.py eval --config configs/ade_eval.conf
```

### Fetch data
## Download Datasets 
- ADE: [ADE](http://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/)   
- DDI: [DDI]( https://hulat.inf.uc3m.es/ddicorpus ) 
- CoNLL04 : [CoNLL04](http://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/)   
 
## dataset process
 The syntactic dependency parses and POS labels are obtained by using [Stanford CoreNLP Toolkit ](https://stanfordnlp.github.io/CoreNLP/)

processed data formate like sample_data.json
```json
{
 "tokens": ["We", "report", "a", "case", "of", "fulminant", "hepatic", "failure", "associated", "with", "didanosine",  "and", "masquerading", "as", "a", "surgical", "abdomen", "and", "compare", "the", "clinical", ",", "biologic", ",", "histologic", ",", "and", "ultrastructural", "findings", "with", "reports", "described", "previously", "."], 
 "dep_label": ["nsubj", "ROOT", "det", "obj", "case", "amod", "amod", "nmod", "dep", "case", "obl", "cc", "conj", "case", "det", "amod", "obl", "cc", "conj", "obj", "dep", "punct", "dep", "punct", "dep", "punct", "cc", "amod", "conj", "case", "nmod", "acl", "advmod", "punct"], 
 "dep_label_indices": [11, 10, 8, 13, 3, 4, 4, 6, 12, 3, 9, 15, 14, 3, 8, 4, 9, 15, 14, 13, 12, 5, 12, 5, 12, 5, 15, 4, 14, 3, 6, 21, 16, 5], 
 "dep": ["2", "0", "4", "2", "8", "8", "8", "4", "8", "11", "9", "13", "9", "17", "17", "17", "13", "19", "2", "19", "20", "21", "22", "23", "24", "25", "29", "29", "26", "31", "29", "31", "32", "29"], 
 "pos_label": ["PRP", "VBP", "DT", "NN", "IN", "JJ", "JJ", "NN", "VBN", "IN", "NN", "CC", "VBG", "IN", "DT", "JJ", "NN", "CC", "VB", "DT", "JJ", ",", "JJ", ",", "JJ", ",", "CC", "JJ", "NNS", "IN", "NNS", "VBN", "RB", "."], "pos_indices": [18, 15, 5, 2, 3, 4, 4, 2, 7, 3, 2, 10, 14, 3, 5, 4, 2, 10, 17, 5, 4, 9, 4, 9, 4, 9, 10, 4, 8, 3, 8, 7, 13, 6], 
 "entities": [{"type": "Adverse-Effect", "start": 5, "end": 8}, {"type": "Drug", "start": 10, "end": 11}], 
 "relations": [{"type": "Adverse-Effect", "head": 0, "tail": 1}], 
 "orig_id": 412
 },
```


## References
```
[1] Eberts, Markus, and Adrian Ulges. "Span-based joint entity and relation extraction with transformer pre-training." arXiv preprint arXiv:1909.07755 (2019).
```



