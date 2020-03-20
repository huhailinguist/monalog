# MonaLog: a Lightweight System for Natural Language Inference Based on Monotonicity and Natural Logic

Code for a natural language inference (NLI) system. That is, given two sentences--a premise and a hypothesis, MonaLog outputs the inference relation from the premise to the hypothesis: `entailment`, `neutral` or `contradiction`.

## pipeline

`premise as raw sentence` --> `CCG parse tree` --> `polarized sentence with arrows` --> 

`generate entailments and contradictions of the premise` --> `check if hypothesis \in entailments/contradiction`


## replicating our work

- `parse.sh`:
Takes raw sentences and add polarity markings on each token. See [ccg2mono](https://github.com/huhailinguist/ccg2mono) for more information. 

- `sick.py`:
For experiment 1 in our 2020 SCiL [paper](https://scholarworks.umass.edu/scil/vol3/iss1/31/), which tries to solve the [SICK](http://marcobaroni.org/composes/sick.html) dataset. Use
`sick.py -h` to see options. The predictions from our system are included: `pred_monalog_preprocess_fwd.txt` for the forward direction in SICK and  `pred_monalog_preprocess_bkwd.txt` for the backward direction.

- `generate.py`:
For experiment 2 in our 2020 SCiL [paper](https://scholarworks.umass.edu/scil/vol3/iss1/31/), which generates more training data
for data augmentation in SICK. 

- `filter.py`:
For experiment 2 in our 2020 SCiL [paper](https://scholarworks.umass.edu/scil/vol3/iss1/31/), which does some very simple data filtering on the data generated in the last step. The resulting augmented training data is included in this foler: `gen.neutral_no.all.plus.train.all_labels.shuf`. You can use this file as training data for BERT or other models. 

- `fracas.py`:
For the 2019 IWCS [paper](https://www.aclweb.org/anthology/W19-0502/), which works on section 1 of the [FraCaS](https://nlp.stanford.edu/~wcmac/downloads/fracas.xml) dataset. Use
`fracas.py -h` to see options.

- other scripts contain helper functions. 

Parse trees of SICK and FraCaS sentences are already in this folder. 

## reference

If you use the code/system, please cite:

1. SCiL [paper](https://scholarworks.umass.edu/scil/vol3/iss1/31/)

```
@inproceedings{monalog,
  title={MonaLog: a Lightweight System for Natural Language Inference Based on Monotonicity},
  author={Hu, Hai and Chen, Qi and Richardson, Kyle and Mukherjee, Atreyee and Moss, Lawrence S and Kuebler, Sandra},
  booktitle={Proceedings of the Society for Computation in Linguistics (SCiL) 2020},
  pages=319--329,
  year={2020}
}
```

2. IWCS [paper](https://www.aclweb.org/anthology/W19-0502/)

```
@inproceedings{hu-etal-2019-natural,
    title = "Natural Language Inference with Monotonicity",
    author = "Hu, Hai  and Chen, Qi  and Moss, Larry",
    booktitle = "Proceedings of the 13th International Conference on Computational Semantics - Short Papers",
    month = may,
    year = "2019",
    pages = "8--15"
}
```

## contact

Hai Hu: huhai[at]indiana.edu

## license

Attribution-NonCommercial 4.0 International
