# DiscourseConstituencyInduction-ViterbiEM

(c) 2020 Noriki Nishida

This is an implementation of an unsupervised discourse constituency parser described in the paper:

> Noriki Nishida and Hideki Nakayama. 2020. [Unsupervised Discourse Constituency Parsing Using Viterbi EM](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00312). Transactions of the Association for Computational Linguistics, vol.8, pp.215-230.

## Task Definition ##

- Unsupervised discourse constituency parsing based on Rhetorical Structure Theory
- Input: EDUs, syntactic features, sentence/paragraph boundaries
- Output: Unlabeled RST-style constituent tree

## Setup ##

### Requirements

- numpy
- spacy >= 2.1.9
- chainer >= 6.1.0
- multiset
- jsonlines
- pyprind

### Clone this repository and create directories to store preprocessed data and outputs

```
$ git clone https://github.com/norikinishida/DiscourseConstituencyInduction-ViterbiEM
$ cd ./DiscourseConstituencyInduction-ViterbiEM
$ mkdir ./data
$ mkdir ./results
```

### Edit ```./run_preprocessing.sh``` as follows:

```shell
STORAGE=./data
```

### Edit ```./config/path.ini``` as follows:

```INI
data = "./data"
results = "./results"
pretrained_word_embeddings = "/path/to/your/pretrained_word_embeddings"
rstdt = "/path/to/rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0"
ptbwsj = "/path/to/LDC99T42/treebank_3/raw/wsj"
```

### Clone other libraries

```
$ mkdir ./tmp
$ cd ./tmp
$ pip install pandas
$ pip install scikit-learn
$ pip install gensim
$ pip install nltk
$ git clone https://github.com/norikinishida/utils.git
$ git clone https://github.com/norikinishida/treetk.git
$ cp -r ./utils/utils ..
$ cp -r ./treetk/treetk ..
```

## Preprocessing ##

```
./run_preprocessing.sh
```

- The following directories will be generated:
    - ./data/rstdt/wsj/{train,test} (preprocessed RST-DT)
    - ./data/ptbwsj_wo_rstdt (preprocessed PTB-WSJ)
    - ./data/rstdt-vocab (vocabularies)

- NOTE: We rewrote this part from scratch using spaCy to make the codes much simpler than the previous ones. (2020/05/11)

## Training ##

- Training data: RST-DT training set

```
python main.py --gpu 0 --model spanbasedmodel2 --initial_tree_sampling RB2_RB_LB --config ./config/hyperparams_2.ini --name trial1 --actiontype train --max_epoch 15
```

- The following files will be generated:
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.training.log
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.training.jsonl
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.model
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.valid_pred.ctrees (optional)
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.valid_gold.ctrees (optional)
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.validation.jsonl (optional)

## Evaluation ##

- Metrics: RST PARSEVAL by Morey et al. (2018)
- Test data: RST-DT test set

```
python main.py --gpu 0 --model spanbasedmodel2 --initial_tree_sampling RB2_RB_LB --config ./config/hyperparams_2.ini --name trial1 --actiontype evaluate
```

- The following files will be generated:
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.evaluation.ctrees
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.evaluation.json

## Citation ##

If you use the code in research publications, please cite:

```
@article{nishida2018unsupervised,
    author={Nishida, Noriki and Nakayama, Hideki},
    title={Unsupervised Discourse Constituency Parsing Using Viterbi EM},
    journal={Transactions of the Association for Computational Linguistics},
    volume={8},
    number={},
    pages={215-230},
    year={2020},
    doi={10.1162/tacl\_a\_00312},
    URL={https://doi.org/10.1162/tacl_a_00312},
}
```

