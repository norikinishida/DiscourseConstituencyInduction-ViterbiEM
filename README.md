# Unsupervised Discourse Constituency Parsing Using Viterbi EM


- Source codes used in our TACL paper, "Unsupervised Discourse Constituency Parsing Using Viterbi EM" (Nishida and Nakayama, 2020).

- Task: Unsupervised discourse constituency parsing based on Rhetorical Structure Theory
    - Input: EDUs + syntactic features (i.e., POS tags, dependency graphs) + sentence/paragraph boundaries
    - Output: Text-level unlabeled tree

## Requirements ##

- numpy
- chainer >= 6.1.0
- multiset
- jsonlines
- pyprind
- [Stanford Tokenizer](https://nlp.stanford.edu/static/software/tokenizer.shtml)
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html)

## Configuration ##

1. Clone this repository and create directories to store preprocessed data and outputs

```
$ git clone https://github.com/norikinishida/DiscourseConstituencyInduction-ViterbiEM
$ cd ./DiscourseConstituencyInduction-ViterbiEM
$ mkdir ./data
$ mkdir ./results
```

2. Edit ```./run_preprocessing.sh``` as follows:

```shell
STORAGE=./data
```

3. Edit ```./config/path.ini``` as follows:

```INI
data = "./data"
results = "./results"
pretrained_word_embeddings = "/path/to/your/pretrained_word_embeddings"
rstdt = "/path/to/rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0"
ptbwsj = "/path/to/LDC99T42/treebank_3/raw/wsj"
```

4. Clone other libraries

```
$ mkdir ./tmp
$ cd ./tmp
$ pip install pandas
$ pip install scikit-learn
$ pip install gensim
$ pip install nltk
$ git clone https://github.com/norikinishida/utils.git
$ git clone https://github.com/norikinishida/treetk.git
$ git clone https://github.com/norikinishida/textpreprocessor.git
$ cp -r ./utils/utils ..
$ cp -r ./treetk/treetk ..
$ cp -r ./textpreprocessor/textpreprocessor ..
```

## Preprocessing ##

- Run the following script:

```
./run_preprocessing.sh
```

- The following directories will be generated:
    - ./data/rstdt/renamed (preprocessed data of RST-DT)
    - ./data/rstdt-vocab (vocabularies and class names)
    - ./data/ptbwsj_wo_rstdt (preprocessed data of PTB-WSJ)

## Parsing Model: Span-based Model ##

- EDU-level feature extraction
    - word embeddings of the beginning/end words
    - POS embeddings of the beginning/end words
    - word/POS/dependency embeddings of the head word

- Span-level feature extraction
    - bidirectional LSTM
    - span differences
    - no template features

- Span scoring
    - MLP for bracket scoring

- Decoding algorithm (unlabeled tree-building)
    - CKY

- Labeling
    - Relations (+ nuclearities) are "ELABORATION-NS" (i.e., majority label)

## Training ##

- Viterbi EM (i.e., self training) + initial-tree sampling based on prior knowledge
- Loss function: Margin-based criterion
- Training data: RST-DT training set
- Run the following command:

```
python main.py --gpu 0 --model spanbasedmodel2 --initial_tree_sampling RB2_RB_LB --config ./config/hyperparams_2.ini --name trial1 --actiontype train --max_epoch 10
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
- Run the following command:

```
python main.py --gpu 0 --model spanbasedmodel2 --initial_tree_sampling RB2_RB_LB --config ./config/hyperparams_2.ini --name trial1 --actiontype evaluate
```

- The following files will be generated:
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.evaluation.ctrees
    - ./results/spanbasedmodel2.RB2_RB_LB.hyperparams_2.aug_False.trial1.evaluation.json

