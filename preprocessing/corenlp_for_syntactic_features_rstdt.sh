#!/usr/bin/env sh

STORAGE=$1

java edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,ssplit,pos,lemma,depparse \
    -tokenize.whitespace \
    -ssplit.eolonly \
    -outputFormat conll \
    -filelist ${STORAGE}/rstdt/tmp.preprocessing/filelist.corenlp2.txt \
    -outputDirectory ${STORAGE}/rstdt/tmp.preprocessing
