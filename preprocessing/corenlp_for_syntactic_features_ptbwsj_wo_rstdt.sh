#!/usr/bin/env sh

STORAGE=$1

java edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,ssplit,pos,lemma,depparse \
    -tokenize.whitespace \
    -ssplit.eolonly \
    -outputFormat conll \
    -filelist ${STORAGE}/ptbwsj_wo_rstdt/tmp.preprocessing/filelist.corenlp2.txt \
    -outputDirectory ${STORAGE}/ptbwsj_wo_rstdt/tmp.preprocessing
