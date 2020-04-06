#!/usr/bin/env sh

STORAGE=$1

java edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,ssplit \
    -tokenize.whitespace \
    -outputFormat conll \
    -filelist ${STORAGE}/ptbwsj_wo_rstdt/tmp.preprocessing/filelist.corenlp.txt \
    -outputDirectory ${STORAGE}/ptbwsj_wo_rstdt/tmp.preprocessing

