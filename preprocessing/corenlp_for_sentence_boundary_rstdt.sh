#!/usr/bin/env sh

STORAGE=$1

java edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,ssplit \
    -tokenize.whitespace \
    -outputFormat conll \
    -filelist ${STORAGE}/rstdt/tmp.preprocessing/filelist.corenlp.txt \
    -outputDirectory ${STORAGE}/rstdt/tmp.preprocessing
