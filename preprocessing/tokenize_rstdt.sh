#!/usr/bin/env sh

STORAGE=$1

java edu.stanford.nlp.process.PTBTokenizer \
    --ioFileList ${STORAGE}/rstdt/tmp.preprocessing/filelist.ptbtokenizer.txt \
    --preserveLines \
    --options "normalizeSpace=false"
