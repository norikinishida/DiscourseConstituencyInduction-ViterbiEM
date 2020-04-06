#!/usr/bin/env sh

STORAGE=$1

for NUMBER in 103 143 145 243 276 290
do
    wdiff \
        ${STORAGE}/rstdt/tmp.preprocessing/train.${NUMBER}.edus.tokenized \
        ${STORAGE}/rstdt/tmp.preprocessing/train.${NUMBER}.raw.tokenized \
        > ./train.${NUMBER}.edus.wdiff
    vim -o ${STORAGE}/rstdt/tmp.preprocessing/train.${NUMBER}.raw.tokenized ./train.${NUMBER}.edus.wdiff
done

