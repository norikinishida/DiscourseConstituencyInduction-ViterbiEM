#!/usr/bin/env sh

STORAGE=/mnt/hdd/projects/discourse/PreprocessedRSTDT/data

mkdir ./patch_data
mkdir ./patch_data/train
mkdir ./patch_data/test

python ./preprocessing/find_conflictions_btw_goldedus_and_document.py --check_token --check_char

for number in 0430 0764 0766 1110 1156 1158 1328 1339 1367 1377 2172 2349
do
    wdiff \
        ${STORAGE}/rstdt/wsj/train/wsj_${number}.doc.tokens \
        ${STORAGE}/rstdt/wsj/train/wsj_${number}.edus.tokens \
        > ./patch_data/train/wsj_${number}.edus.tokens
    cp ${STORAGE}/rstdt/wsj/train/wsj_${number}.doc.tokens ./patch_data/train/wsj_${number}.doc.tokens
done

### HERE: Edit ./patch_data/train/wsj_xxxx{.edus.tokens, .doc.tokens}

cp ./patch_data/train/*.tokens ${STORAGE}/rstdt/wsj/train/

python ./preprocessing/find_conflictions_btw_goldedus_and_document.py --check_boundary

for number in 1123 1139 1373 1398 2317 2366
do
    cp ${STORAGE}/rstdt/wsj/train/wsj_${number}.edus.tokens ./patch_data/train/wsj_${number}.edus.tokens
    cp ${STORAGE}/rstdt/wsj/train/wsj_${number}.doc.tokens ./patch_data/train/wsj_${number}.doc.tokens
    python ./preprocessing/fix_paragraph_boundary_conflictions.py --path ./patch_data/train/wsj_${number}.doc.tokens
    mv ./patch_data/train/wsj_${number}.doc.tokens.fixed ./patch_data/train/wsj_${number}.doc.tokens
done

for number in 1376
do
    cp ${STORAGE}/rstdt/wsj/test/wsj_${number}.edus.tokens ./patch_data/test/wsj_${number}.edus.tokens
    cp ${STORAGE}/rstdt/wsj/test/wsj_${number}.doc.tokens ./patch_data/test/wsj_${number}.doc.tokens
    python ./preprocessing/fix_paragraph_boundary_conflictions.py --path ./patch_data/test/wsj_${number}.doc.tokens
    mv ./patch_data/test/wsj_${number}.doc.tokens.fixed ./patch_data/test/wsj_${number}.doc.tokens
done

