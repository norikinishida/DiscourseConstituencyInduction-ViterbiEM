#!/usr/bin/env sh

STORAGE=/mnt/hdd/projects/discourse/PreprocessedRSTDT/data

mkdir ../data/ptbwsj_wo_rstdt.out

python run.py --segment --input_files ${STORAGE}/ptbwsj_wo_rstdt/*.sents.tokens --result_dir ../data/ptbwsj_wo_rstdt.out/

python rename.py --input_files ../data/ptbwsj_wo_rstdt.out/*.sents.tokens --src_ext ".sents.tokens" --dst_ext ".edus.tokens"

cp ../data/ptbwsj_wo_rstdt.out/*.edus.tokens ${STORAGE}/ptbwsj_wo_rstdt

