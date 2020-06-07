#!/usr/bin/env sh

STORAGE=/mnt/hdd/projects/discourse/DiscourseConstituencyInduction-ViterbiEM/data

#####################
# PTB-WSJ

mkdir ../data/ptbwsj_wo_rstdt.out

python run.py --segment --input_files ${STORAGE}/ptbwsj_wo_rstdt/*.sents.tokens --result_dir ../data/ptbwsj_wo_rstdt.out/

python rename.py --input_files ../data/ptbwsj_wo_rstdt.out/*.sents.tokens --src_ext ".sents.tokens" --dst_ext ".edus.tokens"

cp ../data/ptbwsj_wo_rstdt.out/*.edus.tokens ${STORAGE}/ptbwsj_wo_rstdt

#####################
# ACL Anthology Reference Corpus (AARC; only abstract)

mkdir ../data/aarc_abst.out

find ${STORAGE}/aarc_abst -name *.sents.tokens > ../data/aarc_abst.out/filelist.txt
python run.py --segment --input_file_list ../data/aarc_abst.out/filelist.txt --result_dir ../data/aarc_abst.out/

find ../data/aarc_abst.out -name *.sents.tokens > ../data/aarc_abst.out/filelist2.txt
python rename.py --input_file_list ../data/aarc_abst.out/filelist2.txt --src_ext ".sents.tokens" --dst_ext ".edus.tokens"

find ../data/aarc_abst.out -name *.edus.tokens -print0 | xargs -0 -I {} cp {} ${STORAGE}/aarc_abst

#####################
# CORD-19 (only abstract)

mkdir ../data/cord19_abst.out

find ${STORAGE}/cord19_abst -name *.sents.tokens > ../data/cord19_abst.out/filelist.txt
python run.py --segment --input_file_list ../data/cord19_abst.out/filelist.txt --result_dir ../data/cord19_abst.out/

find ../data/cord19_abst.out -name *.sents.tokens > ../data/cord19_abst.out/filelist2.txt
python rename.py --input_file_list ../data/cord19_abst.out/filelist2.txt --src_ext ".sents.tokens" --dst_ext ".edus.tokens"

find ../data/cord19_abst.out -name *.edus.tokens -print0 | xargs -0 -I {} cp {} ${STORAGE}/cord19_abst

