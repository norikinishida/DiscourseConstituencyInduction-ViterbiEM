#!/usr/bin/env sh

STORAGE=/mnt/hdd/projects/discourse/DiscourseConstituencyInduction-ViterbiEM/data

################################
# RST-DT, PTB-WSJ

# Step 1 (RST-DT): wsj_xxxx.edus.tokens (original), wsj_xxx.*.*.ctree
python ./preprocessing/prepare_rstdt.py

# Step 2 (RST-DT, PTB-WSJ): wsj_xxxx.doc.tokens
python ./preprocessing/prepare_ptbwsj.py --inside_rstdt
python ./preprocessing/prepare_ptbwsj.py --outside_rstdt

# Step 3 (RST-DT): wsj_xxxx.edus.tokens (fixed version), wsj_xxxx.doc.tokens (fixed version)
### ./preprocessing/create_patch_data.sh
cp ./patch_data/train/*.tokens ${STORAGE}/rstdt/wsj/train/
cp ./patch_data/test/*.tokens ${STORAGE}/rstdt/wsj/test/
python ./preprocessing/find_conflictions_btw_goldedus_and_document.py --check_token --check_char --check_boundary

# Step 4 (RST-DT, PTB-WSJ): wsj_xxxx.sents.tokens, wsj_xxxx.sents.postags, wsj_xxxx.sents.arcs, wsj_xxxx.pbnds
python ./preprocessing/doc2sents.py --path ${STORAGE}/rstdt/wsj/train --with_gold_edus
python ./preprocessing/doc2sents.py --path ${STORAGE}/rstdt/wsj/test --with_gold_edus
python ./preprocessing/doc2sents.py --path ${STORAGE}/ptbwsj_wo_rstdt

# Step 5 (PTB-WSJ): wsj_xxxx.edus.tokens
# NOTE: Use PKU-TANGENT/NeuralEDUSeg.git to segment PTB-WSJ documents into EDUs. (c.f., ./preprocessing/segment.sh)

# Step 6 (RST-DT, PTB-WSJ): wsj_xxxx.edus.postags, wsj_xxxx.edus.arcs, wsj_xxxx.edus.heads, wsj_xxxx.edus.deprels, wsj_xxxx.sbnds
python ./preprocessing/sents2edus.py --path ${STORAGE}/rstdt/wsj/train
python ./preprocessing/sents2edus.py --path ${STORAGE}/rstdt/wsj/test
python ./preprocessing/sents2edus.py --path ${STORAGE}/ptbwsj_wo_rstdt
python ./preprocessing/extract_head_from_arcs.py --path ${STORAGE}/rstdt/wsj/train
python ./preprocessing/extract_head_from_arcs.py --path ${STORAGE}/rstdt/wsj/test
python ./preprocessing/extract_head_from_arcs.py --path ${STORAGE}/ptbwsj_wo_rstdt
python ./preprocessing/extract_deprels_from_arcs.py --path ${STORAGE}/rstdt/wsj/train
python ./preprocessing/extract_deprels_from_arcs.py --path ${STORAGE}/rstdt/wsj/test
python ./preprocessing/extract_deprels_from_arcs.py --path ${STORAGE}/ptbwsj_wo_rstdt

# Step 7 (RST-DT, PTB-WSJ): Preprocessing: wsj_xxxx.edus.tokens.preprocessed, {words,postags,deprels}.vocab.txt
python ./preprocessing/preprocess_tokens.py --files ${STORAGE}/rstdt/wsj/train/*.edus.tokens
python ./preprocessing/preprocess_tokens.py --files ${STORAGE}/rstdt/wsj/test/*.edus.tokens
python ./preprocessing/preprocess_tokens.py --files ${STORAGE}/ptbwsj_wo_rstdt/*.edus.tokens

mkdir ${STORAGE}/rstdt-vocab
python ./preprocessing/build_vocabulary.py --files ${STORAGE}/rstdt/wsj/*/*.edus.tokens.preprocessed --vocab ${STORAGE}/rstdt-vocab/words.vocab.txt
python ./preprocessing/build_vocabulary.py --files ${STORAGE}/rstdt/wsj/*/*.edus.postags --vocab ${STORAGE}/rstdt-vocab/postags.vocab.txt
python ./preprocessing/build_vocabulary.py --files ${STORAGE}/rstdt/wsj/*/*.edus.deprels --vocab ${STORAGE}/rstdt-vocab/deprels.vocab.txt

python ./preprocessing/replace_oov_tokens.py --files ${STORAGE}/rstdt/wsj/train/*.edus.tokens.preprocessed --vocab ${STORAGE}/rstdt-vocab/words.vocab.txt
python ./preprocessing/replace_oov_tokens.py --files ${STORAGE}/rstdt/wsj/test/*.edus.tokens.preprocessed --vocab ${STORAGE}/rstdt-vocab/words.vocab.txt
python ./preprocessing/replace_oov_tokens.py --files ${STORAGE}/ptbwsj_wo_rstdt/*.edus.tokens.preprocessed --vocab ${STORAGE}/rstdt-vocab/words.vocab.txt

# Step 8 (RST-DT): supervision
python ./preprocessing/build_discourse_label_vocabulary_rstdt.py
python ./preprocessing/convert_ctrees_to_dtrees_rstdt.py
python ./preprocessing/prepare_for_evaluation_rstdt.py

