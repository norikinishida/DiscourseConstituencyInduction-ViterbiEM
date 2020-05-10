#!/usr/bin/env sh

STORAGE=/mnt/hdd/projects/discourse/PreprocessedRSTDT/data

# Step 1 (RST-DT): wsj_xxxx.edus.tokens, wsj_xxx.*.*.ctree
python ./preprocessing/prepare_rstdt.py

# Step 2 (RST-DT, PTB-WSJ): wsj_xxxx.doc.tokens
python ./preprocessing/prepare_ptbwsj.py --inside_rstdt
python ./preprocessing/prepare_ptbwsj.py --outside_rstdt

# Step 3 (RST-DT): wsj_xxxx.edus.tokens
### ./preprocessing/create_patch_data.sh
cp ./patch_data/train/*.tokens ${STORAGE}/rstdt/wsj/train/
cp ./patch_data/test/*.tokens ${STORAGE}/rstdt/wsj/test/
python ./preprocessing/find_conflictions_btw_goldedus_and_document.py --check_token --check_char --check_boundary

# Step 4 (RST-DT, PTB-WSJ): wsj_xxxx.pbnds, wsj_xxxx.sents.tokens, wsj_xxxx.sents.postags, wsj_xxxx.sents.arcs
python ./preprocessing/doc2sents.py --path ${STORAGE}/rstdt/wsj/train --with_gold_edus
python ./preprocessing/doc2sents.py --path ${STORAGE}/rstdt/wsj/test --with_gold_edus
python ./preprocessing/doc2sents.py --path ${STORAGE}/ptbwsj_wo_rstdt

# Step 5 (RST-DT): wsj_xxxx.sbnds, wsj_xxxx.edus.postags, wsj_xxxx.edus.arcs, wsj_xxxx.edus.heads
python ./preprocessing/sents2edus.py --path ${STORAGE}/rstdt/wsj/train
python ./preprocessing/sents2edus.py --path ${STORAGE}/rstdt/wsj/test
python ./preprocessing/extract_head_from_sub_arcs.py --path ${STORAGE}/rstdt/wsj/train
python ./preprocessing/extract_head_from_sub_arcs.py --path ${STORAGE}/rstdt/wsj/test

# Step 6 (PTB-WSJ): wsj_xxxx.sbnds, wsj_xxxx.edus.tokens, wsj_xxxx.edus.postags, wsj_xxxx.edus.arcs, wsj_xxxx.edus.heads
# NOTE: Use PKU-TANGENT/NeuralEDUSeg.git to segment PTB-WSJ documents into EDUs. (c.f., ./preprocessing/segment.sh)
python ./preprocessing/sents2edus.py --path ${STORAGE}/ptbwsj_wo_rstdt
python ./preprocessing/extract_head_from_sub_arcs.py --path ${STORAGE}/ptbwsj_wo_rstdt

# Step 7 (RST-DT, PTB-WSJ): Lowercasing and replacing OOV tokens: wsj_xxxx.edus.tokens.preprocessed, {words,postags,deprels}.vocab.txt
python ./preprocessing/preprocess_tokens.py --path ${STORAGE}/rstdt/wsj/train
python ./preprocessing/preprocess_tokens.py --path ${STORAGE}/rstdt/wsj/test
python ./preprocessing/preprocess_tokens.py --path ${STORAGE}/ptbwsj_wo_rstdt
python ./preprocessing/build_vocabulary_words.py
python ./preprocessing/replace_oov_tokens.py --path ${STORAGE}/rstdt/wsj/train
python ./preprocessing/replace_oov_tokens.py --path ${STORAGE}/rstdt/wsj/test
python ./preprocessing/replace_oov_tokens.py --path ${STORAGE}/ptbwsj_wo_rstdt
python ./preprocessing/build_vocabulary_postags.py
python ./preprocessing/build_vocabulary_deprels.py

# Step 8 (RST-DT): supervision
python ./preprocessing/build_vocabulary_disclabels.py
python ./preprocessing/convert_ctrees_to_dtrees_rstdt.py
python ./preprocessing/prepare_for_evaluation_rstdt.py

