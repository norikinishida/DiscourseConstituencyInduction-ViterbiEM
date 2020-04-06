#!/usr/bin/env sh

STORAGE=/mnt/hdd/projects/discourse/DiscourseConstituencyInduction-ViterbiEM/data

python ./preprocessing/prepare_rstdt.py

# Preprocess texts
python ./preprocessing/make_filelist_for_tokenizing_rstdt.py
./preprocessing/tokenize_rstdt.sh ${STORAGE}
python ./preprocessing/lowercasing_rstdt.py
python ./preprocessing/replace_digits_rstdt.py
python ./preprocessing/build_vocabulary_rstdt.py
# python ./preprocessing/build_vocabulary_rstdt.py --with_root
python ./preprocessing/replace_rare_words_rstdt.py

# Convert constituent trees -> dependency trees
python ./preprocessing/convert_ctrees_to_dtrees_rstdt.py

# Extract sentence boundaries
python ./preprocessing/make_filelist_for_sentence_boundary_rstdt.py
./preprocessing/corenlp_for_sentence_boundary_rstdt.sh ${STORAGE}
python ./preprocessing/create_sentence_boundary_rstdt.py

# Extract paragraph boundaries
python ./preprocessing/make_filelist_for_paragraph_boundary_rstdt.py
./preprocessing/tokenize_for_paragraph_boundary_rstdt.sh ${STORAGE}
./preprocessing/edit_for_paragraph_boundary_rstdt.sh ${STORAGE}
python ./preprocessing/create_paragraph_boundary_rstdt.py

# Adjust sentence/paragraph boundaries
python ./preprocessing/create_hierarchical_boundary_rstdt.py

# Extract syntactic features
python ./preprocessing/make_filelist_for_syntactic_features_rstdt.py
./preprocessing/corenlp_for_syntactic_features_rstdt.sh ${STORAGE}
python ./preprocessing/extract_postags_for_edus_rstdt.py
python ./preprocessing/extract_syntactic_dependencies_for_edus_rstdt.py
python ./preprocessing/build_postag_vocabulary_rstdt.py
# python ./preprocessing/build_postag_vocabulary_rstdt.py --with_root
python ./preprocessing/build_deprel_vocabulary_rstdt.py
# python ./preprocessing/build_deprel_vocabulary_rstdt.py --with_root

python ./preprocessing/prepare_for_evaluation_rstdt.py

#############################################################

python ./preprocessing/prepare_ptbwsj_wo_rstdt.py

# Obtain EDU segmentation
mkdir -p ${STORAGE}/ptbwsj_wo_rstdt/segmented
# HERE: EDU segmentation using https://github.com/PKU-TANGENT/NeuralEDUSeg
python ./preprocessing/copy_edu_segmentation_ptbwsj_wo_rstdt.py

# Preprocess texts
python ./preprocessing/make_filelist_for_tokenizing_ptbwsj_wo_rstdt.py
./preprocessing/tokenize_ptbwsj_wo_rstdt.sh ${STORAGE}
python ./preprocessing/lowercasing_ptbwsj_wo_rstdt.py
python ./preprocessing/replace_digits_ptbwsj_wo_rstdt.py
python ./preprocessing/replace_rare_words_ptbwsj_wo_rstdt.py

# Extract sentence boundaries
python ./preprocessing/make_filelist_for_sentence_boundary_ptbwsj_wo_rstdt.py
./preprocessing/corenlp_for_sentence_boundary_ptbwsj_wo_rstdt.sh ${STORAGE}
python ./preprocessing/create_sentence_boundary_ptbwsj_wo_rstdt.py

# Extract paragraph boundaries
python ./preprocessing/make_filelist_for_paragraph_boundary_ptbwsj_wo_rstdt.py
./preprocessing/tokenize_for_paragraph_boundary_ptbwsj_wo_rstdt.sh ${STORAGE}
python ./preprocessing/create_paragraph_boundary_ptbwsj_wo_rstdt.py

# Adjust sentence/paragraph boundaries
python ./preprocessing/create_hierarchical_boundary_ptbwsj_wo_rstdt.py

# Extract syntactic features
python ./preprocessing/make_filelist_for_syntactic_features_ptbwsj_wo_rstdt.py
./preprocessing/corenlp_for_syntactic_features_ptbwsj_wo_rstdt.sh ${STORAGE}
python ./preprocessing/extract_postags_for_edus_ptbwsj_wo_rstdt.py
python ./preprocessing/extract_syntactic_dependencies_for_edus_ptbwsj_wo_rstdt.py

