#!/usr/bin/env sh

GPU=0
MODEL=spanbasedmodel2
INITIAL_TREE_SAMPLING=RB2_RB_LB
CONFIG=./config/hyperparams_2.ini
NAME=trial1

MAX_EPOCH=15
python main.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --initial_tree_sampling ${INITIAL_TREE_SAMPLING} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype train \
    --max_epoch ${MAX_EPOCH}

python main.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --initial_tree_sampling ${INITIAL_TREE_SAMPLING} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype evaluate

