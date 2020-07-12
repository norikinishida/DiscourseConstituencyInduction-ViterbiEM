#!/usr/bin/env sh

for name in trial1 trial2 trial3 trial4 trial5
do
    python evaluate_baseline.py \
        --tree_sampling X_X_BU \
        --name ${name}
done

for name in trial1 trial2 trial3 trial4 trial5
do
    python evaluate_baseline.py \
        --tree_sampling BU_X_BU \
        --name ${name}
done

for name in trial1 trial2 trial3 trial4 trial5
do
    python evaluate_baseline.py \
        --tree_sampling BU_BU_BU \
        --name ${name}
done

#####

python evaluate_baseline.py \
    --tree_sampling X_X_RB \
    --name trial1

python evaluate_baseline.py \
    --tree_sampling RB_X_RB \
    --name trial1

python evaluate_baseline.py \
    --tree_sampling RB_RB_RB \
    --name trial1

#####

python evaluate_baseline.py \
    --tree_sampling X_X_LB \
    --name trial1

python evaluate_baseline.py \
    --tree_sampling LB_X_LB \
    --name trial1

python evaluate_baseline.py \
    --tree_sampling LB_LB_LB \
    --name trial1

#####

python evaluate_baseline.py \
    --tree_sampling RB2_RB_RB \
    --name trial1

python evaluate_baseline.py \
    --tree_sampling RB2_RB_LB \
    --name trial1

