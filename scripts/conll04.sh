#!/bin/bash

# Define arrays for the train subsets and corresponding num_train_epochs
train_subsets=(0.01 0.05 0.1)
num_train_epochs=(1000 800 800)

# Loop through the arrays and execute the command for each combination
for i in "${!train_subsets[@]}"; do
    subset="${train_subsets[i]}"
    epochs="${num_train_epochs[i]}"
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 python fish_dip.py conll04 \
    --model_name_or_path t5-large \
    --train_subset "$subset" \
    --num_train_epochs "$epochs" \
    --episodes 1-3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --percentage 0.01 \
    --subsequent_param_percentage 0.01 \
    --reevaluate_after_steps 100 \
    --do_train \
    --verbose_results
done
