#!/bin/bash

# Define arrays for the train subsets and corresponding num_train_epochs
train_subsets=(0.005 0.01 0.05)
num_train_epochs=(500 400 400)

# Loop through the arrays and execute the command for each combination
for i in "${!train_subsets[@]}"; do
    subset="${train_subsets[i]}"
    epochs="${num_train_epochs[i]}"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fish_dip.py multi_woz \
    --model_name_or_path t5-large \
    --train_subset "$subset" \
    --num_train_epochs "$epochs" \
    --episodes 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --percentage 0.01 \
    --do_train \
    --subsequent_param_percentage 0.01 \
    --reevaluate_after_steps 100 \
    --verbose_results 
done
