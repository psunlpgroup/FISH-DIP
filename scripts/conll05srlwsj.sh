#!/bin/bash

# Define arrays for the train subsets and corresponding num_train_epochs
train_subsets=(0.005 0.01 0.05)
num_train_epochs=(200 160 160)

# Loop through the arrays and execute the command for each combination
for i in "${!train_subsets[@]}"; do
    subset="${train_subsets[i]}"
    epochs="${num_train_epochs[i]}"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python fish_dip.py conll05_srl \
    --train_subset "$subset" \
    --num_train_epochs "$epochs" \
    --model_name_or_path t5-large \
    --episodes 1-3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --percentage 0.01 \
    --subsequent_param_percentage 0.01 \
    --reevaluate_after_steps 100 \
    --verbose_results
done
