# At one time, only keep either step 1 or step 2 uncommented.
# First step, meta train. Second step use meta train model to "transfer" to target

# Step 1: Meta training

#CUDA_VISIBLE_DEVICES=2,3 python fish_dip.py fewrel_meta --model_name_or_path t5-large --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --percentage 0.01 --subsequent_param_percentage 0.01 --reevaluate_after_steps 100 --train_subset 1.00 

# Step 2: run 4 settings : 1S5W (1 shot 5 way), 5S5W, 1S10W, 5S10W
# you need to adjust the saved_model from previous experiment

saved_model="experiments/fewrel_meta/-fast_tanl_sparse_exp_boosted_t5-large_1_fewrel_meta_percentage-0.01_refresh-perc-0.01_subset-1.00_epochs-Nonereeval_steps-100/-t5-large-ep1-len256-b16-train/episode1"
#
CUDA_VISIBLE_DEVICES=0,1,2,3 python fish_dip.py fewrel_1shot_5way --train_subset 1.00 --model_name_or_path "${saved_model}" --percentage 0.01 --subsequent_param_percentage 0.01  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --reevaluate_after_steps 100
#
#CUDA_VISIBLE_DEVICES=0,1 python fish_dip.py fewrel_5shot_5way --train_subset 1.00 --model_name_or_path "${saved_model}" --percentage 0.01 --subsequent_param_percentage 0.01  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --reevaluate_after_steps 100
#
#CUDA_VISIBLE_DEVICES=0,1 python fish_dip.py fewrel_1shot_10way --train_subset 1.00 --model_name_or_path "${saved_model}" --percentage 0.01 --subsequent_param_percentage 0.01  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --reevaluate_after_steps 100
#
#CUDA_VISIBLE_DEVICES=0,1 python fish_dip.py fewrel_5shot_10way --train_subset 1.00 --model_name_or_path "${saved_model}" --percentage 0.01 --subsequent_param_percentage 0.01  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --reevaluate_after_steps 100