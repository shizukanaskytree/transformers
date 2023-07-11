#!/bin/bash

########################
# NOT working
########################

export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

shard_dir="/home/xiaofeng.wu/prjs/transformers/examples/pytorch/language-modeling/wikibook_fairseq_format"
train_files=""
valid_files=""

for shard in {0..15}; do
    train_files+=",$shard_dir/bin-shard${shard}-8/train.bin"
    valid_files+=",$shard_dir/bin-shard${shard}-8/valid.bin"
done

train_files=${train_files:1}  # Remove the leading comma
valid_files=${valid_files:1}  # Remove the leading comma

python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file $train_files \
    --validation_file $valid_files \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm \
    --overwrite_output_dir

    # --vocab_file $shard_dir/bin-shard0-8/dict.txt \

# python run_mlm.py \
#     --model_name_or_path roberta-base \
#     # --dataset_name wikitext \
#     # --dataset_config_name wikitext-2-raw-v1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --num_train_epochs 1 \
#     --do_train \
#     --do_eval \
#     --output_dir /tmp/test-mlm \
#     --overwrite_output_dir


# python run_mlm.py \
#     --model_name_or_path roberta-base \
#     --train_file wikibook_fairseq_format \
#     --validation_file  wikibook_fairseq_format\
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --output_dir /tmp/test-mlm
