export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm \
    --overwrite_output_dir
