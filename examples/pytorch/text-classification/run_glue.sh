export TASK_NAME=mrpc

# export NCCL_SOCKET_IFNAME=lo
# export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0

# export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
# export CUDA_LAUNCH_BLOCKING=1

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
