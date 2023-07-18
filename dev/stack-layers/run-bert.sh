export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export NCCL_P2P_DISABLE=1
python pretrainingbert_stack.py
