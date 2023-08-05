# export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES=0

CKPT_DIR=/home/xiaofeng.wu/prjs/ckpts/bert-base-uncased
mkdir -p $CKPT_DIR
TOKENIZER_DIR=skytree/tokenizer-bert-wiki-bookcorpus

### NOTE:
### DO NOT set --model_name_or_path bert-base-uncased when we train from scratch.

python run_mlm.py \
    --config_name bert-base-uncased \
    --tokenizer_name $TOKENIZER_DIR \
    \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 10 \
    --do_train \
    \
    --do_eval \
    --evaluation_strategy steps --logging_steps 1 \
    \
    --output_dir $CKPT_DIR \
    --overwrite_output_dir True \
    --save_total_limit 3 \
    \
    --report_to wandb \
    --run_name bert-base-uncased_$(date +'%Y%m%d_%H%M%S') \
    2>&1 | tee logs/run_mlm_bert_from_scratch_$(date +'%Y%m%d_%H%M%S').log


# --evaluation_strategy epoch \
# --resume_from_checkpoint $CKPT_DIR \
