# export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES=0

CKPT_DIR=/home/xiaofeng.wu/prjs/ckpts/bert-base-uncased
mkdir -p $CKPT_DIR

### NOTE: chang this to a specific checkpoint in the path: /home/xiaofeng.wu/prjs/ckpts/bert-base-uncased
### checkpoints: 1694500, 2023-08-25; 1965000, 2023-08-27 11:58:43; 2528500, 2023-09-02 11:14;
RESUME_FROM_CHECKPOINT=$CKPT_DIR/checkpoint-2528500

TOKENIZER_DIR=skytree/tokenizer-bert-wiki-bookcorpus

### NOTE:
### DO NOT set --model_name_or_path bert-base-uncased when we train from scratch.

python run_mlm.py \
    --config_name bert-base-uncased \
    --tokenizer_name $TOKENIZER_DIR \
    \
    --dataset_name wikitext_bookcorpus \
    \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 40 \
    \
    --learning_rate 1e-4 \
    --warmup_steps 10000 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    \
    --do_train \
    \
    --do_eval False \
    \
    --output_dir $CKPT_DIR \
    --overwrite_output_dir True \
    --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
    --save_total_limit 3 \
    \
    --report_to wandb \
    --run_name bert-base-uncased_$(date +'%Y%m%d_%H%M%S') \
    \
    2>&1 | tee logs/run_mlm_bert_from_scratch_$(date +'%Y%m%d_%H%M%S').log


### cache options
# --evaluation_strategy steps --logging_steps 1 \
# --evaluation_strategy epoch \
#
# --dataset_name wikitext \
# --dataset_config_name wikitext-2-raw-v1 \
