### each run has a different log/2023... folder
RUN_LOGS=logs/$(date +'%Y%m%d_%H%M%S')

mkdir -p $RUN_LOGS

### uses different path for config (config_name), tokenizer (tokenizer_name), model ckpt (model_name_or_path) etc.
CONFIG_NAME=/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/ckpt-bert-wiki-bookcorpus/pretrained-bert-12-layers/checkpoint-200000
### tokenizer ckpt:
TOKENIZER_NAME=/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/ckpt-bert-wiki-bookcorpus
### model ckpt:
MODEL_NAME_OR_PATH=/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/ckpt-bert-wiki-bookcorpus/pretrained-bert-12-layers/checkpoint-200000

TASKS=("cola" "sst2" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte" "wnli")

MAX_SEQ_LENGTH=128
PER_DEVICE_TRAIN_BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=3

for TASK_NAME in "${TASKS[@]}"
do
    python run_glue_stacked_bert.py \
        --config_name $CONFIG_NAME \
        --tokenizer_name $TOKENIZER_NAME \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --max_seq_length $MAX_SEQ_LENGTH \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --output_dir "/tmp/$TASK_NAME/" \
        --overwrite_output_dir 2>&1 | tee ${RUN_LOGS}/${TASK_NAME}_glue_stacked_bert.log
done
