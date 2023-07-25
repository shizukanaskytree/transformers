export TASK_NAME=mrpc

### uses different path for config (config_name), tokenizer (tokenizer_name), model ckpt (model_name_or_path) etc.
CONFIG_NAME=/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/ckpt-bert-wiki-bookcorpus/pretrained-bert-2-layers/checkpoint-101000
### tokenizer ckpt:
TOKENIZER_NAME=/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/ckpt-bert-wiki-bookcorpus
### model ckpt:
MODEL_NAME_OR_PATH=/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/ckpt-bert-wiki-bookcorpus/pretrained-bert-2-layers/checkpoint-101000

python run_glue_stacked_bert.py \
  --config_name         $CONFIG_NAME \
  --tokenizer_name      $TOKENIZER_NAME \
  --model_name_or_path  $MODEL_NAME_OR_PATH \
  --task_name           $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
