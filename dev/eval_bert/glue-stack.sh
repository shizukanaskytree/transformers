export TASK_NAME=mrpc

### tokenizer ckpt:
### dev/pytorch_bert_stack/pretrained-bert-1-layers

### model ckpt:
### dev/pytorch_bert_stack/pretrained-bert-1-layers/checkpoint-30

python run_glue_stacked_bert.py \
  --model_name_or_path /home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/pretrained-bert-1-layers/checkpoint-28 \
  --config_name /home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/pretrained-bert-1-layers/checkpoint-28 \
  --tokenizer_name /home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/pretrained-bert-1-layers \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
