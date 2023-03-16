python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm \
    --overwrite_output_dir \
    --report_to tensorboard

# notion note: https://www.notion.so/xiaofengwu/x-853e064f7dbd4e4e82ac8a9705865fce?pvs=4
