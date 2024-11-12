# To calculate the perplexity (PPL) of the model without fine-tuning, you can
# utilize the evaluation-only functionality provided in the script. Here’s how
# you can modify and run the script:

python run_clm.py \
    --model_name_or_path openai-community/gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_eval \
    --output_dir /tmp/test-clm-wo-finetuning

