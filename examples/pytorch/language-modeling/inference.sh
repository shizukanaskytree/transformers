CKPT_DIR=/home/xiaofeng.wu/prjs/ckpts/bert-base-uncased
TOKENIZER_DIR=skytree/tokenizer-bert-wiki-bookcorpus

python inference.py \
    --model_name_or_path $CKPT_DIR \
    --tokenizer_name $TOKENIZER_DIR \
    --output_dir .