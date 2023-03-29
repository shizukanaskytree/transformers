# (base) wxf@seir19:~/atom_prj/transformers/examples/pytorch/language-modeling$ bash run_mlm_bert_no_trainer.sh
python prepare_input.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tokenizer_name bert-base-uncased \
    --output_dir /tmp/test-mlm
