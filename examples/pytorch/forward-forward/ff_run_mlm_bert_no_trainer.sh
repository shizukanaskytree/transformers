# (base) wxf@seir19:~/atom_prj/transformers/examples/pytorch/language-modeling$ bash run_mlm_bert_no_trainer.sh
python ff_run_mlm_no_trainer_no_accelerator.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path bert-base-uncased \
    --output_dir /tmp/test-mlm