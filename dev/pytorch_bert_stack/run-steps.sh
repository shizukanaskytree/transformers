### All steps to do
### terminal logs
mkdir -p logs

### train tokenizer on wiki and book corpus
python train_tokenizer_by_dataset.py 2>&1 | tee logs/log_train_tokenizer_by_dataset_$(date +'%Y%m%d_%H%M%S').log

### upload tokenizer to hf hub
python upload_tokenizer.py 2>&1 | tee logs/log_upload_tokenizer.log

### Get Permission '/data/wxf_tokenized_dataset'
sudo chown -R $USER: /data/wxf_tokenized_dataset

### tokenize datasets and save to local disk
python save_tokenized_datasets.py 2>&1 | tee logs/log_save_tokenized_datasets_$(date +'%Y%m%d_%H%M%S').log

#-------------------------------------------------------------------------------

DEST_STACKED_MODEL_CKPT_DIR="ckpt-bert-wiki-bookcorpus/pretrained-bert-1-layers"
python pretrain_bert_wiki_book.py --num_hidden_layers 1 --path_to_ckpts $DEST_STACKED_MODEL_CKPT_DIR 2>&1 | tee logs/log_pretrain_bert_wiki_book_1_$(date +'%Y%m%d_%H%M%S').log

#-------------------------------------------------------------------------------

### eval the model with 1 encoder layer: go to dev/eval_bert/run-steps.sh

#-------------------------------------------------------------------------------

SRC_CKPT_FOLDER="ckpt-bert-wiki-bookcorpus/pretrained-bert-1-layers"
DEST_STACKED_MODEL_CKPT_DIR="ckpt-bert-wiki-bookcorpus/pretrained-bert-2-layers"
python copy_ckpt_files.py --src_ckpts_folder $SRC_CKPT_FOLDER --dest_ckpt_folder $DEST_STACKED_MODEL_CKPT_DIR
python stack_model.py --to_be_copied_layer_num 0 --src_ckpts_folder $SRC_CKPT_FOLDER --stacked_ckpts_folder $DEST_STACKED_MODEL_CKPT_DIR
python stack_optim.py --to_be_copied_layer_num 0 --src_ckpts_folder $SRC_CKPT_FOLDER --stacked_ckpts_folder $DEST_STACKED_MODEL_CKPT_DIR

### continue training
python pretrain_bert_wiki_book.py --num_hidden_layers 2 --path_to_ckpts $DEST_STACKED_MODEL_CKPT_DIR 2>&1 | tee logs/log_pretrain_bert_wiki_book_2_$(date +'%Y%m%d_%H%M%S').log

#-------------------------------------------------------------------------------

### eval the model with 2 encoder layer: go to dev/eval_bert/run-steps.sh

#-------------------------------------------------------------------------------

SRC_CKPT_FOLDER="pretrained-bert-2-layers"
DEST_STACKED_MODEL_CKPT_DIR="pretrained-bert-3-layers"
python copy_ckpt_files.py --src_ckpts_folder $SRC_CKPT_FOLDER --dest_ckpt_folder $DEST_STACKED_MODEL_CKPT_DIR
python stack_model.py --to_be_copied_layer_num 1 --src_ckpts_folder $SRC_CKPT_FOLDER --stacked_ckpts_folder $DEST_STACKED_MODEL_CKPT_DIR
python stack_optim.py --to_be_copied_layer_num 1 --src_ckpts_folder $SRC_CKPT_FOLDER --stacked_ckpts_folder $DEST_STACKED_MODEL_CKPT_DIR

### continue training
python pretrain_bert.py --num_hidden_layers 3 --path_to_ckpts $DEST_STACKED_MODEL_CKPT_DIR  2>&1 | tee logs/log_pretrain_bert_wiki_book_3_$(date +'%Y%m%d_%H%M%S').log
