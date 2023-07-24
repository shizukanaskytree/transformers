### All steps to do

mkdir -p logs

### train tokenizer on wiki and book corpus
python train_tokenizer_by_dataset.py 2>&1 | tee logs/log_train_tokenizer_by_dataset.log

### upload tokenizer to hf hub
python upload_tokenizer.py 2>&1 | tee logs/log_upload_tokenizer.log

### tokenize datasets and save to local disk
python save_tokenized_datasets.py 2>&1 | tee logs/log_save_tokenized_datasets.log

#-------------------------------------------------------------------------------

python pretrain_bert_wiki_book.py --num_hidden_layers 1 2>&1 | tee logs/log_pretrain_bert_wiki_book_1.log

#-------------------------------------------------------------------------------

SRC_CKPT_FOLDER="pretrained-bert-1-layers"
DEST_STACK_CKPT_FOLDER="pretrained-bert-2-layers"
python copy_ckpt_files.py --src_ckpts_folder $SRC_CKPT_FOLDER --dest_ckpt_folder $DEST_STACK_CKPT_FOLDER
python stack_model.py --to_be_copied_layer_num 0 --src_ckpts_folder $SRC_CKPT_FOLDER --stacked_ckpts_folder $DEST_STACK_CKPT_FOLDER
python stack_optim.py --to_be_copied_layer_num 0 --src_ckpts_folder $SRC_CKPT_FOLDER --stacked_ckpts_folder $DEST_STACK_CKPT_FOLDER

### continue training
python pretrain_bert.py --num_hidden_layers 2 --path_to_ckpts $DEST_STACK_CKPT_FOLDER

#----------------------------------------

SRC_CKPT_FOLDER="pretrained-bert-2-layers"
DEST_STACK_CKPT_FOLDER="pretrained-bert-3-layers"
python copy_ckpt_files.py --src_ckpts_folder $SRC_CKPT_FOLDER --dest_ckpt_folder $DEST_STACK_CKPT_FOLDER
python stack_model.py --to_be_copied_layer_num 1 --src_ckpts_folder $SRC_CKPT_FOLDER --stacked_ckpts_folder $DEST_STACK_CKPT_FOLDER
python stack_optim.py --to_be_copied_layer_num 1 --src_ckpts_folder $SRC_CKPT_FOLDER --stacked_ckpts_folder $DEST_STACK_CKPT_FOLDER

### continue training
python pretrain_bert.py --num_hidden_layers 3 --path_to_ckpts $DEST_STACK_CKPT_FOLDER
