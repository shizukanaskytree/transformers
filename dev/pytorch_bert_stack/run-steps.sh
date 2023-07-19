### All steps to do
python pretrain_bert.py --num_hidden_layers 1

#----------------------------------------

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

