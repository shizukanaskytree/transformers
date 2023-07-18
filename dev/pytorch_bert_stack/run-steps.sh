### All steps to do

python pretrain_1_layer.py

SRC_CKPT_FOLDER="pretrained-bert-1-layer/checkpoint-6"
DEST_STACK_CKPT_FOLDER="pretrained-bert-2-layers/checkpoint-6-stack"
python copy_ckpt_files.py --source_folder $SRC_CKPT_FOLDER --destination_folder $DEST_STACK_CKPT_FOLDER

python stack_model.py --ckpt_folder $SRC_CKPT_FOLDER --stack_ckpt_folder $DEST_STACK_CKPT_FOLDER
python stack_optim.py --ckpt_folder $SRC_CKPT_FOLDER --stack_ckpt_folder $DEST_STACK_CKPT_FOLDER

python pretrain_2_layers.py --path_to_checkpoint $DEST_STACK_CKPT_FOLDER

# python copy_ckpt_files.py
