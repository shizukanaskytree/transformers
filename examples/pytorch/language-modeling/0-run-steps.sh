mkdir -p logs

#-------------------------------------------------------------------------------

### train a tokenizer from an old one
bash train-tokenizer/run-steps.sh

#-------------------------------------------------------------------------------

### run bert from scratch
bash train-bert-base-uncase-from-scratch.sh

### inference the bert model
# bash inference.sh # under construction

### use pipeline api to do inference
python pipeline-bert.py

#-------------------------------------------------------------------------------

### glue evaluation
bash eval_glue_bert.sh

