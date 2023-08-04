# all steps to run

### baseline
bash glue.sh

### run glue on stacked_bert
bash glue-stack.sh

### get metrics from logs and compare the data with the baseline metrics after bash glue-stack.sh
python collect_metrics.py --log_dir "/home/xiaofeng.wu/prjs/transformers/dev/eval_bert/logs/20230803_223226"
