# all steps to run

### baseline
bash glue.sh

### run glue on stacked_bert
bash glue-stack.sh

### get metrics from logs and compare the data with the baseline metrics
python collect_metrics.py
