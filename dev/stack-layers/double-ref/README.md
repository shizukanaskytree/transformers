# Docs
1.
https://www.notion.so/xiaofengwu/StackBert-Source-code-for-Efficient-Training-of-BERT-by-Progressively-Stacking-Code-GitHub-048bedc6c00f4a5f97acc36e85808bb4?pvs=4

2.
https://www.notion.so/xiaofengwu/log-related-to-checkpoint-498babbb233e41a8a532f711de1974d3?pvs=4


# How to run double.py and run training?

## 1. run_stack_mlm_L1 so we train layer 1

```
(py38) xiaofeng.wu@Fairfax4way04RTX4090:/home/xiaofeng.wu/prjs/DinkyTrain$ bash run_stack_mlm_L1.sh
```


## 2. run double.sh to prepare checkpoint for 2 layers model

```
(py38) xiaofeng.wu@Fairfax4way04RTX4090:/home/xiaofeng.wu/prjs/DinkyTrain/double$ bash run-double.sh
```

## 3. run run_stack_mlm_L2.sh to train 2 layers model

```
(py38) xiaofeng.wu@Fairfax4way04RTX4090:/home/xiaofeng.wu/prjs/DinkyTrain$ bash run_stack_mlm_L2.sh
```