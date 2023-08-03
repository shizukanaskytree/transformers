# Reported Metrics

## Baseline

for different task, extract different metric according to the table

```
| Task | Metric | Result | Training time |
| --- | --- | --- | --- |
| CoLA | Matthews corr | 56.53 | 3:17 |
| SST-2 | Accuracy | 92.32 | 26:06 |
| MRPC | F1/Accuracy | 88.85/84.07 | 2:21 |
| STS-B | Pearson/Spearman corr. | 88.64/88.48 | 2:13 |
| QQP | Accuracy/F1 | 90.71/87.49 | 2:22:26 |
| MNLI | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23 |
| QNLI | Accuracy | 90.66 | 40:57 |
| RTE | Accuracy | 65.70 | 57 |
| WNLI | Accuracy | 56.34 | 24 |
```

Note:
BERT was originally pre-trained on 1 Million Steps (1_000_000) with a global batch size of 256 : We train with batch size of 256 sequences (256 sequences * 512 tokens = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus.


## 1_000_000 steps


## 450_000 steps



## 400_000 steps

20230802_220031

```
Task    Metric            Value (%)    Baseline (%)    Difference (%)
------  --------------  -----------  --------------  ----------------
CoLA    Matthews_corr          9.29           56.53            -47.24
SST-2   Accuracy              81.31           92.32            -11.01
MRPC    Accuracy              68.38           84.07            -15.69
MRPC    F1                    79.75           88.85             -9.1
STS-B   Pearson_corr          14.1            88.64            -74.54
STS-B   Spearman_corr         11.23           88.48            -77.25
QQP     Accuracy              83.24           90.71             -7.47
QQP     F1                    78.77           87.49             -8.72
MNLI    Matched_acc           65.69           83.91            -18.22
MNLI    Mismatched_acc        66.33           84.1             -17.77
QNLI    Accuracy              76.3            90.66            -14.36
RTE     Accuracy              47.29           65.7             -18.41
WNLI    Accuracy              32.39           56.34            -23.95
```



## 300_000 steps

```
Task    Metric            Value (%)    Baseline (%)    Difference (%)
------  --------------  -----------  --------------  ----------------
CoLA    Matthews_corr         12.21           56.53            -44.32
SST-2   Accuracy              80.16           92.32            -12.16
MRPC    Accuracy              68.14           84.07            -15.93
MRPC    F1                    79.56           88.85             -9.29
STS-B   Pearson_corr          14.04           88.64            -74.6
STS-B   Spearman_corr         11.9            88.48            -76.58
QQP     Accuracy              83.76           90.71             -6.95
QQP     F1                    79.08           87.49             -8.41
MNLI    Matched_acc           65.86           83.91            -18.05
MNLI    Mismatched_acc        66.51           84.1             -17.59
QNLI    Accuracy              76.17           90.66            -14.49
RTE     Accuracy              47.29           65.7             -18.41
WNLI    Accuracy              30.99           56.34            -25.35
```


## 200_000 steps

```
Task    Metric            Value (%)    Baseline (%)    Difference (%)
------  --------------  -----------  --------------  ----------------
CoLA    Matthews_corr         12.09           56.53            -44.44
SST-2   Accuracy              80.39           92.32            -11.93
MRPC    Accuracy              68.87           84.07            -15.2
MRPC    F1                    80.19           88.85             -8.66
STS-B   Pearson_corr          10.71           88.64            -77.93
STS-B   Spearman_corr          8.25           88.48            -80.23
QQP     Accuracy              83.71           90.71             -7
QQP     F1                    78.98           87.49             -8.51
MNLI    Matched_acc           65.65           83.91            -18.26
MNLI    Mismatched_acc        66.32           84.1             -17.78
QNLI    Accuracy              76.66           90.66            -14
RTE     Accuracy              50.9            65.7             -14.8
WNLI    Accuracy              29.58           56.34            -26.76
```


## GLUE logs

```
for cola:
  epoch                     =        3.0
  eval_loss                 =     0.6403
  eval_matthews_correlation =    -0.0259
  eval_runtime              = 0:00:00.35
  eval_samples              =       1043
  eval_samples_per_second   =    2904.77
  eval_steps_per_second     =    364.837

for mrpc:
  epoch                   =        3.0
  eval_accuracy           =     0.6838
  eval_combined_score     =      0.739
  eval_f1                 =     0.7943
  eval_loss               =     0.5997
  eval_runtime            = 0:00:00.14
  eval_samples            =        408
  eval_samples_per_second =   2812.369
  eval_steps_per_second   =    351.546

for qnli:
  epoch                   =        3.0
  eval_accuracy           =     0.6394
  eval_loss               =     0.6461
  eval_runtime            = 0:00:01.88
  eval_samples            =       5463
  eval_samples_per_second =   2899.858
  eval_steps_per_second   =    362.549

for qqp:
  epoch                   =        3.0
  eval_accuracy           =     0.8192
  eval_combined_score     =     0.7849
  eval_f1                 =     0.7505
  eval_loss               =     0.4121
  eval_runtime            = 0:00:13.84
  eval_samples            =      40430
  eval_samples_per_second =   2920.869
  eval_steps_per_second   =    365.127

for sst2:
  epoch                   =        3.0
  eval_accuracy           =     0.8234
  eval_loss               =     0.5013
  eval_runtime            = 0:00:00.30
  eval_samples            =        872
  eval_samples_per_second =   2882.706
  eval_steps_per_second   =    360.338

for stsb:
  epoch                   =        3.0
  eval_combined_score     =       0.12
  eval_loss               =     2.6235
  eval_pearson            =     0.1268
  eval_runtime            = 0:00:00.51
  eval_samples            =       1500
  eval_samples_per_second =   2937.983
  eval_spearmanr          =     0.1132
  eval_steps_per_second   =    368.227

for rte:
  epoch                   =        3.0
  eval_accuracy           =     0.5126
  eval_loss               =     0.7079
  eval_runtime            = 0:00:00.09
  eval_samples            =        277
  eval_samples_per_second =   2838.635
  eval_steps_per_second   =    358.672

for wnli:
  epoch                   =        3.0
  eval_accuracy           =     0.3662
  eval_loss               =     0.7808
  eval_runtime            = 0:00:00.02
  eval_samples            =         71
  eval_samples_per_second =   2595.689
  eval_steps_per_second   =    329.031
```


```
(py311) xiaofeng.wu@Fairfax4way05RTX4090:/home/xiaofeng.wu/prjs/transformers/dev/eval_bert$ python collect_metrics.py
Task    Metric            Value (%)    Baseline (%)    Difference (%)
------  --------------  -----------  --------------  ----------------
CoLA    Matthews_corr         -2.07           56.53            -58.6
SST-2   Accuracy              82.34           92.32             -9.98
MRPC    Accuracy              68.38           84.07            -15.69
MRPC    F1                    79.43           88.85             -9.42
STS-B   Pearson_corr          12.68           88.64            -75.96
STS-B   Spearman_corr         11.32           88.48            -77.16
QQP     Accuracy              81.92           90.71             -8.79
QQP     F1                    75.05           87.49            -12.44
MNLI    Matched_acc           60.09           83.91            -23.82
MNLI    Mismatched_acc        60.48           84.1             -23.62
QNLI    Accuracy              63.94           90.66            -26.72
RTE     Accuracy              51.26           65.7             -14.44
WNLI    Accuracy              39.44           56.34            -16.9
```