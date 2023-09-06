# BERT Glue result

`/home/xiaofeng.wu/prjs/ckpts/bert-base-uncased/checkpoint-2528000`

todo

```
(py311) xiaofeng.wu@Fairfax4way05RTX4090:/home/xiaofeng.wu/prjs/transformers/examples/pytorch/language-modeling$ python collect_metrics.py --log_dir "./logs/glue-20230902_110318"

Task    Metric            Value (%)    Baseline (%)  Difference (%)
------  --------------  -----------  --------------  ----------------
CoLA    Matthews_corr         31.47           52.1   -20.63
SST-2   Accuracy              92.55           93.5   -0.95
MRPC    Accuracy              69.61                  N/A
MRPC    F1                    81.82           88.9   -7.08
STS-B   Pearson_corr          74.43           85.8   -11.37
STS-B   Spearman_corr         75.48                  N/A
QQP     Accuracy              88.51           71.2   17.31
QQP     F1                    84.68                  N/A
MNLI    Matched_acc           82.69           84.6   -1.91
MNLI    Mismatched_acc        83.55           83.4   0.15
QNLI    Accuracy              89.71           90.5   -0.79
RTE     Accuracy              49.1            66.4   -17.3
WNLI    Accuracy              42.25           56.34  -14.09
```

This seems much normal...

===


`(py311) xiaofeng.wu@Fairfax4way05RTX4090:/home/xiaofeng.wu/prjs/transformers/examples/pytorch/language-modeling$ python collect_metrics.py --log_dir "./logs/glue-20230901_231338"`


`/home/xiaofeng.wu/prjs/ckpts/bert-base-uncased/checkpoint-2528500`

```
Task    Metric            Value (%)    Baseline (%)  Difference (%)
------  --------------  -----------  --------------  ----------------
CoLA    Matthews_corr          9.33           52.1   -42.77
SST-2   Accuracy              89.79           93.5   -3.71
MRPC    Accuracy              74.26                  N/A
MRPC    F1                    82.59           88.9   -6.31
STS-B   Pearson_corr          74.56           85.8   -11.24
STS-B   Spearman_corr         75.24                  N/A
QQP     Accuracy              89.66           71.2   18.46
QQP     F1                    85.95                  N/A
MNLI    Matched_acc           56.86           84.6   -27.74
MNLI    Mismatched_acc        59.77           83.4   -23.63
QNLI    Accuracy              89.55           90.5   -0.95
RTE     Accuracy              51.99           66.4   -14.41
WNLI    Accuracy              53.52           56.34  -2.82
```


```
./logs/glue-20230823_032808"
Task    Metric            Value (%)    Baseline (%)  Difference (%)
------  --------------  -----------  --------------  ----------------
CoLA    Matthews_corr         32.64           52.1   -19.46
SST-2   Accuracy              90.71           93.5   -2.79
MRPC    Accuracy              73.28                  N/A
MRPC    F1                    82.62           88.9   -6.28
STS-B   Pearson_corr          58.06           85.8   -27.74
STS-B   Spearman_corr         60.99                  N/A
QQP     Accuracy              89.61           71.2   18.41
QQP     F1                    85.91                  N/A
MNLI    Matched_acc           81.67           84.6   -2.93
MNLI    Mismatched_acc        82.41           83.4   -0.99
QNLI    Accuracy              88.98           90.5   -1.52
RTE     Accuracy              52.71           66.4   -13.69
WNLI    Accuracy              43.66           56.34  -12.68
```

