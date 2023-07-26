# WNLI

https://huggingface.co/anirudh21/bert-base-uncased-finetuned-wnli

This model is a fine-tuned version of bert-base-uncased on the glue dataset. It achieves the following results on the evaluation set:

- Loss: 0.6854
- **Accuracy**: 0.5634

```
The following hyperparameters were used during training:

learning_rate: 2e-05
train_batch_size: 16
eval_batch_size: 16
seed: 42
optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
lr_scheduler_type: linear
num_epochs: 5
```

