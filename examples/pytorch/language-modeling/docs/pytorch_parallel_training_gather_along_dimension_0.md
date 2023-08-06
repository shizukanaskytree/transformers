# Discussion and solution

https://github.com/huggingface/transformers/issues/14128#issuecomment-998199941

log snippet with issues:

```bash
/home/xiaofeng.wu/anaconda3/envs/py311/lib/python3.11/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
100%|██████████| 88/88 [01:01<00:00,  1.64it/s][INFO|trainer.py:1977] 2023-08-05 05:10:26,977 >>

Training completed. Do not forget to share your model on huggingface.co/models =)
```

all log: https://gist.github.com/shizukanaskytree/d6c1f9596581b5f29afd5453abbdb21a
