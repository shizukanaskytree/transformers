# wiki and bookcorpus dataset

Tutorial:

https://github.com/philschmid/deep-learning-habana-huggingface/blob/master/pre-training/pre-training-bert.ipynb

```py
from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

assert bookcorpus.features.type == wiki.features.type
raw_datasets = concatenate_datasets([bookcorpus, wiki])
```