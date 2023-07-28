from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
# print(f"wiki: {wiki}")
### wiki: Dataset({
###     features: ['id', 'url', 'title', 'text'],
###     num_rows: 6458670
### })

wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

assert bookcorpus.features.type == wiki.features.type
raw_datasets = concatenate_datasets([bookcorpus, wiki])

### downloaded:
### (py311) xiaofeng.wu@Fairfax4way04RTX4090:/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/tests/dataset_concat_wiki_book$ python concat_wiki_book_dataset.py
### Found cached dataset bookcorpus (/home/xiaofeng.wu/.cache/huggingface/datasets/bookcorpus/plain_text/1.0.0/eddee3cae1cc263a431aa98207d4d27fd8a73b0a9742f692af0e6c65afa4d75f)
### Downloading and preparing dataset wikipedia/20220301.en to /home/xiaofeng.wu/.cache/huggingface/datasets/wikipedia/20220301.en/2.0.0/aa542ed919df55cc5d3347f42dd4521d05ca68751f50dbc32bae2a7f1e167559...
### Downloading: 100%|███████████████████████████████████████████████████████████████████████████████████| 15.3k/15.3k [00:00<00:00, 68.9MB/s]
### Downloading: 100%|███████████████████████████████████████████████████████████████████████████████████| 20.3G/20.3G [09:23<00:00, 36.0MB/s]
### Dataset wikipedia downloaded and prepared to /home/xiaofeng.wu/.cache/huggingface/datasets/wikipedia/20220301.en/2.0.0/aa542ed919df55cc5d3347f42dd4521d05ca68751f50dbc32bae2a7f1e167559. Subsequent calls will reuse this data.
### (py311) xiaofeng.wu@Fairfax4way04RTX4090:/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/tests/dataset_concat_wiki_book$

print(raw_datasets)