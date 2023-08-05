from transformers import AutoTokenizer
from datasets import DatasetDict, concatenate_datasets, load_dataset

### 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522

### maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512

### upload tokenizer to hub: https://huggingface.co/skytree/model-growth-tokenizer/tree/main
# remote_hub_ckpts_path = "skytree/tokenizer-bert-wiki-bookcorpus"

### path to save tokenized datasets locally
# tokenized_train_datasets_path = '/data/wxf_tokenized_dataset/wiki_bookcorpus/train'
# tokenized_test_datasets_path = None

bookcorpus = load_dataset("bookcorpus")
print(f"bookcorpus (no split): {bookcorpus}")

### https://huggingface.co/datasets/bookcorpus
### num of rows: 74,004,228
bookcorpus = load_dataset("bookcorpus", split="train")
# print(f"bookcorpus: {bookcorpus}")
### bookcorpus: Dataset({
###     features: ['text'],
###     num_rows: 74_004_228
### })

wiki = load_dataset("wikipedia", "20220301.en", split="train")
# print(f"wiki: {wiki}")
### wiki: Dataset({
###     features: ['id', 'url', 'title', 'text'],
###     num_rows: 6_458_670
### })

wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column
assert bookcorpus.features.type == wiki.features.type
dataset = concatenate_datasets([bookcorpus, wiki])
# print(f"type(concatenate_datasets): {type(dataset)}")
### <class 'datasets.arrow_dataset.Dataset'>

d = DatasetDict({'train': dataset})
# print(f"type(d): {type(d)}")
### class 'datasets.dataset_dict.DatasetDict'
# print(f"d: {d}")
### d: DatasetDict({
###     train: Dataset({
###         features: ['text'],
###         num_rows: 80462898
###     })
### })

def get_training_corpus():
    dataset = d["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["text"]

training_corpus = get_training_corpus()


old_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size)

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''
print('example:\n', example)

tokens = tokenizer.tokenize(example)
print('new tokens:\n', tokens)
print('new tokens len:\n', len(tokens))
print('len(old_tokenizer.tokenize(example)):\n', len(old_tokenizer.tokenize(example)))

################################################################################

### Saving the tokenizer
tokenizer.save_pretrained("tokenizer-bert-wiki-bookcorpus")

################################################################################


