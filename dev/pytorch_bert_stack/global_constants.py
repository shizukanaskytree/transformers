from datasets import concatenate_datasets, load_dataset


### local tokenizer ckpt path
ckpts_path = "pretrained-bert-tokenizer"

### path to upload tokenizer to hub
remote_hub_ckpts_path = "skytree/model-growth-tokenizer"

### path to save tokenized datasets
tokenized_datasets_path = '/data/wxf_tokenized_dataset'

### 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522

### maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512

global_batch_size = 64

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column
assert bookcorpus.features.type == wiki.features.type
dataset = concatenate_datasets([bookcorpus, wiki])

### We do not split the dataset into training (90%) and testing (10%); API doc: If test_size is None, the value is set to the complement of the train size.
d = dataset.train_test_split(test_size=None)
