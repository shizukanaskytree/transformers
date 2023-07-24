from datasets import concatenate_datasets, load_dataset


### 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522

### maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512

### batch size for all GPUs
global_batch_size = 128 # 128: 22_416 MiB / 24_564 MiB

"Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
"If smaller than 1, will be interpreted as ratio of total training steps."
save_ckpt_every_X_steps = 50

# DATASET_TYPE="wiki_and_bookcorpus"
DATASET_TYPE="wiki_and_bookcorpus"

if DATASET_TYPE == "wiki_and_bookcorpus":
    ### local tokenizer ckpt path
    ckpts_path = "pretrained-bert-tokenizer-wiki-bookcorpus"

    ### upload tokenizer to hub: https://huggingface.co/skytree/model-growth-tokenizer/tree/main
    remote_hub_ckpts_path = "skytree/model-growth-tokenizer-wiki-bookcorpus"

    ### path to save tokenized datasets
    tokenized_train_datasets_path = '/data/wxf_tokenized_dataset/wiki_bookcorpus/train'
    tokenized_test_datasets_path = None

    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column
    assert bookcorpus.features.type == wiki.features.type
    dataset = concatenate_datasets([bookcorpus, wiki])
    print(f"type of dataset: {type(dataset)}")

    ### We do not split the dataset into training (90%) and testing (10%); API doc: If test_size is None, the value is set to the complement of the train size.
    d = dataset.train_test_split(test_size=None)

    # print(f"type(d): {type(d)}") # <class 'datasets.dataset_dict.DatasetDict'>
    # print(f"d: {d}")
    ### d: DatasetDict({
    ###     train: Dataset({
    ###         features: ['text'],
    ###         num_rows: 60347173
    ###     })
    ###     test: Dataset({
    ###         features: ['text'],
    ###         num_rows: 20115725
    ###     })
    ### })

    # print(f"len of d: {len(d)}") # 2
    # print(f"d['train']: {d['train']}")
    ### d['train']: Dataset({
    ###     features: ['text'],
    ###     num_rows: 60347173
    ### })

    # print(f"📏 len of d['train']: {len(d['train'])}")
    ### num_rows: 60_347_173

else:
    ### local tokenizer ckpt path
    ckpts_path = "pretrained-bert-tokenizer-cc-news"

    ### upload tokenizer to hub: https://huggingface.co/skytree/model-growth-tokenizer-cc-news/tree/main
    remote_hub_ckpts_path = "skytree/model-growth-tokenizer-cc-news"

    ### path to save tokenized datasets
    tokenized_train_datasets_path = '/data/wxf_tokenized_dataset/cc_news/train'
    tokenized_test_datasets_path = '/data/wxf_tokenized_dataset/cc_news/valid'

    dataset = load_dataset("cc_news", split="train[:1%]")
    ### split the dataset into training (90%) and testing (10%)
    d = dataset.train_test_split(test_size=0.1)
