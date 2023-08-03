from datasets import DatasetDict, concatenate_datasets, load_dataset


### 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522

### maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512

### 1 encoder: global_batch_size 128: 22_416 MiB / 24_564 MiB
### 2 encoder: global_batch_size 64:  9_242 MiB / 24_564 MiB
### 12 encoder: global_batch_size 64: 18_424 MiB / 24_564 MiB
global_batch_size = 64 # batch size for all GPUs

"Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
"If smaller than 1, will be interpreted as ratio of total training steps."
save_ckpt_every_X_steps = 1000

# DATASET_TYPE="cc_news"
DATASET_TYPE="wiki_and_bookcorpus"
if DATASET_TYPE == "wiki_and_bookcorpus":
    ### local tokenizer and model ckpt path
    tokenizer_ckpt_path = "ckpt-bert-wiki-bookcorpus-tokenizer"
    model_ckpts_path = "ckpt-bert-wiki-bookcorpus-model"

    ### upload tokenizer to hub: https://huggingface.co/skytree/model-growth-tokenizer/tree/main
    remote_hub_ckpts_path = "skytree/model-growth-tokenizer-wiki-bookcorpus"

    ### path to save tokenized datasets locally
    tokenized_train_datasets_path = '/data/wxf_tokenized_dataset/wiki_bookcorpus/train'
    tokenized_test_datasets_path = None

    ### https://huggingface.co/datasets/bookcorpus
    bookcorpus = load_dataset("bookcorpus", split="train")
    ###
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column
    assert bookcorpus.features.type == wiki.features.type
    dataset = concatenate_datasets([bookcorpus, wiki])
    # print(f"type(concatenate_datasets): {type(dataset)}")
    ### <class 'datasets.arrow_dataset.Dataset'>

    d = DatasetDict({'train': dataset})
    # print(f"type(d): {type(d)}")
    ### class 'datasets.dataset_dict.DatasetDict'

    # print(f"d['train']: {d['train']}")
    ### d['train']: Dataset({
    ###     features: ['text'],
    ###     num_rows: 80_462_898
    ### })

    #---------------------------------------------------------------------------
    ### Unless we want to split the dataset into split

    ### We do NOT split the dataset into training (90%) and testing (10%); API doc: If test_size is None, the value is set to the complement of the train size.
    # d = dataset.train_test_split(test_size=0.1) # 10% for testing

    # print(f"type(d) after split: {type(d)}")
    ### <class 'datasets.dataset_dict.DatasetDict'>

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
    ### local tokenizer and model ckpt path
    tokenizer_ckpt_path = "ckpt-bert-cc-news-tokenizer"
    model_ckpts_path = "ckpt-bert-cc-news-model"

    ### upload tokenizer to hub: https://huggingface.co/skytree/model-growth-tokenizer-cc-news/tree/main
    remote_hub_ckpts_path = "skytree/model-growth-tokenizer-cc-news"

    ### path to save tokenized datasets locally
    tokenized_train_datasets_path = '/data/wxf_tokenized_dataset/cc_news/train'
    tokenized_test_datasets_path = '/data/wxf_tokenized_dataset/cc_news/valid'

    dataset = load_dataset("cc_news", split="train[:1%]")
    ### split the dataset into training (90%) and testing (10%)
    d = dataset.train_test_split(test_size=0.1)
