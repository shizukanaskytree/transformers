# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()
from itertools import chain

from global_constants import d, max_length, remote_hub_ckpts_path, tokenized_datasets_path

from transformers import BertTokenizerFast


for t in d["train"]["text"][:2]:
    print(t)
    print("="*50)

### whether to truncate
truncate_longer_samples = False

### Since we have trained the tokenizer, then load BertTokenizerFast as from remote.
tokenizer = BertTokenizerFast.from_pretrained(remote_hub_ckpts_path)

def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
    """Mapping function to tokenize the sentences passed without truncation"""
    return tokenizer(examples["text"], return_special_tokens_mask=True)

### the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

### tokenizing the train dataset
train_dataset = d["train"].map(encode, batched=True)
### tokenizing the testing dataset
# test_dataset = d["test"].map(encode, batched=True)

if truncate_longer_samples:
    ### remove other columns and set input_ids and attention_mask as PyTorch tensors
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    ### remove other columns, and remain them as Python lists
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    # test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

### Main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length.
### grabbed from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
def group_texts(examples):
    ### Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    ### We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    ### Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

### Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
### remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
### might be slower to preprocess.
###
### To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
### https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
if not truncate_longer_samples:
    train_dataset = train_dataset.map(group_texts, batched=True, desc=f"Grouping texts in chunks of {max_length}")
    # test_dataset = test_dataset.map(group_texts, batched=True, desc=f"Grouping texts in chunks of {max_length}")
    ### convert them from lists to torch tensors
    train_dataset.set_format("torch")
    # test_dataset.set_format("torch")

# print(f"len of train_dataset: {len(train_dataset)}, len of test_dataset: {len(test_dataset)}")
# print(f"train_dataset: {train_dataset}")
### train_dataset: Dataset({
###     features: ['title', 'text', 'domain', 'date', 'description', 'url', 'image_url', 'input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'],
###     num_rows: 691
### })

### Save your tokenized dataset to /data/wxf_tokenized_dataset
train_dataset.save_to_disk(tokenized_datasets_path)
# test_dataset.save_to_disk(tokenized_datasets_path)
