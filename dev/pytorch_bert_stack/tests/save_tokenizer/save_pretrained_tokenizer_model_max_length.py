# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()
import os
import argparse
import json
import glob

from datasets import load_dataset
from transformers import BertForMaskedLM, BertConfig, DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments, BertTokenizerFast, pipeline
from tokenizers import BertWordPieceTokenizer

os.environ['WANDB_MODE'] = 'offline'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Copy checkpoint files from source to destination folder.')
parser.add_argument('--path_to_ckpts', default="", required=False, help='Path to the checkpoint folder')
parser.add_argument('--num_hidden_layers', required=True, type=int, help='The number of layers in the stacked model, e.g., 2')
args = parser.parse_args()

# download and prepare cc_news dataset, we select 1% for fast demo, use
# split="train" for all training dataset
dataset = load_dataset("cc_news", split="train[:1%]")

# split the dataset into training (90%) and testing (10%)
d = dataset.train_test_split(test_size=0.1)
# print(d["train"], d["test"])

# for t in d["train"]["text"][:3]:
#     print(t)
#     print("="*50)

# if you have your custom dataset
# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="path/to/data.txt",
#     block_size=64,
# )

# or if you have huge custom dataset separated into files
# load the splitted files
# files = ["train1.txt", "train2.txt"] # train3.txt, etc.
# dataset = load_dataset("text", data_files=files, split="train")

# if you want to train the tokenizer from scratch (especially if you have custom
# dataset loaded as datasets object), then run this cell to save it as files
# but if you already have your custom data as text files, there is no point using this
def dataset_to_text(dataset, output_filename="data.txt"):
    """Utility function to save dataset text to disk,
    useful for using the texts to train the tokenizer
    (as the tokenizer accepts files)"""
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)

# save the training set to train.txt
dataset_to_text(d["train"], "train.txt")
# save the testing set to test.txt
dataset_to_text(d["test"], "test.txt")

special_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]

# if you want to train the tokenizer on both sets
# files = ["train.txt", "test.txt"]
# training the tokenizer on the training set
files = ["train.txt"]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512
# whether to truncate
truncate_longer_samples = False

# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()
# train the tokenizer
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
# enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

ckpts_path = f"pretrained-bert-{args.num_hidden_layers}-layers"
if not os.path.isdir(ckpts_path):
    os.mkdir(ckpts_path)

###################################
# save the tokenizer
# tokenizer.save_pretrained(ckpts_path)
###################################
### error:
# Traceback (most recent call last):
#   File "/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack/tests/save_tokenizer/save_pretrained_tokenizer.py", line 89, in <module>
#     tokenizer.save_pretrained(ckpts_path)
#     ^^^^^^^^^^^^^^^^^^^^^^^^^
# AttributeError: 'BertWordPieceTokenizer' object has no attribute 'save_pretrained'

### correct:
###################################
new_tokenizer = BertTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=max_length,
)

new_tokenizer.save_pretrained(ckpts_path)
print(f"saved tokenizer to {ckpts_path}")
###################################

#-------------------------------------------------------------------------------


# # dumping some of the tokenizer config to config file,
# # including special tokens, whether to lower case and the maximum sequence length
# with open(os.path.join(ckpts_path, "config.json"), "w") as f:
#     tokenizer_cfg = {
#         "do_lower_case": True,
#         "unk_token": "[UNK]",
#         "sep_token": "[SEP]",
#         "pad_token": "[PAD]",
#         "cls_token": "[CLS]",
#         "mask_token": "[MASK]",
#         "model_max_length": max_length,
#         "max_len": max_length,
#     }
#     json.dump(tokenizer_cfg, f, indent=4)

# # when the tokenizer is trained and configured, load it as BertTokenizerFast
# tokenizer = BertTokenizerFast.from_pretrained(ckpts_path)

# def encode_with_truncation(examples):
#     """Mapping function to tokenize the sentences passed with truncation"""
#     return tokenizer(examples["text"], truncation=True, padding="max_length",
#                     max_length=max_length, return_special_tokens_mask=True)

# def encode_without_truncation(examples):
#     """Mapping function to tokenize the sentences passed without truncation"""
#     return tokenizer(examples["text"], return_special_tokens_mask=True)

# # the encode function will depend on the truncate_longer_samples variable
# encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# # tokenizing the train dataset
# train_dataset = d["train"].map(encode, batched=True)
# # tokenizing the testing dataset
# test_dataset = d["test"].map(encode, batched=True)

# if truncate_longer_samples:
#     # remove other columns and set input_ids and attention_mask as PyTorch tensors
#     train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
#     test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
# else:
#     # remove other columns, and remain them as Python lists
#     test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
#     train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

# from itertools import chain
# # Main data processing function that will concatenate all texts from our dataset and generate chunks of
# # max_seq_length.
# # grabbed from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= max_length:
#         total_length = (total_length // max_length) * max_length
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
#         for k, t in concatenated_examples.items()
#     }
#     return result

# # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
# # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
# # might be slower to preprocess.
# #
# # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
# if not truncate_longer_samples:
#     train_dataset = train_dataset.map(group_texts, batched=True,
#                                         desc=f"Grouping texts in chunks of {max_length}")
#     test_dataset = test_dataset.map(group_texts, batched=True,
#                                     desc=f"Grouping texts in chunks of {max_length}")
#     # convert them from lists to torch tensors
#     train_dataset.set_format("torch")
#     test_dataset.set_format("torch")

# print(len(train_dataset), len(test_dataset))

# # initialize the model with the config
# model_config = BertConfig(
#     vocab_size=vocab_size,
#     max_position_embeddings=max_length,
#     num_hidden_layers=args.num_hidden_layers,
# )
# print(f"model_config: {model_config}")
# model = BertForMaskedLM(config=model_config)
