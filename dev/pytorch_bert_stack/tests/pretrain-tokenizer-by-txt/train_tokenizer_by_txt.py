# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()
import os

from datasets import concatenate_datasets, load_dataset
from dev.pytorch_bert_stack.commons import ckpts_path, max_length, vocab_size
from tokenizers import BertWordPieceTokenizer

from transformers import BertTokenizerFast


bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column
assert bookcorpus.features.type == wiki.features.type
dataset = concatenate_datasets([bookcorpus, wiki])
# print(type(dataset)) ### datasets.arrow_dataset.Dataset

### We do not split the dataset into training (90%) and testing (10%); API doc: If test_size is None, the value is set to the complement of the train size.
d = dataset.train_test_split(test_size=None)
# print(d["train"], d["test"])

# for t in d["train"]["text"][:3]:
#     print(t)
#     print("="*50)

### if you want to train the tokenizer from scratch (especially if you have custom
### dataset loaded as datasets object), then run this cell to save it as files
### but if you already have your custom data as text files, there is no point using this
def dataset_to_text(dataset, output_filename="data.txt"):
    """Utility function to save dataset text to disk,
    useful for using the texts to train the tokenizer
    (as the tokenizer accepts files)"""
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)

### save the training set to train.txt
dataset_to_text(d["train"], "train.txt")
### save the testing set to test.txt. If we set test_size=None, then the test set is the complement of the train set
# dataset_to_text(d["test"], "test.txt")

special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

### if you want to train the tokenizer on both sets
# files = ["train.txt", "test.txt"]
### training the tokenizer on the training set
files = ["train.txt"]

### initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()
### train the tokenizer
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
### enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

if not os.path.isdir(ckpts_path):
    os.mkdir(ckpts_path)

new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer, model_max_length=max_length)
new_tokenizer.save_pretrained(ckpts_path)
print(f"saved tokenizer to {ckpts_path}")
