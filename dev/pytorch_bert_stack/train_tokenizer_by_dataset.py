# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()
import os

from global_constants import ckpts_path, d, max_length, vocab_size
from tokenizers import BertWordPieceTokenizer

from transformers import BertTokenizerFast


special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

### Get the dataset iterator
def dataset_iterator(dataset):
    for example in dataset:
        yield example["text"]

### Initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()

### Train the tokenizer directly from the dataset iterator
tokenizer.train_from_iterator(dataset_iterator(d["train"]), vocab_size=vocab_size, special_tokens=special_tokens)

### Enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

# if not os.path.isdir(ckpts_path):
#     os.mkdir(ckpts_path)

### Create the new tokenizer object
new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer, model_max_length=max_length)
new_tokenizer.save_pretrained(ckpts_path)
print(f"saved tokenizer to {ckpts_path}")
