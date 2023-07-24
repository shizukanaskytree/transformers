# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()
from commons import d, max_length, tokenizer_ckpt_path, vocab_size
from tokenizers import BertWordPieceTokenizer

from transformers import BertTokenizerFast


special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

### Get the dataset iterator
def dataset_iterator(dataset):
    for example in dataset:
        yield example["text"]

### Initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()

### Train the tokenizer directly from the dataset iterator, doc: https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/bert_wordpiece.py#L118
tokenizer.train_from_iterator(dataset_iterator(d["train"]), vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True)

### Enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

### todo: tokenizer.save(tokenizer_ckpt_path)

### Create the new tokenizer object
new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer, model_max_length=max_length)
new_tokenizer.save_pretrained(tokenizer_ckpt_path)
print(f"saved tokenizer to local path: {tokenizer_ckpt_path}")
