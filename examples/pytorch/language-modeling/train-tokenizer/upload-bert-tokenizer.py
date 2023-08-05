from huggingface_hub import notebook_login
from transformers import AutoTokenizer

### terminal: huggingface-cli login
notebook_login()

### load a new tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer-bert-wiki-bookcorpus")

### test the loaded tokenizer
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''
print('example:\n', example)

tokens = tokenizer.tokenize(example)
print('new tokens:\n', tokens)
print('new tokens len:\n', len(tokens))

#-------------------------------------------------------------------------------

### upload tokenizer to hub: https://huggingface.co/skytree/tokenizer-bert-wiki-bookcorpus
tokenizer.push_to_hub("tokenizer-bert-wiki-bookcorpus")
