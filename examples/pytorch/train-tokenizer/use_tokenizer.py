# tutorial: https://huggingface.co/learn/nlp-course/chapter6/2
# hub: https://huggingface.co/skytree/code-search-net-tokenizer

from transformers import AutoTokenizer

# Replace "huggingface-course" below with your actual namespace to use your own tokenizer
tokenizer = AutoTokenizer.from_pretrained("skytree/code-search-net-tokenizer")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''
print('example:\n', example)

tokens = tokenizer.tokenize(example)
print('new tokens:\n', tokens)
print('new tokens len:\n', len(tokens))
