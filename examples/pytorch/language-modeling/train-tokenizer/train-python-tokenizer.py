### https://huggingface.co/learn/nlp-course/chapter6/2
from datasets import load_dataset

### This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python")

# print(raw_datasets["train"])
# print('-'*80)


# print(type(raw_datasets["train"]))
# print('-'*80)


# print(raw_datasets["train"][123456]["whole_func_string"])
# print('-'*80)


# print(type(raw_datasets["train"][123456]))
# print('-'*80)


# print(raw_datasets["train"][123456])
# print('-'*80)

# for k, v in raw_datasets["train"][123456].items():
#     print(k, '\n', v)
#     print('-'*20)

# print('-'*80)


### Using a Python generator, we can avoid Python loading anything into memory
### until it’s actually necessary. To create such a generator, you just to need to
### replace the brackets with parentheses. This line of code doesn’t fetch any
### elements of the dataset; it just creates an object you can use in a Python for
### loop. The texts will only be loaded when you need them (that is, when you’re
### at the step of the for loop that requires them), and only 1,000 texts at a
### time will be loaded. This way you won’t exhaust all your memory even if you
### are processing a huge dataset.
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)


# def get_training_corpus():
#     return (
#         raw_datasets["train"][i : i + 1000]["whole_func_string"]
#         for i in range(0, len(raw_datasets["train"]), 1000)
#     )

### 需要注意:
### it will produce the exact same generator as before, but allows you to use more
### complex logic than you can in a list comprehension.
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]


training_corpus = get_training_corpus()


# for i, x in enumerate(training_corpus):
#     print(type(x))
#     print(len(x))
#     if i == 1:
#         break
#     print('-' * 80)

################################################################################

### Training a new tokenizer
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''
print('example:\n', example)

tokens = old_tokenizer.tokenize(example)
print('tokens:\n', tokens)

### Ġ and Ċ, which denote spaces and newlines
### ['def', 'Ġadd', '_', 'n', 'umbers', '(', 'a', ',', 'Ġb', '):', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 'Ġ`', 'b', '`', '."', '""', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']

################################################################################

### Let’s train a new tokenizer and see if it solves those issues.
### For this, we’ll use the method train_new_from_iterator():
### This is quite a compact representation;
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokens = tokenizer.tokenize(example)
print('new tokens:\n', tokens)
print('new tokens len:\n', len(tokens))
print('len(old_tokenizer.tokenize(example)):\n', len(old_tokenizer.tokenize(example)))


example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """

print(f'example: {example}')
print('tokenizer.tokenize(example)\n', tokenizer.tokenize(example))


################################################################################

### Saving the tokenizer
tokenizer.save_pretrained("code-search-net-tokenizer")

################################################################################

from huggingface_hub import notebook_login

notebook_login()

# huggingface-cli login

tokenizer.push_to_hub("code-search-net-tokenizer")

