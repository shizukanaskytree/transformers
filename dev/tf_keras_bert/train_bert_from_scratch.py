# Pretraining BERT with Hugging Face Transformers

import nltk
import random
import logging

import tensorflow as tf
from tensorflow import keras

nltk.download("punkt")
# Only log error messages
tf.get_logger().setLevel(logging.ERROR)
# Set random seed
tf.keras.utils.set_random_seed(42)


TOKENIZER_BATCH_SIZE = 256  # Batch-size to train the tokenizer on
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

BLOCK_SIZE = 128  # Maximum number of tokens in an input sample
NSP_PROB = 0.50  # Probability that the next sentence is the actual next sentence in NSP
SHORT_SEQ_PROB = 0.1  # Probability of generating shorter sequences to minimize the mismatch between pretraining and fine-tuning.
MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

MLM_PROB = 0.2  # Probability with which tokens are masked in MLM

TRAIN_BATCH_SIZE = 2  # Batch-size for pretraining the model on
MAX_EPOCHS = 1  # Maximum number of epochs to train the model for
LEARNING_RATE = 1e-4  # Learning rate for training the model

MODEL_CHECKPOINT = "bert-base-cased"  # Name of pretrained model from 🤗 Model Hub

# Load the WikiText dataset

from datasets import load_dataset

# 大类: wikitext; 小类: wikitext-2-raw-v1
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
# print(dataset)

#-------------------------------------------------------------------------------

# Training a new Tokenizer

# First we train our own tokenizer from scratch on our corpus, so that can we
# can use it to train our language model from scratch.

# But why would you need to train a tokenizer? That's because Transformer models
# very often use subword tokenization algorithms, and they need to be trained to
# identify the parts of words that are often present in the corpus you are
# using.

# Subword tokenization algorithms rely on the principle that frequently used
# words should not be split into smaller subwords, but rare words should be
# decomposed into meaningful subwords. For instance "annoyingly" might be
# considered a rare word and could be decomposed into "annoying" and "ly" .
# tutorial video: https://huggingface.co/docs/transformers/main/tokenizer_summary#subword-tokenization

# The 🤗 Transformers Tokenizer (as the name indicates) will tokenize the inputs
# (including converting the tokens to their corresponding IDs in the pretrained
# vocabulary) and put it in a format the model expects, as well as generate the
# other inputs that model requires.

# First we make a list of all the raw documents from the WikiText corpus:
# print('-'*80) print('dataset["train"]: ', dataset["train"]) print('-'*80)

"""
dataset["train"]: Dataset({
    features: ['text'], num_rows: 36718
})
"""

all_texts = [
    doc for doc in dataset["train"]["text"] if len(doc) > 0 and not doc.startswith(" =")
]

# print("Total number of documents:", len(all_texts)) print('留存率:',
# len(all_texts) / len(dataset["train"]["text"])) print("0 Sample document:",
# all_texts[0][:200]) print('-'*80) print("1 Sample document:",
# all_texts[1][:200]) print('-'*80)


def batch_iterator():
    for i in range(0, len(all_texts), TOKENIZER_BATCH_SIZE):
        yield all_texts[i : i + TOKENIZER_BATCH_SIZE]


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Now we train our tokenizer using the entire train split of the Wikitext-2
# dataset.
tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(), vocab_size=TOKENIZER_VOCABULARY
)

#-------------------------------------------------------------------------------

dataset["train"] = dataset["train"].select([i for i in range(1000)])
dataset["validation"] = dataset["validation"].select([i for i in range(1000)])


