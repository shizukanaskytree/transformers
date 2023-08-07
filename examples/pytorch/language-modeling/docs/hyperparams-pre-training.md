# hyperparams: Pretraining and Finetuning

- https://stephanheijl.com/notes_on_bert.html
- https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it


| Setting | Pretraining | Finetuning |
| --- | --- | --- |
| **Adam settings** | | |
| Learning rate | 1e-4 | Best of [5e-5, 4e-5, 3e-5, 2e-5] |
| Learning rate warmup | 10000 steps | 10000 steps |
| L2 weight decay | 0.01 | 0.01 |
| β1 | 0.9 | 0.9 |
| β2 | 0.999 | 0.999 |
| **Architecture settings** | | |
| Dropout | 0.1 | 0.1 |
| Activation | GELU | GELU |
| **Training settings** | | |
| Batch size | 256 | Best of [16, 32] |
| Epochs | 40 | Best of [2, 3, 4] |

# wandb

check all params:
https://wandb.ai/shizukanaskytree/huggingface/runs/63ijp44l/overview?workspace=user-shizukanaskytree

# Summary

On the whole, this paper was easy to read, with very little mathematics included, save the occasional vector or softmax definition. The focus was primarily (and rightfully) on the empirical results, not so much the technical decisions; these were offloaded onto the Transformers paper (Attention is all you need), which I intend to read next.

As an overview, the BERT paper introduces a neural network based on the Transformer architecture which should serve as a base of myriad natural language processing tasks. The model is pretrained on a very large corpus (3.3 billion words) and the user need only make small refinements to the architecture to adjust to a specific NLP task.

Pretraining happens with two tasks that should simultaneously learn the network the meaning of tokens as well as the structure of its inputs. BERT accepts a vector of up to 512 tokens that can optionally be split into two segments (A and B). BERT uses the different segments to natively perform better in two part NLP tests (ie. question answering), while only retaining the A segment for classification tasks.

The first pretraining task is the Masked Language Model, where the model is fed sentences with masked tokens and is asked to produce the true vocabulary ID of the missing word(piece). Here I learned that tokenization happens along “word-pieces”, which appears to be an unsupervised tokenization strategy. Google has expanded on this in this github repo: https://github.com/google/sentencepiece . I also found this technique similar to the skip-grams used in Word2Vec. The principle appears the same, except that W2V is never presented with a representation of the sentence, just the center word of the window where the target word is picked. Some strategies, such as replacing the [MASK] token with a random word (or even the true word in rare cases) are used to adjust the model to real world tasks where the [MASK] token is not used.

The second pretraining task is the sentence prediction task. Here the BERT model is presented with two sentences, encoded with segment A and B embeddings as part of one input. Sometimes the model is presented with two consecutive sentences from the corpus, other times one second sentence is a non-sequitor, randomly picked from the dataset. This pretraining task is intended to improve the performance of the dual segment component of the model.

In all pretraining tasks, inputs are represented by a vector containing a token embedding, a segment embedding and a position embedding. The final embedding is used because there is no recurrence and no convolution, so there is no inherent notion of (relative) position for each token.

When performing classification tasks, the activations for the vector corresponding to the final token ([CLS]) are used as a representative for the entire input. I am not certain at what step the model would allocate meaning to that specific token, or why the sentence start token was not used, given that the model is bidirectional.

The pretraining learning rate is set to 1e-4, not an uncommon learning rate for Adam. The first 10.000 steps are subject to learning rate warm-up, where the lr is linearly increased from 0 to the target. After that point, learning rate decay starts.

When the BERT model is used for a specific NLP task, only small architecture changes are required. These include adding a final layer with softmax for binary or multiclass classification, adding two layers with softmax for start and end tokens in a span in the case of question answering or transforming the vectors for every token when perform entity recognition. It should be noted that for each of these adjustments, the entire model must be fine-tuned, not just the top layers. This happens for only 3-4 epochs with relatively low learning rates. It is also demonstrated that activations of several layers may be extracted and used as input features, at the cost of some performance for NER.

Some final, short notes:

I appreciated the use of an ablation experiment to show the influence of the different pretraining tasks. This offers more insight into what truely benefits BERT, outside the larger structure with 340M parameters and the larger training set.
In the ablation experiment, a BiLSTM was added on top of the model subject only to Left-to-right pretraining. While this improved the score in SQUAD Q/A tasks by 10 points, it decreased performance in GLUE tasks significantly. I wager that this is related to the difference between binary classification tasks and the start/end span task.
I noticed many tasks listed already have an ELMo baseline, which I found a useful comparison point, even though it was released very recently.
Some datasets incorporate a testing platform with rigorous standards for participation (SQUAD & GLUE). This should help in the evaluation of new algorithms in the future.
