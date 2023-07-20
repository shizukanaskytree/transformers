# Sequence classification with IMDb reviews
# https://huggingface.co/docs/transformers/v4.15.0/custom_datasets#question-answering-with-squad

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer

import os
# set env of WANDB_MODE=offline
os.environ["WANDB_MODE"] = "offline"
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#############################################################
### Load IMDb dataset
# The 🤗 Datasets library makes it simple to load a dataset:
imdb = load_dataset("imdb")

# eg = imdb["train"][0]
# print(eg)

### {'label': 1, 'text': 'Bromwell High is a cartoon comedy. It ran at the same
###  time as some other programs about school life, such as "Teachers". My 35
### years in the teaching profession lead me to believe that Bromwell High\'s
### satire is much closer to reality than is "Teachers". The scramble to survive
### financially, the insightful students who can see right through their pathetic
### teachers\' pomp, the pettiness of the whole situation, all remind me of the
### schools I knew and their students. When I saw the episode in which a student
### repeatedly tried to burn down the school, I immediately recalled ......... at
### .......... High. A classic line: INSPECTOR: I\'m here to sack one of your
### teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my
### age think that Bromwell High is far fetched. What a pity that it isn\'t!' },
##
### { 'text': 'I rented I AM CURIOUS-YELLOW from my video store because of all the
###     controversy that surrounded it when it was first released in 1967. I also
###         heard that at first it was seized by U.S. customs if it ever tried to
###     enter this country, therefore being a fan of films considered
###         "controversial" I really had to see this for myself.<br /><br />The
###         plot is centered around a young Swedish drama student named Lena who
###         wants to learn everything she can about life. In particular she wants
###         to focus her attentions to making some sort of documentary on what the
###         average Swede thought about certain political issues such as the
###         Vietnam War and race issues in the United States. In between asking
###         politicians and ordinary denizens of Stockholm about their opinions on
###         politics, she has sex with her drama teacher, classmates, and married
###         men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40
###         years ago, this was considered pornographic. Really, the sex and
###         nudity scenes are few and far between, even then it\'s not shot like
###         some cheaply made porno. While my countrymen mind find it shocking, in
###         reality sex and nudity are a major staple in Swedish cinema. Even
###         Ingmar Bergman, arguably their answer to good old boy John Ford, had
###         sex scenes in his films.<br /><br />I do commend the filmmakers for
###         the fact that any sex shown in the film is shown for artistic purposes
###         rather than just to shock people and make money to be shown in
###         pornographic theaters in America. I AM CURIOUS-YELLOW is a good film
###         for anyone wanting to study the meat and potatoes (no pun intended) of
###         Swedish cinema. But really, this film doesn\'t have much of a plot.',
###         'label': 0
### }

### label 0 is negative; 1 is positive


#############################################################
### Preprocess

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#############################################################
### Fine-tune with the Trainer API

# DistilBertForSequenceClassification, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print(f"model: {model}")

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

