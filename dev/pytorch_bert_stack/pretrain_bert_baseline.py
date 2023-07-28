# -*- coding: utf-8 -*-
"""
Original file PretrainingBERT_PythonCodeTutorial.ipynb is located at
    https://colab.research.google.com/drive/1An1VNpKKMRVrwcdQQNSe7Omh_fl2Gj-2
"""
# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

import argparse
import glob
import os

import torch
from commons import (
    global_batch_size,
    max_length,
    remote_hub_ckpts_path,
    save_ckpt_every_X_steps,
    tokenized_test_datasets_path,
    tokenized_train_datasets_path,
    vocab_size,
)
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# os.environ['WANDB_MODE'] = 'offline'
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

################################################################################

parser = argparse.ArgumentParser(description='Copy checkpoint files from source to destination folder.')
parser.add_argument('--path_to_ckpts', default="", required=False, help='Path to the checkpoint folder')
parser.add_argument('--num_hidden_layers', required=True, type=int, help='The number of layers in the stacked model, e.g., 2')
args = parser.parse_args()

################################################################################

### test for loading tokenizer from hub
tokenizer = AutoTokenizer.from_pretrained(remote_hub_ckpts_path)

### Load your tokenized dataset, we do not split wiki and bookcorpus
tokenized_train_dataset = load_from_disk(tokenized_train_datasets_path)
tokenized_test_dataset = load_from_disk(tokenized_test_datasets_path) if tokenized_test_datasets_path else None

### initialize the model with the config
model_config = BertConfig(
    vocab_size=vocab_size,
    max_position_embeddings=max_length,
    num_hidden_layers=args.num_hidden_layers,
)
print('-'*80)
print(f"model_config:\n{model_config}")
print('-'*80)

model = BertForMaskedLM(config=model_config)
# print('-'*80)
# print(f"model:\n{model}")
# print('-'*80)

### initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
### Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

num_of_gpus = torch.cuda.device_count()

### finally, e.g., ckpt-bert-wiki-bookcorpus/pretrained-bert-1-layers/checkpoint-10
curr_file_path = os.path.abspath(__file__)
curr_file_dir = os.path.dirname(curr_file_path)
model_ckpt_path = os.path.join(curr_file_dir, args.path_to_ckpts) # e.g., args.path_to_ckpts: "pretrained-bert-1-layers", inside the folder "pretrained-bert-1-layers", we have "checkpoint-10" etc.

### TrainingArguments for bert reference:
### https://github.com/philschmid/deep-learning-habana-huggingface/blob/master/pre-training/pre-training-bert.ipynb
training_args = TrainingArguments(
    output_dir=model_ckpt_path,                                                # output directory to where save model checkpoint
    do_eval=False,
    # evaluation_strategy="steps",                                              # evaluate each `logging_steps` steps
    # eval_steps=10,                                                            # Evaluate every 500 training steps
    overwrite_output_dir=True,
    max_steps=300_000,                                                          # Limit the total number of training steps to 100_000
    # num_train_epochs=10,                                                      # If I set max_steps, I will not set num_train_epochs; number of training epochs, feel free to tweak, original code settting is 10
    per_device_train_batch_size=global_batch_size//num_of_gpus,                 # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=1,                                              # accumulating the gradients before updating the weights
    per_device_eval_batch_size=global_batch_size//num_of_gpus,                  # evaluation batch size
    logging_strategy='steps',
    logging_steps=1000,                                                         # evaluate, log and save model checkpoints every 1000 step, original 1000, for debug and testing with a smaller number e.g., 1
    save_steps=save_ckpt_every_X_steps,                                         # original 1000, for debug and testing 1
    # load_best_model_at_end=True,                                              # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=4,                                                         # whether you don't have much space so you let only 3 model weights saved in the disk
    report_to="wandb",                                                          # https://docs.wandb.ai/guides/integrations/huggingface
)

### initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset, # we do not have an eval_dataset since we do not split bookcorpus and wiki datasets
)

### resume from last stacked checkpoint if it exists
pattern = 'checkpoint-*' # e.g., checkpoint-50-stacked
### path_to_ckpts: ckpt-bert-wiki-bookcorpus/pretrained-bert-1-layers
ckpt_dir = glob.glob(f'{args.path_to_ckpts}/{pattern}')
# print(f"ckpt_dir: {ckpt_dir}")
# print(f"ckpt_dir[-1]: {ckpt_dir[-1]}")

### train the model
trainer.train(resume_from_checkpoint=ckpt_dir[-1])

################################################################################

# ### when you load from pretrained
# model = BertForMaskedLM.from_pretrained(os.path.join(model_ckpts_path, "checkpoint-10"))
# tokenizer = BertTokenizerFast.from_pretrained(tokenizer_ckpt_path)
# ### or simply use pipeline
# fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# ### perform predictions
# example = "It is known that [MASK] is the capital of Germany"
# for prediction in fill_mask(example):
#     print(prediction)

# ### perform predictions
# examples = [
#     "Today's most trending hashtags on [MASK] is Donald Trump",
#     "The [MASK] was cloudy yesterday, but today it's rainy.",
# ]
# for example in examples:
#     for prediction in fill_mask(example):
#         print(f"{prediction['sequence']}, confidence: {prediction['score']}")
#     print("="*50)

# !nvidia-smi
