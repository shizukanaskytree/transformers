import os
from os.path import expanduser, join

from commons import remote_hub_ckpts_path, tokenizer_ckpt_path
from dotenv import load_dotenv
from huggingface_hub import login

from transformers import AutoTokenizer, BertTokenizerFast


### Get the absolute path to the .env file in the user's home directory
dotenv_path = join(expanduser("~"), ".env")
### Load environment variables from .env file
load_dotenv(dotenv_path)
### Get the token from the environment variables
token = os.getenv("HUGGINGFACE_TOKEN")

### login to huggingface hub; https://huggingface.co/docs/huggingface_hub/quick-start
login(token=token)

#-------------------------------------------------------------------------------

### load tokenizer files from the local dir since we have already trained the model tokenizer
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_ckpt_path)

### upload tokenizer to hub, remote_hub_ckpts_path
tokenizer.push_to_hub(remote_hub_ckpts_path)

#-------------------------------------------------------------------------------

### test for loading tokenizer from hub
tokenizer = AutoTokenizer.from_pretrained(remote_hub_ckpts_path)
### What are downloaded:
### (py311) xiaofeng.wu@Fairfax4way04RTX4090:/home/xiaofeng.wu/prjs/transformers/dev/pytorch_bert_stack$ python upload_tokenizer.py
### Downloading (…)okenizer_config.json: 100%|█████████████████████████████████████████████████████████████████| 314/314 [00:00<00:00, 5.38MB/s]
### Downloading (…)solve/main/vocab.txt: 100%|███████████████████████████████████████████████████████████████| 216k/216k [00:00<00:00, 26.8MB/s]
### Downloading (…)/main/tokenizer.json: 100%|███████████████████████████████████████████████████████████████| 695k/695k [00:00<00:00, 46.9MB/s]
### Downloading (…)cial_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████| 125/125 [00:00<00:00, 953kB/s]
