### tutorial: https://huggingface.co/bert-base-uncased
from transformers import pipeline
from pprint import pprint

model_path = "/home/xiaofeng.wu/prjs/ckpts/bert-base-uncased/checkpoint-34000"
tokenizer_path = "skytree/tokenizer-bert-wiki-bookcorpus"

pipe = pipeline("fill-mask", model=model_path, tokenizer=tokenizer_path)

### output = pipe("The man worked as a [MASK].")
outputs = pipe("I no longer [MASK] her, true, [MASK] perhaps I [MASK] her.", top_k=5)
# print(f"outputs: {outputs}")

for output in outputs:
    pprint(f"output: {output}")
    for prediction in output:
        sentence = prediction['sequence']
        sentence = sentence.replace('[CLS]', '').replace('[SEP]', '')
        print(sentence)

