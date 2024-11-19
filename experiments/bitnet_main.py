# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

model = AutoModelForCausalLM.from_pretrained("HF1BitLLM/Llama3-8B-1.58-100B-tokens", device_map="cuda", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
print(model)
input_text = "Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:"

input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
output = model.generate(input_ids, max_length=100, do_sample=False)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
