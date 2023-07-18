# load the optimizer.pt from
# /Users/wxf/Documents/prjs/2023/transformers/dev/pytorch_bert_stack_cpu/pretrained-bert/checkpoint-66/optimizer.pt

# The full list is
# config.json
# generation_config.json
# optimizer.pt
# pytorch_model.bin
# rng_state.pth
# scheduler.pt
# trainer_state.json
# training_args.bin

# In the following path, we also have  optimiezr_grouped_parameter_names.json which contains the parameter names for both need weight decay and no weight decay
# The full list is:
# (base) wxf@Xiaofengs-MacBook-Pro:~/Documents/prjs/2023/transformers/dev/pytorch_bert_stack_cpu/pretrained-bert$ ls -1
# checkpoint-66/
# checkpoint-67/
# checkpoint-68/
# config.json
# optimiezr_grouped_parameter_names.json
# runs/
# vocab.txt




#--------------------------------------------
# bert.embeddings.word_embeddings.weight
# bert.embeddings.position_embeddings.weight
# bert.embeddings.token_type_embeddings.weight
# bert.embeddings.LayerNorm.weight
# bert.embeddings.LayerNorm.bias
# bert.encoder.layer.0.attention.self.query.weight
# bert.encoder.layer.0.attention.self.query.bias
# bert.encoder.layer.0.attention.self.key.weight
# bert.encoder.layer.0.attention.self.key.bias
# bert.encoder.layer.0.attention.self.value.weight
# bert.encoder.layer.0.attention.self.value.bias
# bert.encoder.layer.0.attention.output.dense.weight
# bert.encoder.layer.0.attention.output.dense.bias
# bert.encoder.layer.0.attention.output.LayerNorm.weight
# bert.encoder.layer.0.attention.output.LayerNorm.bias
# bert.encoder.layer.0.intermediate.dense.weight
# bert.encoder.layer.0.intermediate.dense.bias
# bert.encoder.layer.0.output.dense.weight
# bert.encoder.layer.0.output.dense.bias
# bert.encoder.layer.0.output.LayerNorm.weight
# bert.encoder.layer.0.output.LayerNorm.bias
# cls.predictions.bias
# cls.predictions.transform.dense.weight
# cls.predictions.transform.dense.bias
# cls.predictions.transform.LayerNorm.weight
# cls.predictions.transform.LayerNorm.bias
# cls.predictions.decoder.weight
# cls.predictions.decoder.bias
#--------------------------------------------




# --------------------------------------------------------------------------------
# pprint(optim_state_after_grow)
# [{'params': [ 0 'bert.embeddings.word_embeddings.weight',
#             1 'bert.embeddings.position_embeddings.weight',
#             2 'bert.embeddings.token_type_embeddings.weight',
#             3 'bert.encoder.layer.0.attention.self.query.weight',
#             4 'bert.encoder.layer.0.attention.self.key.weight',
#             5 'bert.encoder.layer.0.attention.self.value.weight',
#             6 'bert.encoder.layer.0.attention.output.dense.weight',
#             7 'bert.encoder.layer.0.intermediate.dense.weight',
#             8 'bert.encoder.layer.0.output.dense.weight',
#             9 'bert.encoder.layer.1.attention.self.query.weight',
#            10 'bert.encoder.layer.1.attention.self.key.weight',
#            11 'bert.encoder.layer.1.attention.self.value.weight',
#            12 'bert.encoder.layer.1.attention.output.dense.weight',
#            13 'bert.encoder.layer.1.intermediate.dense.weight',
#            14 'bert.encoder.layer.1.output.dense.weight',
#            15 'cls.predictions.transform.dense.weight']},
#
# {'params': [ 16 'bert.embeddings.LayerNorm.weight',
#            17 'bert.embeddings.LayerNorm.bias',
#            18 'bert.encoder.layer.0.attention.self.query.bias',
#            19 'bert.encoder.layer.0.attention.self.key.bias',
#            20 'bert.encoder.layer.0.attention.self.value.bias',
#            21 'bert.encoder.layer.0.attention.output.dense.bias',
#            22 'bert.encoder.layer.0.attention.output.LayerNorm.weight',
#            23 'bert.encoder.layer.0.attention.output.LayerNorm.bias',
#            24 'bert.encoder.layer.0.intermediate.dense.bias',
#            25 'bert.encoder.layer.0.output.dense.bias',
#            26 'bert.encoder.layer.0.output.LayerNorm.weight',
#            27 'bert.encoder.layer.0.output.LayerNorm.bias',
#            28 'bert.encoder.layer.1.attention.self.query.bias',
#            29 'bert.encoder.layer.1.attention.self.key.bias',
#            30 'bert.encoder.layer.1.attention.self.value.bias',
#            31 'bert.encoder.layer.1.attention.output.dense.bias',
#            32 'bert.encoder.layer.1.attention.output.LayerNorm.weight',
#            33 'bert.encoder.layer.1.attention.output.LayerNorm.bias',
#            34 'bert.encoder.layer.1.intermediate.dense.bias',
#            35 'bert.encoder.layer.1.output.dense.bias',
#            36 'bert.encoder.layer.1.output.LayerNorm.weight',
#            37 'bert.encoder.layer.1.output.LayerNorm.bias',
#            38 'cls.predictions.bias',
#            39 'cls.predictions.transform.dense.bias',
#            40 'cls.predictions.transform.LayerNorm.weight',
#            41 'cls.predictions.transform.LayerNorm.bias']}]
# --------------------------------------------------------------------------------


import os
import torch
import json
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from pprint import pprint
from transformers import BertForMaskedLM, BertConfig
import copy

# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

def main():
    ckpt_folder = "./pretrained-bert/checkpoint-66"
    ckpt_name = 'pytorch_model.bin'
    stacked_ckpt_name = 'pytorch_model_stack.bin'

    ckpt_model_path = os.path.join(ckpt_folder, ckpt_name)
    stack_model_path = os.path.join(ckpt_folder, stacked_ckpt_name)

    model_ckpt = torch.load(ckpt_model_path)
    # for x in model_ckpt.keys():
    #     print(x)

    start_idx, end_idx = None, None
    new_model_ckpt = {}
    for name, p in model_ckpt.items():
        print(name, p.shape)
        if 'bert.encoder.layer.0' in name:
            ...



if __name__ == '__main__':
    main()
