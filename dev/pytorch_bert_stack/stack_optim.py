# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

import os
import argparse
import copy
from pprint import pprint
import json

import torch

from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers import BertForMaskedLM, BertConfig

def main():
    parser = argparse.ArgumentParser(description='Copy checkpoint files from source to destination folder.')
    parser.add_argument('--ckpt_folder', required=True, help='Path to the src ckpt folder, e.g., "./pretrained-bert-1-layer/checkpoint-68"')
    parser.add_argument('--stack_ckpt_folder', required=True, help='Path to the destination ckpt folder, "./pretrained-bert-2-layers/checkpoint-68-stack"')
    args = parser.parse_args()

    # ckpt_folder = "./pretrained-bert-1-layer/checkpoint-68" ### XXX
    # stack_ckpt_folder = "./pretrained-bert-2-layers/checkpoint-68-stack" ### XXX

    # make dir if not exist
    if not os.path.exists(args.stack_ckpt_folder):
        os.makedirs(args.stack_ckpt_folder)

    optim_ckpt_name = 'optimizer.pt'
    ckpt_optim_path = os.path.join(args.ckpt_folder, optim_ckpt_name)
    stack_optim_path = os.path.join(args.stack_ckpt_folder, optim_ckpt_name)

    ckpt_optim = torch.load(ckpt_optim_path)

    # print(ckpt_optim.keys())
    ### dict_keys(['state', 'param_groups'])

    # print(ckpt_optim['state'].keys())
    ckpt_optim_idx = list(ckpt_optim['state'].keys())
    ### dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    ### who to duplicate, so you know now! from where to where. but how to write the code?

    parameter_names_path = os.path.join("pretrained-bert-1-layer", "optimiezr_grouped_parameter_names.json")
    with open(parameter_names_path, 'r') as f:
        parameter_names_before_grow = json.load(f)

    # print('-'*80)
    param_names_before_grow = []
    for group in parameter_names_before_grow:
        for i, param_name in enumerate(group['params']):
            # print(i, param_name)
            param_names_before_grow.append(param_name)
    # print('-'*80)

    # --------------------------------------------------------------------------------
    # 0  bert.embeddings.word_embeddings.weight
    # 1  bert.embeddings.position_embeddings.weight
    # 2  bert.embeddings.token_type_embeddings.weight
    # 3  bert.encoder.layer.0.attention.self.query.weight   <-- this is the start to duplicate
    # 4  bert.encoder.layer.0.attention.self.key.weight
    # 5  bert.encoder.layer.0.attention.self.value.weight
    # 6  bert.encoder.layer.0.attention.output.dense.weight
    # 7  bert.encoder.layer.0.intermediate.dense.weight
    # 8  bert.encoder.layer.0.output.dense.weight           <-- this is the end to duplicate
    # 9  cls.predictions.transform.dense.weight
    # 10 bert.embeddings.LayerNorm.weight

    # 11 bert.embeddings.LayerNorm.bias
    # 12 bert.encoder.layer.0.attention.self.query.bias     <-- this is the start to duplicate
    # 13 bert.encoder.layer.0.attention.self.key.bias
    # 14 bert.encoder.layer.0.attention.self.value.bias
    # 15 bert.encoder.layer.0.attention.output.dense.bias
    # 16 bert.encoder.layer.0.attention.output.LayerNorm.weight
    # 17 bert.encoder.layer.0.attention.output.LayerNorm.bias
    # 18 bert.encoder.layer.0.intermediate.dense.bias
    # 19 bert.encoder.layer.0.output.dense.bias
    # 20 bert.encoder.layer.0.output.LayerNorm.weight
    # 21 bert.encoder.layer.0.output.LayerNorm.bias         <-- this is the start to duplicate

    # 22 cls.predictions.bias
    # 23 cls.predictions.transform.dense.bias
    # 24 cls.predictions.transform.LayerNorm.weight
    # 25 cls.predictions.transform.LayerNorm.bias
    # --------------------------------------------------------------------------------


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

    start_idx_with_decay, end_idx_with_decay = None, None
    start_idx_without_decay, end_idx_without_decay = None, None

    for i, (name, idx) in enumerate(zip(param_names_before_grow, ckpt_optim_idx)):
        if 'bert.encoder.layer.0' in name:
            if start_idx_with_decay is None:
                start_idx_with_decay = idx
            elif end_idx_with_decay is None and (i == len(param_names_before_grow) - 1 or 'bert.encoder.layer.0' not in param_names_before_grow[i+1]): ### XXX
                end_idx_with_decay = idx
            elif start_idx_without_decay is None and end_idx_with_decay is not None:
                start_idx_without_decay = idx
            elif end_idx_without_decay is None and end_idx_with_decay is not None and (i == len(param_names_before_grow) - 1 or 'bert.encoder.layer.0' not in param_names_before_grow[i+1]): ### XXX
                end_idx_without_decay = idx

    assert start_idx_with_decay is not None
    assert end_idx_with_decay is not None
    assert start_idx_without_decay is not None
    assert end_idx_without_decay is not None

    # print('-'*80)
    # print('start_idx_with_decay', start_idx_with_decay, 'end_idx_with_decay', end_idx_with_decay)
    # print('start_idx_without_decay', start_idx_without_decay, 'end_idx_without_decay', end_idx_without_decay)
    # print('-'*80)

    # Duplicate the key-value pairs within the range
    duplicated_state = {}
    for key in range(start_idx_with_decay):
        duplicated_state[key] = copy.deepcopy(ckpt_optim['state'][key])

    # display_name_tensor_shape(duplicated_state)

    #-----------------------------------------

    ### 插入新的, 但是保持原来的顺序
    for key in range(start_idx_with_decay, end_idx_with_decay + 1):
        duplicated_state[key] = copy.deepcopy(ckpt_optim['state'][key])

    # display_name_tensor_shape(duplicated_state)

    offset_key_with_decay = end_idx_with_decay - start_idx_with_decay + 1
    for key in range(start_idx_with_decay, end_idx_with_decay + 1):
        new_key = key + offset_key_with_decay
        duplicated_state[new_key] = copy.deepcopy(ckpt_optim['state'][key])

    # display_name_tensor_shape(duplicated_state)

    #-----------------------------------------

    for key in range(end_idx_with_decay+1, start_idx_without_decay):
        new_key = key + offset_key_with_decay
        duplicated_state[new_key] = copy.deepcopy(ckpt_optim['state'][key])

    # display_name_tensor_shape(duplicated_state)

    #-----------------------------------------
    ### 插入新的, 但是保持原来的顺序
    for key in range(start_idx_without_decay, end_idx_without_decay + 1):
        new_key = key + offset_key_with_decay
        duplicated_state[new_key] = copy.deepcopy(ckpt_optim['state'][key])

    # display_name_tensor_shape(duplicated_state)

    offset_key_without_decay = end_idx_without_decay - start_idx_without_decay + 1
    for key in range(start_idx_without_decay, end_idx_without_decay + 1):
        new_key = key + offset_key_with_decay + offset_key_without_decay
        duplicated_state[new_key] = copy.deepcopy(ckpt_optim['state'][key])

    # display_name_tensor_shape(duplicated_state)

    #-----------------------------------------

    for key in range(end_idx_without_decay + 1, len(ckpt_optim['state'])):
        new_key = key + offset_key_with_decay + offset_key_without_decay
        duplicated_state[new_key] = copy.deepcopy(ckpt_optim['state'][key])

    # display_name_tensor_shape(duplicated_state)

    assert len(duplicated_state) == len(ckpt_optim['state']) + \
            (end_idx_with_decay - start_idx_with_decay + 1) + \
            (end_idx_without_decay - start_idx_without_decay + 1)

    ########################################################

    # 30,522 vocab is BERT's default vocab size, feel free to tweak
    vocab_size = 30_522
    # maximum sequence length, lowering will result to faster training (when increasing batch size)
    max_length = 512

    # initialize the model with the config
    model_config = BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_length,
        num_hidden_layers=2,                    ### XXX
    )
    # print(f"model_config: {model_config}")
    model = BertForMaskedLM(config=model_config)
    # print(model)

    optim_state_after_grow = create_optimizer_group_names(model)

    param_groups_idx = []
    i = 0
    for param_group in optim_state_after_grow:
        idx = []
        for param in param_group['params']:
            idx.append(i)
            i += 1
        param_groups_idx.append(idx)

    # print(param_groups_idx)

    new_param_groups = copy.deepcopy(ckpt_optim['param_groups'])
    for i, param_group in enumerate(new_param_groups):
        param_group['params'] = param_groups_idx[i]

    # Create the duplicated ckpt_optim
    duplicated_ckpt_optim = {
        'state': duplicated_state,
        'param_groups': new_param_groups,
    }

    torch.save(duplicated_ckpt_optim, stack_optim_path)
    print(f"saved to {stack_optim_path}")


def create_optimizer_group_names(model):
    """
    Setup the optimizer.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """
    opt_model = model

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    ### 迭代所有名称
    ### print('-'*80)
    optimizer_grouped_parameter_names = [
        {
            "params": [
                n for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
        },
        {
            "params": [
                n for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
        },
    ]

    return optimizer_grouped_parameter_names


def display_name_tensor_shape(x):
    print('-'*80)
    for k, v in x.items():
        print(k, v['exp_avg'].shape)
    print('-'*80)


if __name__ == '__main__':
    main()


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
