# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); #debugpy.breakpoint()

import argparse
import os
import copy
import torch
from torch import nn

os.environ['WANDB_MODE'] = 'offline'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

ENCODER_LAYER_0_LAST_LAYER_NAME = 'bert.encoder.layer.0.output.LayerNorm.bias'
BERT_ENCODER_LAYER_0_PREFIX = 'bert.encoder.layer.0.'
BERT_ENCODER_LAYER_1_PREFIX = 'bert.encoder.layer.1.'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, help='The path to the model checkpoint')
    args = parser.parse_args()

    ckpt_path = args.ckpt_path

    model_checkpoint_path = os.path.join(ckpt_path, "checkpoint-2", "pytorch_model.bin")
    loaded_model = torch.load(model_checkpoint_path)

    optim_checkpoint_path = os.path.join(ckpt_path, "checkpoint-2", "optimizer.pt")
    loaded_optim = torch.load(optim_checkpoint_path)

    len_optim = len(loaded_optim['state'])
    for i in range(len_optim):
        shape = str(loaded_optim['state'][i]['exp_avg'].shape)
        print(f"{i:<2}\t{shape:<10}\t")

    # i = 0
    # for k, v in loaded_model.items():
    #     print(f"{i:<2}\t{k:<50}\t{v.shape}")
    #     i += 1


    # Double the encoder layers
    new_loaded_model = copy.deepcopy(loaded_model)



    for k, v in loaded_model.items():
        if k.startswith('bert.encoder.layer'):
            # Extract the layer number
            layer_num = int(k.split('.')[3])

            # Duplicate the layer
            new_layer_num = layer_num + 1
            new_layer_key = k.replace(f'.{layer_num}.', f'.{new_layer_num}.')
            new_loaded_model[new_layer_key] = v

    # Insert the new layers and optimizer info after encoder 0
    insert_index = None # last encoder layer is NO.21
    encoder_layer_0_end_idx = None
    for idx, (k, v) in enumerate(new_loaded_model.items()):
        if k == ENCODER_LAYER_0_LAST_LAYER_NAME:
            insert_index = idx + 1
            encoder_layer_0_end_idx = idx
            break

    encoder_layer_0_start_idx = None
    i = 0
    new_layers = []
    for k, v in new_loaded_model.items():
        # print(i, '\t', k)
        if k.startswith(BERT_ENCODER_LAYER_0_PREFIX):
            if encoder_layer_0_start_idx is None:
                encoder_layer_0_start_idx = i
                # print(f" *** encoder_layer_0_start_idx: NO.{encoder_layer_0_start_idx}") ### NO.5

            new_layer_key = k.replace(BERT_ENCODER_LAYER_0_PREFIX, BERT_ENCODER_LAYER_1_PREFIX)
            new_layers.append((new_layer_key, copy.deepcopy(v)))
        i += 1

    ### 测试一下, 至此

    if insert_index is not None:
        new_loaded_model = dict(
            list(new_loaded_model.items())[:insert_index] +
            new_layers +
            list(new_loaded_model.items())[insert_index:]
        )

    # i = 0
    # for k, v in new_loaded_model.items():
    #     print(i, '\t', k)
    #     i += 1

    # for k, v in new_loaded_model.items():
    #     print(k)

    ############################################################################

    # Duplicate the optimizer state for the new layers
    new_loaded_optim = copy.deepcopy(loaded_optim)

    ### 瞎写
    # for layer_num in range(len(loaded_optim['state'])):
    #     new_layer_num = layer_num + 1
    #     new_layer_state = copy.deepcopy(loaded_optim['state'][layer_num])
    #     new_loaded_optim['state'][new_layer_num] = new_layer_state

    # Split the optimizer parameter groups based on the rule
    ALL_LAYERNORM_LAYERS = ['LayerNorm.bias', 'LayerNorm.weight']
    decay_parameters = get_parameter_names(new_loaded_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    #=============================================================
    ### purpose: 扩展 loaded_optim['state'] 的 step, exp_avg, exp_avg_sq
    ### 思路:
    ### 对于前面的 0-20 层, 保持原样
    ### 对于新的 21-37 层, 复制 0-20 层的 step, exp_avg, exp_avg_sq
    ### 对于剩余的部分 38-41 层, 保持原样

    ### k is a number, it indicates the layer id corresponding to the new_loaded_model layer names
    ### k is critical, we need to ensure the k_id is corresponding to the new_loaded_model layer names

    ### 重新造个呗
    loaded_optim_x = {}

    ### 加塞 怎么处理 index, len, where to where, who to where 等问题. 做完后写写笔记, 整理整理, 不知道这辈子还有没有机会用到.

    # num_prediction_layers = len(loaded_model) - encoder_layer_0_end_idx - 1
    num_prediction_layers_from_optim = len(loaded_optim['state']) - encoder_layer_0_end_idx - 1
    print(f"num_prediction_layers_from_optim: {num_prediction_layers_from_optim}") ### 7

    num_encoder_layers = encoder_layer_0_end_idx - encoder_layer_0_start_idx + 1
    # print(num_encoder_layers) ### 16

    loaded_optim_x['state'] = {k: copy.deepcopy(new_loaded_optim['state'][k]) for k in range(encoder_layer_0_end_idx+1)}

    # encoder_layer_1_start_idx = encoder_layer_0_start_idx + num_encoder_layers # 不用 +1
    # encoder_layer_1_end_idx = encoder_layer_0_end_idx + num_encoder_layers + 1 # idx start

    loaded_optim_x['state'].update(
        {k + num_encoder_layers: copy.deepcopy(new_loaded_optim['state'][k])
            for k in range(encoder_layer_0_start_idx, encoder_layer_0_end_idx+1)}
    )

    ### 取出的是原来的, k 需要调整挪位
    ### k is used to get the layer id corresponding to the new_loaded_model layer names
    original_prediction_start_idx = encoder_layer_0_end_idx + 1
    original_prediction_end_idx = encoder_layer_0_end_idx + num_prediction_layers_from_optim + 1
    loaded_optim_x['state'].update(
        {k + num_encoder_layers: copy.deepcopy(new_loaded_optim['state'][k])
            for k in range(original_prediction_start_idx, original_prediction_end_idx)}
    )

    # print(loaded_optim_x)

    #-------------------------------------------------------------

    ### how to get the decay_parameters layer id from

    ### todo: need decay layer's id not need name right now,
    ### Why?
    ### We need to use the layer id to update the optimizer state params list

    # optimizer_grouped_parameters = [
    #     {
    #         "params": [
    #             p for n, p in new_loaded_model.items() if (n in decay_parameters and p.requires_grad)
    #         ],
    #         "weight_decay": 0.1,  # Set the desired weight decay value
    #     },
    #     {
    #         "params": [
    #             p for n, p in new_loaded_model.items() if (n not in decay_parameters and p.requires_grad)
    #         ],
    #         "weight_decay": 0.0,
    #     },
    # ]

    # Update the optimizer parameter groups in new_loaded_optim
    # new_loaded_optim['param_groups'] = optimizer_grouped_parameters


    # # Update the layer numbers in new_loaded_optim['param_groups'] params list
    # for param_group in new_loaded_optim['param_groups']:
    #     layer_ids = param_group['params']
    #     updated_layer_ids = [
    #         layer_id + len(loaded_optim['param_groups'][0]['params'])
    #         if layer_id >= len(loaded_optim['param_groups'][0]['params'])
    #         else layer_id
    #         for layer_id in layer_ids
    #     ]
    #     param_group['params'] = updated_layer_ids

    # new_loaded_optim['param_groups'] = copy.deepcopy(loaded_optim['param_groups']) + \
    #                                   copy.deepcopy(new_loaded_optim_param_groups)


### 没想到, 被调用的函数可以写在调用者的下面
def get_parameter_names(model, exclude_names):
    keep_parameter_names = []
    for name in model.keys():
        for exclude_name in exclude_names:
            if exclude_name not in name:
                keep_parameter_names.append(name)
    return keep_parameter_names


if __name__ == "__main__":
    main()














# import debugpy
# import argparse
# import os
# import copy
# import torch
# from torch import nn

# os.environ['WANDB_MODE'] = 'offline'
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt_path', type=str, help='The path to the model checkpoint')
#     args = parser.parse_args()

#     ckpt_path = args.ckpt_path

#     model_checkpoint_path = os.path.join(ckpt_path, "checkpoint-2", "pytorch_model.bin")
#     loaded_model = torch.load(model_checkpoint_path)

#     optim_checkpoint_path = os.path.join(ckpt_path, "checkpoint-2", "optimizer.pt")
#     loaded_optim = torch.load(optim_checkpoint_path)

#     # Double the encoder layers
#     new_loaded_model = copy.deepcopy(loaded_model)

#     for k, v in loaded_model.items():
#         if k.startswith('bert.encoder.layer'):
#             # Extract the layer number
#             layer_num = int(k.split('.')[3])

#             # Duplicate the layer
#             new_layer_num = layer_num + 1
#             new_layer_key = k.replace(f'.{layer_num}.', f'.{new_layer_num}.')
#             new_loaded_model[new_layer_key] = v

#     # Duplicate the optimizer state for the new layers
#     new_loaded_optim = copy.deepcopy(loaded_optim)

#     for layer_num in range(len(loaded_optim['state'])):
#         new_layer_num = layer_num + 1
#         new_layer_state = copy.deepcopy(loaded_optim['state'][layer_num])
#         new_loaded_optim['state'][new_layer_num] = new_layer_state

#     # Insert the new layers and optimizer info after encoder 0
#     insert_index = None
#     for idx, (k, v) in enumerate(new_loaded_model.items()):
#         if k == 'bert.encoder.layer.0.output.LayerNorm.bias':
#             insert_index = idx + 1
#             break

#     new_layers = []
#     for k, v in new_loaded_model.items():
#         if k.startswith('bert.encoder.layer.0.'):
#             new_layer_key = k.replace('bert.encoder.layer.0.', 'bert.encoder.layer.1.')
#             new_layers.append((new_layer_key, copy.deepcopy(v)))

#     if insert_index is not None:
#         new_loaded_model = dict(
#             list(new_loaded_model.items())[:insert_index] +
#             new_layers +
#             list(new_loaded_model.items())[insert_index:]
#         )

#     new_loaded_optim['state'] = copy.deepcopy(new_loaded_optim['state'])[:insert_index] + \
#                                 copy.deepcopy(new_loaded_optim['state'])[:len(new_layers)] + \
#                                 copy.deepcopy(new_loaded_optim['state'])[insert_index:]

#     # Split the optimizer parameter groups based on the rule
#     ALL_LAYERNORM_LAYERS = ['layer_norm.bias', 'layer_norm.weight']
#     decay_parameters = get_parameter_names(new_loaded_model, ALL_LAYERNORM_LAYERS)
#     decay_parameters = [name for name in decay_parameters if "bias" not in name]

#     optimizer_grouped_parameters = [
#         {
#             "params": [
#                 p for n, p in new_loaded_model.items() if (n in decay_parameters and p.requires_grad)
#             ],
#             "weight_decay": 0.1,  # Set the desired weight decay value
#         },
#         {
#             "params": [
#                 p for n, p in new_loaded_model.items() if (n not in decay_parameters and p.requires_grad)
#             ],
#             "weight_decay": 0.0,
#         },
#     ]

#     # Update the optimizer parameter groups in new_loaded_optim
#     new_loaded_optim['param_groups'] = optimizer_grouped_parameters

#     # Update the layer numbers in new_loaded_optim['param_groups'] params list
#     for param_group in new_loaded_optim['param_groups']:
#         layer_ids = param_group['params']
#         updated_layer_ids = [
#             layer_id + len(loaded_optim['param_groups'][0]['params'])
#             if layer_id >= len(loaded_optim['param_groups'][0]['params'])
#             else layer_id
#             for layer_id in layer_ids
#         ]
#         param_group['params'] = updated_layer_ids

#     # new_loaded_optim['param_groups'] = copy.deepcopy(loaded_optim['param_groups']) + \
#     #                                   copy.deepcopy(new_loaded_optim_param_groups)


# def get_parameter_names(model, names):
#     parameter_names = []
#     for name in model.keys():
#         for n in names:
#             if n in name:
#                 parameter_names.append(name)
#     return parameter_names


# if __name__ == "__main__":
#     debugpy.listen(5678)
#     debugpy.wait_for_client()
#     debugpy.breakpoint()
#     main()











# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

# import argparse
# import os
# import copy
# import torch
# from torch import nn

# os.environ['WANDB_MODE'] = 'offline'
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt_path', type=str, help='The path to the model checkpoint')
#     args = parser.parse_args()

#     ckpt_path = args.ckpt_path

#     model_checkpoint_path = os.path.join(ckpt_path, "checkpoint-2", "pytorch_model.bin")
#     loaded_model = torch.load(model_checkpoint_path)

#     optim_checkpoint_path = os.path.join(ckpt_path, "checkpoint-2", "optimizer.pt")
#     loaded_optim = torch.load(optim_checkpoint_path)

#     # Double the encoder layers
#     new_loaded_model = copy.deepcopy(loaded_model)

#     for k, v in loaded_model.items():
#         if k.startswith('bert.encoder.layer'):
#             # Extract the layer number
#             layer_num = int(k.split('.')[3])

#             # Duplicate the layer
#             new_layer_num = layer_num + 1
#             new_layer_key = k.replace(f'.{layer_num}.', f'.{new_layer_num}.')
#             new_loaded_model[new_layer_key] = v

#     # Duplicate the optimizer state for the new layers
#     new_loaded_optim = copy.deepcopy(loaded_optim)

#     for layer_num in range(len(loaded_optim['state'])):
#         new_layer_num = layer_num + 1
#         new_layer_state = copy.deepcopy(loaded_optim['state'][layer_num])
#         new_loaded_optim['state'][new_layer_num] = new_layer_state

#     # Split the optimizer parameter groups based on the rule
#     ALL_LAYERNORM_LAYERS = ['layer_norm.bias', 'layer_norm.weight']
#     decay_parameters = get_parameter_names(new_loaded_model, ALL_LAYERNORM_LAYERS)
#     decay_parameters = [name for name in decay_parameters if "bias" not in name]

#     # Duplicate the optimizer parameter groups for the new layers
#     new_loaded_optim_param_groups = copy.deepcopy(loaded_optim['param_groups'])

#     decay_layer_ids = [
#         layer_id + len(loaded_optim['param_groups'][0]['params'])
#         for layer_id, (_, v) in enumerate(new_loaded_model.items())
#         if v in decay_parameters
#     ]
#     new_loaded_optim_param_groups[0]['params'] += decay_layer_ids

#     non_decay_layer_ids = [
#         layer_id + len(loaded_optim['param_groups'][0]['params']) + len(decay_layer_ids)
#         for layer_id in range(len(new_loaded_model))
#         if layer_id not in decay_layer_ids
#     ]
#     new_loaded_optim_param_groups[1]['params'] += non_decay_layer_ids

#     # Update the optimizer parameter groups in new_loaded_optim
#     new_loaded_optim['param_groups'] = new_loaded_optim_param_groups

#     # Insert the new layers and optimizer info after encoder 0
#     insert_index = None
#     for idx, (k, v) in enumerate(new_loaded_model.items()):
#         if k == 'bert.encoder.layer.0.output.LayerNorm.bias':
#             insert_index = idx + 1
#             break

#     new_layers = []
#     for k, v in new_loaded_model.items():
#         if k.startswith('bert.encoder.layer.0.'):
#             new_layer_key = k.replace('bert.encoder.layer.0.', 'bert.encoder.layer.1.')
#             new_layers.append((new_layer_key, copy.deepcopy(v)))

#     if insert_index is not None:
#         new_loaded_model = dict(
#             list(new_loaded_model.items())[:insert_index] +
#             new_layers +
#             list(new_loaded_model.items())[insert_index:]
#         )

#     new_loaded_optim['state'] = copy.deepcopy(new_loaded_optim['state'])[:insert_index] + \
#                                 copy.deepcopy(new_loaded_optim['state'])[:len(new_layers)] + \
#                                 copy.deepcopy(new_loaded_optim['state'])[insert_index:]

#     # Update the layer numbers in new_loaded_optim['param_groups'] params list
#     for param_group in new_loaded_optim['param_groups']:
#         layer_ids = param_group['params']
#         updated_layer_ids = [
#             layer_id + len(loaded_optim['param_groups'][0]['params']) + len(decay_layer_ids)
#             if layer_id >= len(loaded_optim['param_groups'][0]['params'])
#             else layer_id
#             for layer_id in layer_ids
#         ]
#         param_group['params'] = updated_layer_ids

#     # new_loaded_optim['param_groups'] = copy.deepcopy(loaded_optim['param_groups']) + \
#     #                                   copy.deepcopy(new_loaded_optim_param_groups)


# def get_parameter_names(model, exclude_names):
#     parameter_names = []
#     for name in model.keys():
#         for n in exclude_names:
#             if n in name:
#                 parameter_names.append(name)
#     return parameter_names


# if __name__ == "__main__":
#     main()
















# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

# import argparse
# import os
# import copy

# import torch
# from torch import nn

# os.environ['WANDB_MODE'] = 'offline'
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt_path', type=str, help='The path to the model checkpoint')
#     # parser.add_argument('--optim_path', type=str, help='The path to the optimizer checkpoint')
#     args = parser.parse_args()

#     ckpt_path = args.ckpt_path
#     # optim_path = args.optim_path

#     model_checkpoint_path = os.path.join(ckpt_path, "checkpoint-2", "pytorch_model.bin")
#     loaded_model = torch.load(model_checkpoint_path)

#     optim_checkpoint_path = os.path.join(ckpt_path, "checkpoint-2", "optimizer.pt")
#     loaded_optim = torch.load(optim_checkpoint_path)

#     # Double the encoder layers
#     new_loaded_model = copy.deepcopy(loaded_model)

#     for k, v in loaded_model.items():
#         if k.startswith('bert.encoder.layer'):
#             # Extract the layer number
#             layer_num = int(k.split('.')[3])

#             # Duplicate the layer
#             new_layer_num = layer_num + 1
#             new_layer_key = k.replace(f'.{layer_num}.', f'.{new_layer_num}.')
#             new_loaded_model[new_layer_key] = v

#     # Duplicate the optimizer state for the new layers
#     new_loaded_optim = copy.deepcopy(loaded_optim)

#     for layer_num in range(len(loaded_optim['state'])):
#         new_layer_num = layer_num + 1
#         new_layer_state = copy.deepcopy(loaded_optim['state'][layer_num])
#         new_loaded_optim['state'][new_layer_num] = new_layer_state

#     # # Duplicate the optimizer param groups for the new layers
#     # new_loaded_optim_param_groups = copy.deepcopy(loaded_optim['param_groups'])

#     # for param_group in loaded_optim['param_groups']:
#     #     new_param_group = copy.deepcopy(param_group)
#     #     new_loaded_optim_param_groups.append(new_param_group)

#     # Insert the new layers and optimizer info after encoder 0
#     insert_index = None
#     for idx, (k, v) in enumerate(new_loaded_model.items()):
#         if k == 'bert.encoder.layer.0.output.LayerNorm.bias':
#             insert_index = idx + 1
#             break

#     new_layers = []
#     for k, v in new_loaded_model.items():
#         if k.startswith('bert.encoder.layer.0.'):
#             new_layer_key = k.replace('bert.encoder.layer.0.', 'bert.encoder.layer.1.')
#             new_layers.append((new_layer_key, copy.deepcopy(v)))

#     if insert_index is not None:
#         new_loaded_model = dict(
#             list(new_loaded_model.items())[:insert_index] +
#             new_layers +
#             list(new_loaded_model.items())[insert_index:]
#         )

#     new_loaded_optim['state'] = copy.deepcopy(new_loaded_optim['state'])[:insert_index] + \
#                                  copy.deepcopy(new_loaded_optim['state'])[:len(new_layers)] + \
#                                  copy.deepcopy(new_loaded_optim['state'])[insert_index:]

#     # new_loaded_optim['param_groups'] = copy.deepcopy(loaded_optim['param_groups']) + \
#     #                                    copy.deepcopy(new_loaded_optim_param_groups)

#     for k in new_loaded_model.keys():
#         print(k)

#     print(new_loaded_optim['state'])
#     print(new_loaded_optim['param_groups'])


# if __name__ == "__main__":
#     main()
