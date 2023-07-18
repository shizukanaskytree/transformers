# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

import argparse
import os
import copy

import torch
from torch import nn

os.environ['WANDB_MODE'] = 'offline'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.items():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='The path to the model checkpoint')
    parser.add_argument('--optim_path', type=str, help='The path to the optimizer checkpoint')
    args = parser.parse_args()

    model_path = args.model_path
    optim_path = args.optim_path

    model_checkpoint_path = os.path.join(model_path, "checkpoint-2", "pytorch_model.bin")
    loaded_model = torch.load(model_checkpoint_path)
    # print(f"type(loaded_model): {type(loaded_model)}")

    # i = 0
    # for k, v in loaded_model.items():
    #     print(i, k)
    #     i += 1

    optim_checkpoint_path = os.path.join(model_path, "checkpoint-2", "optimizer.pt")
    loaded_optim = torch.load(optim_checkpoint_path)

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

    # Insert the new layers after encoder 0
    insert_index = None
    for idx, (k, v) in enumerate(new_loaded_model.items()):
        if k == 'bert.encoder.layer.0.output.LayerNorm.bias':
            insert_index = idx + 1
            break

    new_layers = []
    for k, v in new_loaded_model.items():
        if k.startswith('bert.encoder.layer.0.'):
            new_layer_key = k.replace('bert.encoder.layer.0.', 'bert.encoder.layer.1.')
            new_layers.append((new_layer_key, copy.deepcopy(v)))

    if insert_index is not None:
        new_loaded_model = dict(
            list(new_loaded_model.items())[:insert_index] +
            new_layers +
            list(new_loaded_model.items())[insert_index:]
        )

    # for k in new_loaded_model.keys():
    #     print(k)



    # print(loaded_optim)



    # Duplicate the optimizer state for the new layers
    new_loaded_optim = copy.deepcopy(loaded_optim)

    for layer_num in range(len(loaded_optim['state'])):
        new_layer_num = layer_num + 1
        new_layer_state = copy.deepcopy(loaded_optim['state'][layer_num])
        new_loaded_optim['state'][new_layer_num] = new_layer_state

    # Duplicate the optimizer param groups for the new layers
    new_loaded_optim_param_groups = copy.deepcopy(loaded_optim['param_groups'])

    for param_group in loaded_optim['param_groups']:
        new_param_group = copy.deepcopy(param_group)
        new_loaded_optim_param_groups.append(new_param_group)

    new_loaded_optim['state'] = copy.deepcopy(new_loaded_optim['state'])[:insert_index] + \
                                 copy.deepcopy(new_loaded_optim['state'])[:len(new_layers)] + \
                                 copy.deepcopy(new_loaded_optim['state'])[insert_index:]

    new_loaded_optim['param_groups'] = copy.deepcopy(loaded_optim['param_groups']) + \
                                       copy.deepcopy(new_loaded_optim_param_groups)






    # for x in new_loaded_model.keys():
    #     print(x)

    # cnt = 0
    # for layer_name, layer in new_loaded_model.items():
    #     if layer.requires_grad:
    #         cnt += 1
    #         print(layer_name)
    # print(f"cnt: {cnt}")

    # decay_parameters = get_parameter_names(new_loaded_model, forbidden_layer_types=ALL_LAYERNORM_LAYERS)
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # optimizer_grouped_parameters = {
    #     "params": [
    #         p for n, p in new_loaded_model.named_parameters() if (n in decay_parameters and p.requires_grad)
    #     ]
    # }

    #---------------------------------------------------------------------------

    # # Double the encoder layers
    # lst = []
    # for k, v in loaded_model.items():
    #     k_split = k.split('.')
    #     # print(k_split)
    #     ### k_split 的打印结果如下:
    #     ### https://www.notion.so/xiaofengwu/transformer-BERT-layers-22b2d218b328402ca47fbe0e7449e8e4?pvs=4
    #     ### ['bert', 'encoder', 'layer', '0', 'attention', 'self', 'query', 'weight']
    #     ### ['bert', 'encoder', 'layer', '0', 'attention', 'self', 'query', 'bias']


    #     ########################################################
    #     ########################################################
    #     # 保存新的键值对在尾部有问题! 需要对应 optimizer 的 state_dict
    #     ########################################################
    #     ########################################################


    #     if k_split[1] == 'encoder' and k_split[2] == 'layers':
    #         l_id = int(k_split[3])
    #         k_split[3] = str(l_id + 1) ### 为了构造新的数字, 但是其他的部分都不变
    #         new_k = '.'.join(k_split) ### 重新拼接
    #         lst.append([new_k, v.clone()]) ### 保存新的键值对在尾部有问题! 需要对应 optimizer 的 state_dict

    # for k, v in lst: ### 将**新的**键值对添加到原来的字典中
    #     loaded_model[k] = v ### 这里的键值对是新的键值对, 但是值是原来的值

    # # Save the modified model checkpoint
    # new_model_checkpoint_dir = os.path.join(model_path, "doubled_checkpoint", "pytorch_model.bin")
    # os.makedirs(new_model_checkpoint_dir, exist_ok=True)
    # new_model_checkpoint_path = os.path.join(new_model_checkpoint_dir, "pytorch_model.bin")
    # torch.save(loaded_model, new_model_checkpoint_path)

    # #-------------------------------------------------------------

    # encoder_sentence_encoder_layers_1_num_params = 0
    # for k, v in loaded_model.items():
    #     if k.startswith("encoder.sentence_encoder.layers.1"):
    #         encoder_sentence_encoder_layers_1_num_params += v.numel()

    # state = loaded_optim['state']

    # ### Double the optimizer state
    # new_state_dict = {}


    ### loaded_optim is a dict, has `state`` and `param_groups` two keys
    ### loaded_optim['state'] is a dict, has 26 kv pair, 0: {'step': 10, 'exp_avg': tensor, 'exp_avg_sq': tensor}
    ### loaded_optim['param_groups'] is a list with 2 elements, each element is a dict, {'weight_decay': 0.0, 'lr': 4e-05, 'betas': (0.9, 0.999), 'eps': 1e-08, 'correct_bias': True, 'initial_lr': 5e-05, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

    # for k, v in loaded_optim['state_dict'].items():
    #     k_split = k.split('.')
    #     if k_split[0] == 'param_groups':
    #         k_split[1] = str(int(k_split[1]) + 1)
    #     new_k = '.'.join(k_split)
    #     new_state_dict[new_k] = v.clone()

    # Save the modified optimizer checkpoint
    # new_optim_checkpoint_path = os.path.join(model_path, "doubled_checkpoint", "optimizer.pt")

    # torch.save({
    #     'state_dict': new_state_dict,
    #     'param_groups': loaded_optim_checkpoint['param_groups']
    # }, new_optim_checkpoint_path)

if __name__ == '__main__':
    main()
