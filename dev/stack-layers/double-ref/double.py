import pysnooper
import datetime
import os
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_folder = "double/double-py"
os.makedirs(log_folder, exist_ok=True)

import collections
import sys

import torch

# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

def print_nested_keys(dictionary, indent=0):
    for key, value in dictionary.items():
        print('-' * indent + str(key))
        if isinstance(value, dict):
            print_nested_keys(value, indent + 2)


# @pysnooper.snoop(os.path.join(log_folder, f"main-{timestamp}.log"), color=False, max_variable_length=2000)
def main():
    ckpt = torch.load(sys.argv[1])

    ### print all keys from ckpt
    # print_nested_keys(ckpt)

    ### double model layers's state, only encoder layers
    lst = []
    for k, v in ckpt['model'].items():
        k_split = k.split('.')
        if k_split[0] == 'encoder' and k_split[2] == 'layers':
            print(f"k: {k}, v.shape: {v.shape}")
            l_id = int(k_split[3])

            ### 现在是硬编码的, 需要在存的时候设定好.
            # k_split[3] = str(l_id + ckpt['args'].encoder_layers) #  ckpt['args'].encoder_layer is not provided, but it is the current number of encoder layers.
            k_split[3] = str(l_id + 1)

            new_k = '.'.join(k_split)
            # print(f"new_k: {new_k}")
            lst.append([new_k, v.clone()])

    for k, v in lst:
        ckpt['model'][k] = v

    ### ------------------------------

    # total_params = 0

    # for name, param in ckpt['model'].items():
    #     num_params = param.numel()
    #     print(f'{name}: {num_params} parameters')
    #     total_params += num_params

    # total_params = 0
    # total_params_requires_grad = 0

    # for name, param in ckpt['model'].items():
    #     num_params = param.numel()
    #     total_params += num_params

    #     if param.requires_grad:
    #         total_params_requires_grad += num_params

    # print(f'Total parameters: {total_params}')
    # print(f'Total parameters with requires_grad: {total_params_requires_grad}') ### 失效, 为 0.

    ### ------------------------------

    ### Total parameters: 92423002
    ### 是否是有的 param 不需要更新?

    ### all line print, be quick!

    #-------------------------------------------------------------
    encoder_sentence_encoder_layers_1_num_params = 0
    # for key in ckpt.keys():

    for k, v in ckpt['model'].items():
        if k.startswith("encoder.sentence_encoder.layers.1"):
            encoder_sentence_encoder_layers_1_num_params += v.numel()

    ### reset optimizer state
    # Extend tensor size for exp_avg and exp_avg_sq
    if 'last_optimizer_state' in ckpt.keys():
        state = ckpt['last_optimizer_state']['state'].get(0, {})
        if 'exp_avg' in state:
            diff_params = encoder_sentence_encoder_layers_1_num_params # 53819481 - 46731609
            state['exp_avg'] = torch.cat((state['exp_avg'], torch.zeros(diff_params)))
        if 'exp_avg_sq' in state:
            diff_params = encoder_sentence_encoder_layers_1_num_params # 53819481 - 46731609
            state['exp_avg_sq'] = torch.cat((state['exp_avg_sq'], torch.zeros(diff_params)))
        ckpt['last_optimizer_state']['state'][0] = state


    if len(sys.argv) > 3 and sys.argv[3] == '--double-optimizer':
        print('doubling the optimizer')

        ### optimizer state for newly doubled layers
        new_optimizer_state = collections.OrderedDict()
        new_optimizer_state['state'] = collections.OrderedDict() # 置为空
        new_optimizer_state['param_groups'] = [collections.OrderedDict()]

        ### optimizer state that is not needed to be duplicated.
        # 'lr': 7.246376811594203e-05
        # 'bias_correction': True
        # 'betas': (0.9, 0.98)
        # 'eps': 1e-06
        # 'weight_decay': 0.01
        # 'step': 50
        # 'params': [0]

        # for k in ['betas', 'eps', 'weight_decay', 'amsgrad']: ### original keys
        for k in ['lr', 'bias_correction', 'betas', 'eps', 'weight_decay', 'step']:
            new_optimizer_state['param_groups'][0][k] = ckpt['last_optimizer_state']['param_groups'][0][k]


        new_optimizer_state['param_groups'][0]['lr'] = 1e-7
        new_optimizer_state['param_groups'][0]['params'] = []


        ### probe optimizer state
        # for k, v in ckpt['last_optimizer_state']['state'].items():
            # print(f"k: {k}, v: {v}")
            # new_optimizer_state['last_optimizer_state']['state'][k] = v


        # for k, v in ckpt['last_optimizer_state']['param_groups'].items():
        #     print(f"k: {k}, v: {v}")


        # head, layers, tail = [], [], []
        # cnt = 0
        # for k, v in ckpt['last_optimizer_state']['state'].items():
        #     if cnt < 2:
        #         head.append(v)
        #         print(f"head {v['exp_avg'].shape}")
        #     elif cnt < 2 + ckpt['args'].encoder_layers * 8:
        #         layers.append(v)
        #         print(f"layers {v['exp_avg'].shape}")
        #     else:
        #         tail.append(v)
        #         print(f"tail {v['exp_avg'].shape}")
        #     cnt += 1

        # cnt = 0
        # for it in head:
        #     it['step'] = 0
        #     new_optimizer_state['state'][cnt] = it
        #     new_optimizer_state['param_groups'][0]['params'].append(cnt)
        #     cnt += 1

        # for it in layers:
        #     it['step'] = 0
        #     new_optimizer_state['state'][cnt] = it
        #     new_optimizer_state['param_groups'][0]['params'].append(cnt)
        #     cnt += 1

        # for it in layers:
        #     it['step'] = 0
        #     new_optimizer_state['state'][cnt] = it
        #     new_optimizer_state['param_groups'][0]['params'].append(cnt)
        #     cnt += 1

        # for it in tail:
        #     it['step'] = 0
        #     new_optimizer_state['state'][cnt] = it
        #     new_optimizer_state['param_groups'][0]['params'].append(cnt)
        #     cnt += 1

        ckpt['last_optimizer_state'] = new_optimizer_state

    # ckpt['args'].encoder_layers *= 2 # 这是原来的代码, 没有 ckpt['args'] 这个 key
    torch.save(ckpt, sys.argv[2])


if __name__ == '__main__':
    main()
