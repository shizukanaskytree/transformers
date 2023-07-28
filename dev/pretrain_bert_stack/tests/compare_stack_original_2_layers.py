import os
import torch
from pprint import pprint
import copy
# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

ckpt_folder = "./pretrained-bert-2-layers/checkpoint-1"
stack_ckpt_folder = "./pretrained-bert/checkpoint-1-stack"


def stack_optim_ckpt():
    ckpt_name = 'optimizer.pt'
    stack_ckpt_optim_path = os.path.join(stack_ckpt_folder, ckpt_name)
    stack_optim_ckpt = torch.load(stack_ckpt_optim_path)
    print_nested_dict(stack_optim_ckpt)


def original_optim_ckpt():
    ckpt_name = 'optimizer.pt'

    ckpt_optim_path = os.path.join(ckpt_folder, ckpt_name)
    # stack_ckpt_optim_path = os.path.join(stack_ckpt_folder, ckpt_name)

    optim_ckpt = torch.load(ckpt_optim_path)
    # stack_optim_ckpt = torch.load(stack_ckpt_optim_path)

    ### optim_ckpt
    # print_nested_dict(optim_ckpt)

    ### original optim_ckpt
    print_nested_dict(stack_optim_ckpt)


    # print(optim_ckpt.keys())
    # print(optim_ckpt['state'])

    # for k, v in optim_ckpt['state'].items():
    #     print(k)
    #     for kk, vv in v.items():
    #         print(kk)

    # for k, v in optim_ckpt.items():
    #     print(k)
    #     for kk, vv in v.items():
    #         print(kk)


def original_model_ckpt():
    ckpt_name = 'pytorch_model.bin'
    ckpt_model_path = os.path.join(ckpt_folder, ckpt_name)
    model_ckpt = torch.load(ckpt_model_path)

    all_names = []
    for x in model_ckpt.keys():
        print(x)

    # cnt_added = 0
    # new_model_ckpt = copy.deepcopy(model_ckpt)
    # extra_model_ckpt = {}
    # target_layer = 1
    # for name, p in model_ckpt.items():
    #     # print(name, p.shape)
    #     if 'bert.encoder.layer.0' in name:
    #         cnt_added += 1
    #         # Split the original string into different parts
    #         parts = name.split(".")

    #         # Update the layer number in the string
    #         parts[3] = str(target_layer)

    #         # Reconstruct the updated string
    #         updated_name = ".".join(parts)
    #         extra_model_ckpt[updated_name] = p.clone()

    # # print(f"cnt_added: {cnt_added}")
    # new_model_ckpt.update(extra_model_ckpt)
    # # print(new_model_ckpt.keys())
    # # for x in new_model_ckpt.keys():
    #     # print(x)

    # # new_model_ckpt
    # torch.save(new_model_ckpt, stack_model_path)
    # print(f"saved to {stack_model_path}")


def print_nested_dict(dictionary, indent=2):
    for key, value in dictionary.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_nested_dict(value, indent*2)
        else:
            if isinstance(value, torch.Tensor):
                print(' ' * (indent*2) + str(value.shape))
            else:
                print(' ' * (indent*2) + str(value))


if __name__ == '__main__':
    # original_model_ckpt()
    # original_optim_ckpt()
    stack_optim_ckpt()
