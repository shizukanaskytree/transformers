import pysnooper
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

import collections
import sys

import torch

import os
log_folder = "model-probe"
os.makedirs(log_folder, exist_ok=True)

def main():
    ckpt = torch.load(sys.argv[1])

    # print all ckpt keys
    # print(f"ckpt.keys(): {ckpt.keys()}")
    ### ckpt.keys(): dict_keys(['args', 'cfg', 'model', 'criterion', 'optimizer_history', 'task_state', 'extra_state', 'last_optimizer_state'])

    encoder_sentence_encoder_layers_1_num_params = 0

    with open(os.path.join(log_folder, f"ckpt-last_optimizer_state.log"), 'w') as file:
        for key in ckpt.keys():
            print('+' * 80)
            print(f"key: {key}")

            ### special case
            if ckpt[key] is None:
                continue

            if key == 'optimizer_history':
                print(f"key: {key}, value: {ckpt[key]}")
                continue

            if key == 'last_optimizer_state':
                print(f"key: {key}, value: {ckpt[key]}")
                print(type(ckpt[key]))

                for k, v in ckpt[key].items():
                    print(f"k: {k}, v: {v}")
                    if k == 'state':
                        # print(type(v))
                        print('*' * 80)
                        for kk, vv in v.items():
                            # print(f"kk: {kk}, vv: {vv}")
                            print(vv['exp_avg'].shape)
                            print(vv['exp_avg_sq'].shape)
                        print('&' * 80)

            for k, v in ckpt[key].items():
                if isinstance(v, torch.Tensor):
                    content = f"{k},\t{v.shape}"
                    if k.startswith("encoder.sentence_encoder.layers.1"):
                        encoder_sentence_encoder_layers_1_num_params += v.numel()
                else:
                    content = f"{k},\t{v}"

                print(content)
                file.write(content + '\n')

            print('-' * 80)

        print('params num of encoder_sentence_encoder_layers_1_num_params:', encoder_sentence_encoder_layers_1_num_params)
        diff_params = 53819481-46731609
        assert diff_params == encoder_sentence_encoder_layers_1_num_params


if __name__ == '__main__':
    main()
