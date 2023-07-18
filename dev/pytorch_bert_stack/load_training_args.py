import torch

def main():
    training_args_ckpt_path = "./pretrained-bert/checkpoint-66/training_args.bin"
    ckpt = torch.load(training_args_ckpt_path)
    print(type(ckpt))
    print(ckpt)

if __name__ == '__main__':
    main()

###########################################

### 设置的内容:
# training_args = TrainingArguments(
#     output_dir=model_path,          # output directory to where save model checkpoint
#     evaluation_strategy="steps",    # evaluate each `logging_steps` steps
#     overwrite_output_dir=True,
#     num_train_epochs=10,            # number of training epochs, feel free to tweak, original code settting is 10
#     per_device_train_batch_size=1, # the training batch size, put it as high as your GPU memory fits
#     gradient_accumulation_steps=2,  # accumulating the gradients before updating the weights
#     per_device_eval_batch_size=1,  # evaluation batch size
#     logging_steps=1,             # evaluate, log and save model checkpoints every 1000 step, original 1000, for debug and testing 1
#     save_steps=1,                   # original 1000, for debug and testing 1
#     # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
#     save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
# )

###########################################
### Output:

# (base) wxf@Xiaofengs-MacBook-Pro:~/Documents/prjs/2023/transformers/dev/pytorch_bert_stack_cpu$ python load_training_args.py
# <class 'transformers.training_args.TrainingArguments'>
#
# TrainingArguments(
# _n_gpu=0,
# adafactor=False,
# adam_beta1=0.9,
# adam_beta2=0.999,
# adam_epsilon=1e-08,
# auto_find_batch_size=False,
# bf16=False,
# bf16_full_eval=False,
# data_seed=None,
# dataloader_drop_last=False,
# dataloader_num_workers=0,
# dataloader_pin_memory=True,
# ddp_backend=None,
# ddp_broadcast_buffers=None,
# ddp_bucket_cap_mb=None,
# ddp_find_unused_parameters=None,
# ddp_timeout=1800,
# debug=[],
# deepspeed=None,
# disable_tqdm=False,
# do_eval=True,
# do_predict=False,
# do_train=False,
# eval_accumulation_steps=None,
# eval_delay=0,
# eval_steps=1,
# evaluation_strategy=steps,
# fp16=False,
# fp16_backend=auto,
# fp16_full_eval=False,
# fp16_opt_level=O1,
# fsdp=[],
# fsdp_config={'fsdp_min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
# fsdp_min_num_params=0,
# fsdp_transformer_layer_cls_to_wrap=None,
# full_determinism=False,
# gradient_accumulation_steps=2,
# gradient_checkpointing=False,
# greater_is_better=None,
# group_by_length=False,
# half_precision_backend=auto,
# hub_model_id=None,
# hub_private_repo=False,
# hub_strategy=every_save,
# hub_token=<HUB_TOKEN>,
# ignore_data_skip=False,
# include_inputs_for_metrics=False,
# jit_mode_eval=False,
# label_names=None,
# label_smoothing_factor=0.0,
# learning_rate=5e-05,
# length_column_name=length,
# load_best_model_at_end=False,
# local_rank=0,
# log_level=passive,
# log_level_replica=warning,
# log_on_each_node=True,
# logging_dir=pretrained-bert/runs/Jul17_15-04-59_Xiaofengs-MacBook-Pro.local,
# logging_first_step=False,
# logging_nan_inf_filter=True,
# logging_steps=1,
# logging_strategy=steps,
# lr_scheduler_type=linear,
# max_grad_norm=1.0,
# max_steps=-1,
# metric_for_best_model=None,
# mp_parameters=,
# no_cuda=False,
# num_train_epochs=10,
# optim=adamw_hf,
# optim_args=None,
# output_dir=pretrained-bert,
# overwrite_output_dir=True,
# past_index=-1,
# per_device_eval_batch_size=1,
# per_device_train_batch_size=1,
# prediction_loss_only=False,
# push_to_hub=False,
# push_to_hub_model_id=None,
# push_to_hub_organization=None,
# push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
# ray_scope=last,
# remove_unused_columns=True,
# report_to=['tensorboard', 'wandb'],
# resume_from_checkpoint=None,
# run_name=pretrained-bert,
# save_on_each_node=False,
# save_safetensors=False,
# save_steps=1,
# save_strategy=steps,
# save_total_limit=3,
# seed=42,
# sharded_ddp=[],
# skip_memory_metrics=True,
# tf32=None,
# torch_compile=False,
# torch_compile_backend=None,
# torch_compile_mode=None,
# torchdynamo=None,
# tpu_metrics_debug=False,
# tpu_num_cores=None,
# use_ipex=False,
# use_legacy_prediction_loop=False,
# use_mps_device=False,
# warmup_ratio=0.0,
# warmup_steps=0,
# weight_decay=0.0,
# xpu_backend=None,
# )
# (base) wxf@Xiaofengs-MacBook-Pro:~/Documents/prjs/2023/transformers/dev/pytorch_bert_stack_cpu$