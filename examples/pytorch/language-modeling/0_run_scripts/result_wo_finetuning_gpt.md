```log
(lm-eval) (base) xiaofeng.wu@Fairfax4way04RTX4090:/data/xiaofengwu/sn
(lm-eval) (base) xiaofeng.wu@Fairfax4way04RTX4090:/data/xiaofengwu/snn_llm$ cd transformers/
(lm-eval) (base) xiaofeng.wu@Fairfax4way04RTX4090:/data/xiaofengwu/snn_llm/transformers$ bash 0_run_scripts/eval_wo_finetuning.sh
11/11/2024 14:08:17 - WARNING - __main__ - Process rank: 0, device: cuda:0, n_gpu: 4, distributed training: False, 16-bits training: False
11/11/2024 14:08:17 - INFO - __main__ - Training/evaluation parameters TrainingArguments(
_n_gpu=4,
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
average_tokens_across_devices=False,
batch_eval_metrics=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
dispatch_batches=None,
do_eval=True,
do_predict=False,
do_train=False,
eval_accumulation_steps=None,
eval_delay=0,
eval_do_concat_batches=True,
eval_on_start=False,
eval_steps=None,
eval_strategy=IntervalStrategy.NO,
eval_use_gather_object=False,
evaluation_strategy=None,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=False,
hub_strategy=HubStrategy.EVERY_SAVE,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_for_metrics=[],
include_inputs_for_metrics=False,
include_num_input_tokens_seen=False,
include_tokens_per_second=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=5e-05,
length_column_name=length,
load_best_model_at_end=False,
local_rank=0,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=/tmp/test-clm-wo-finetuning/runs/Nov11_14-08-17_Fairfax4way04RTX4090,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=500,
logging_strategy=IntervalStrategy.STEPS,
lr_scheduler_kwargs={},
lr_scheduler_type=SchedulerType.LINEAR,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=None,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_train_epochs=3.0,
optim=OptimizerNames.ADAMW_TORCH,
optim_args=None,
optim_target_modules=None,
output_dir=/tmp/test-clm-wo-finetuning,
overwrite_output_dir=False,
past_index=-1,
per_device_eval_batch_size=4,
per_device_train_batch_size=4,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=[],
restore_callback_states_from_checkpoint=False,
resume_from_checkpoint=None,
run_name=/tmp/test-clm-wo-finetuning,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=500,
save_strategy=SaveStrategy.STEPS,
save_total_limit=None,
seed=42,
skip_memory_metrics=True,
split_batches=None,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torch_empty_cache_steps=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_cpu=False,
use_ipex=False,
use_legacy_prediction_loop=False,
use_liger_kernel=False,
use_mps_device=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.0,
)
Overwrite dataset info from restored data version if exists.
11/11/2024 14:08:19 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
Loading Dataset info from /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
11/11/2024 14:08:19 - INFO - datasets.info - Loading Dataset info from /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
Found cached dataset wikitext (/home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
11/11/2024 14:08:19 - INFO - datasets.builder - Found cached dataset wikitext (/home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
Loading Dataset info from /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
11/11/2024 14:08:19 - INFO - datasets.info - Loading Dataset info from /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
[INFO|configuration_utils.py:692] 2024-11-11 14:08:19,483 >> loading configuration file config.json from cache at /home/xiaofeng.wu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json
[INFO|configuration_utils.py:759] 2024-11-11 14:08:19,483 >> Model config GPT2Config {
  "_name_or_path": "openai-community/gpt2",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.47.0.dev0",
  "use_cache": true,
  "vocab_size": 50257
}

[INFO|configuration_utils.py:692] 2024-11-11 14:08:19,520 >> loading configuration file config.json from cache at /home/xiaofeng.wu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json
[INFO|configuration_utils.py:759] 2024-11-11 14:08:19,520 >> Model config GPT2Config {
  "_name_or_path": "openai-community/gpt2",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.47.0.dev0",
  "use_cache": true,
  "vocab_size": 50257
}

[INFO|tokenization_utils_base.py:2024] 2024-11-11 14:08:19,528 >> loading file vocab.json from cache at /home/xiaofeng.wu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/vocab.json
[INFO|tokenization_utils_base.py:2024] 2024-11-11 14:08:19,528 >> loading file merges.txt from cache at /home/xiaofeng.wu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/merges.txt
[INFO|tokenization_utils_base.py:2024] 2024-11-11 14:08:19,528 >> loading file tokenizer.json from cache at /home/xiaofeng.wu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json
[INFO|tokenization_utils_base.py:2024] 2024-11-11 14:08:19,528 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2024] 2024-11-11 14:08:19,528 >> loading file special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:2024] 2024-11-11 14:08:19,528 >> loading file tokenizer_config.json from cache at /home/xiaofeng.wu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer_config.json
[INFO|configuration_utils.py:692] 2024-11-11 14:08:19,528 >> loading configuration file config.json from cache at /home/xiaofeng.wu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json
[INFO|configuration_utils.py:759] 2024-11-11 14:08:19,528 >> Model config GPT2Config {
  "_name_or_path": "openai-community/gpt2",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.47.0.dev0",
  "use_cache": true,
  "vocab_size": 50257
}

[INFO|modeling_utils.py:3947] 2024-11-11 14:08:19,608 >> loading weights file model.safetensors from cache at /home/xiaofeng.wu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors
[INFO|configuration_utils.py:1104] 2024-11-11 14:08:19,611 >> Generate config GenerationConfig {
  "bos_token_id": 50256,
  "eos_token_id": 50256
}

[INFO|modeling_utils.py:4813] 2024-11-11 14:08:19,654 >> All model checkpoint weights were used when initializing GPT2LMHeadModel.

[INFO|modeling_utils.py:4821] 2024-11-11 14:08:19,654 >> All the weights of GPT2LMHeadModel were initialized from the model checkpoint at openai-community/gpt2.
If your task is similar to the task the model of the checkpoint was trained on, you can already use GPT2LMHeadModel for predictions without further training.
[INFO|configuration_utils.py:1059] 2024-11-11 14:08:19,689 >> loading configuration file generation_config.json from cache at /home/xiaofeng.wu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/generation_config.json
[INFO|configuration_utils.py:1104] 2024-11-11 14:08:19,689 >> Generate config GenerationConfig {
  "bos_token_id": 50256,
  "eos_token_id": 50256
}

Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-e1756a3e767be5d6.arrow
11/11/2024 14:08:19 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-e1756a3e767be5d6.arrow
Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-fa7b91575c24833c.arrow
11/11/2024 14:08:19 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-fa7b91575c24833c.arrow
Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-8dfc74664f83d311.arrow
11/11/2024 14:08:19 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-8dfc74664f83d311.arrow
Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-0e0eb2573b95e50a.arrow
11/11/2024 14:08:19 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-0e0eb2573b95e50a.arrow
Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-3338ea4cab3179c8.arrow
11/11/2024 14:08:19 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-3338ea4cab3179c8.arrow
Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-206e0aa1d3217029.arrow
11/11/2024 14:08:19 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/xiaofeng.wu/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-206e0aa1d3217029.arrow
11/11/2024 14:08:20 - INFO - __main__ - *** Evaluate ***
[INFO|trainer.py:4162] 2024-11-11 14:08:20,479 >>
***** Running Evaluation *****
[INFO|trainer.py:4164] 2024-11-11 14:08:20,479 >>   Num examples = 240
[INFO|trainer.py:4167] 2024-11-11 14:08:20,479 >>   Batch size = 16
/home/xiaofeng.wu/anaconda3/envs/lm-eval/lib/python3.12/site-packages/torch/nn/parallel/_functions.py:71: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn(
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:05<00:00,  2.77it/s]
***** eval metrics *****
  eval_accuracy           =     0.3791
  eval_loss               =      3.422
  eval_runtime            = 0:00:06.67
  eval_samples            =        240
  eval_samples_per_second =      35.93
  eval_steps_per_second   =      2.246
  perplexity              =    30.6306
[INFO|modelcard.py:449] 2024-11-11 14:08:27,236 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}, 'dataset': {'name': 'wikitext wikitext-2-raw-v1', 'type': 'wikitext', 'args': 'wikitext-2-raw-v1'}}
(lm-eval) (base) xiaofeng.wu@Fairfax4way04RTX4090:/data/xiaofengwu/snn_llm/transformers$
```