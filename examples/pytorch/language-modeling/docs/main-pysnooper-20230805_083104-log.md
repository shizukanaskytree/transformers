# pysnooper: main func baseline tracing


```bash
Source path:... /home/xiaofeng.wu/prjs/transformers/examples/pytorch/language-modeling/run_mlm.py
08:31:06.483207 call       244 def main():
08:31:06.483409 line       249     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
New var:....... parser = HfArgumentParser(prog='run_mlm.py', usage=None, description=None, formatter_class=<class 'argparse.ArgumentDefaultsHelpFormatter'>, conflict_handler='error', add_help=True)
08:31:06.488930 line       250     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
08:31:06.489012 line       255         model_args, data_args, training_args = parser.parse_args_into_dataclasses()
New var:....... model_args = ModelArguments(model_name_or_path=None, model_type=None, config_overrides=None, config_name='bert-base-uncased', tokenizer_name='skytree/tokenizer-bert-wiki-bookcorpus', cache_dir=None, use_fast_tokenizer=True, model_revision='main', use_auth_token=False, low_cpu_mem_usage=False)
New var:....... data_args = DataTrainingArguments(dataset_name='wikitext', dataset_config_name='wikitext-2-raw-v1', train_file=None, validation_file=None, overwrite_cache=False, validation_split_percentage=5, max_seq_length=None, preprocessing_num_workers=None, mlm_probability=0.15, line_by_line=False, pad_to_max_length=False, max_train_samples=None, max_eval_samples=None, streaming=False)
New var:....... training_args = TrainingArguments(_n_gpu=4,adafactor=False,adam_beta1=0.9,adam_beta2=0.999,adam_epsilon=1e-08,auto_find_batch_size=False,bf16=False,bf16_full_eval=False,data_seed=None,dataloader_drop_last=False,dataloader_num_workers=0,dataloader_pin_memory=True,ddp_backend=None,ddp_broadcast_buffers=None,ddp_bucket_cap_mb=None,ddp_find_unused_parameters=None,ddp_timeout=1800,debug=[],deepspeed=None,disable_tqdm=False,do_eval=True,do_predict=False,do_train=True,eval_accumulation_steps=None,eval_delay=0,eval_steps=100,evaluation_strategy=IntervalStrategy.STEPS,fp16=False,fp16_backend=auto,fp16_full_eval=False,fp16_opt_level=O1,fsdp=[],fsdp_config={'fsdp_min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},fsdp_min_num_params=0,fsdp_transformer_layer_cls_to_wrap=None,full_determinism=False,gradient_accumulation_steps=1,gradient_checkpointing=False,greater_is_better=None,group_by_length=False,half_precision_backend=auto,hub_model_id=None,hub_private_repo=False,hub_strategy=HubStrategy.EVERY_S...=,no_cuda=False,num_train_epochs=3.0,optim=OptimizerNames.ADAMW_HF,optim_args=None,output_dir=/home/xiaofeng.wu/prjs/ckpts/bert-base-uncased,overwrite_output_dir=True,past_index=-1,per_device_eval_batch_size=16,per_device_train_batch_size=16,prediction_loss_only=False,push_to_hub=False,push_to_hub_model_id=None,push_to_hub_organization=None,push_to_hub_token=<PUSH_TO_HUB_TOKEN>,ray_scope=last,remove_unused_columns=True,report_to=['wandb'],resume_from_checkpoint=/home/xiaofeng.wu/prjs/ckpts/bert-base-uncased,run_name=bert-base-uncased_20230805_083104,save_on_each_node=False,save_safetensors=False,save_steps=500,save_strategy=IntervalStrategy.STEPS,save_total_limit=3,seed=42,sharded_ddp=[],skip_memory_metrics=True,tf32=None,torch_compile=False,torch_compile_backend=None,torch_compile_mode=None,torchdynamo=None,tpu_metrics_debug=False,tpu_num_cores=None,use_ipex=False,use_legacy_prediction_loop=False,use_mps_device=False,warmup_ratio=0.0,warmup_steps=0,weight_decay=0.0,xpu_backend=None,)
08:31:06.517417 line       259     send_example_telemetry("run_mlm", model_args, data_args)
08:31:06.632555 line       262     logging.basicConfig(
08:31:06.633108 line       263         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
08:31:06.633334 line       264         datefmt="%m/%d/%Y %H:%M:%S",
08:31:06.633546 line       265         handlers=[logging.StreamHandler(sys.stdout)],
08:31:06.633780 line       262     logging.basicConfig(
08:31:06.634020 line       268     if training_args.should_log:
08:31:06.634243 line       270         transformers.utils.logging.set_verbosity_info()
08:31:06.634709 line       272     log_level = training_args.get_process_log_level()
New var:....... log_level = 20
08:31:06.634939 line       273     logger.setLevel(log_level)
08:31:06.635234 line       274     datasets.utils.logging.set_verbosity(log_level)
08:31:06.635503 line       275     transformers.utils.logging.set_verbosity(log_level)
08:31:06.635766 line       276     transformers.utils.logging.enable_default_handler()
08:31:06.635966 line       277     transformers.utils.logging.enable_explicit_format()
08:31:06.636170 line       280     logger.warning(
08:31:06.636360 line       281         f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
08:31:06.636589 line       282         + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
08:31:06.636778 line       281         f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
08:31:06.636974 line       280     logger.warning(
08:31:06.637294 line       285     logger.info(f"Training/evaluation parameters {training_args}")
08:31:06.637996 line       288     last_checkpoint = None
New var:....... last_checkpoint = None
08:31:06.638200 line       289     if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
08:31:06.638418 line       303     set_seed(training_args.seed)
08:31:06.639074 line       314     if data_args.dataset_name is not None:
08:31:06.639290 line       316         raw_datasets = load_dataset(
08:31:06.639493 line       317             data_args.dataset_name,
08:31:06.639682 line       318             data_args.dataset_config_name,
08:31:06.639872 line       319             cache_dir=model_args.cache_dir,
08:31:06.640060 line       320             use_auth_token=True if model_args.use_auth_token else None,
08:31:06.640245 line       321             streaming=data_args.streaming,
08:31:06.640431 line       316         raw_datasets = load_dataset(
New var:....... raw_datasets = DatasetDict({    test: Dataset({        features: ['text'],        num_rows: 4358    })    train: Dataset({        features: ['text'],        num_rows: 36718    })    validation: Dataset({        features: ['text'],        num_rows: 3760    })})
08:31:07.049600 line       323         if "validation" not in raw_datasets.keys():
08:31:07.049949 line       383         "cache_dir": model_args.cache_dir,
08:31:07.050183 line       384         "revision": model_args.model_revision,
08:31:07.050408 line       385         "use_auth_token": True if model_args.use_auth_token else None,
08:31:07.050626 line       382     config_kwargs = {
New var:....... config_kwargs = {'cache_dir': None, 'revision': 'main', 'use_auth_token': None}
08:31:07.050836 line       388     """
08:31:07.051062 line       392     if model_args.config_name:
08:31:07.051276 line       393         config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
New var:....... config = BertConfig {  "_name_or_path": "bert-base-uncased",  "architectures": [    "BertForMaskedLM"  ],  "attention_probs_dropout_prob": 0.1,  "classifier_dropout": null,  "gradient_checkpointing": false,  "hidden_act": "gelu",  "hidden_dropout_prob": 0.1,  "hidden_size": 768,  "initializer_range": 0.02,  "intermediate_size": 3072,  "layer_norm_eps": 1e-12,  "max_position_embeddings": 512,  "model_type": "bert",  "num_attention_heads": 12,  "num_hidden_layers": 12,  "pad_token_id": 0,  "position_embedding_type": "absolute",  "transformers_version": "4.31.0.dev0",  "type_vocab_size": 2,  "use_cache": true,  "vocab_size": 30522}
08:31:07.073689 line       405         "cache_dir": model_args.cache_dir,
08:31:07.074175 line       406         "use_fast": model_args.use_fast_tokenizer,
08:31:07.074619 line       407         "revision": model_args.model_revision,
08:31:07.075043 line       408         "use_auth_token": True if model_args.use_auth_token else None,
08:31:07.075506 line       404     tokenizer_kwargs = {
New var:....... tokenizer_kwargs = {'cache_dir': None, 'use_fast': True, 'revision': 'main', 'use_auth_token': None}
08:31:07.075927 line       410     if model_args.tokenizer_name:
08:31:07.076356 line       411         tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
New var:....... tokenizer = BertTokenizerFast(name_or_path='skytree/tokenizer-bert-wiki-bookcorpus', vocab_size=30522, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True)
08:31:07.115247 line       420     if model_args.model_name_or_path:
08:31:07.115737 line       434         logger.info("Training new model from scratch")
08:31:07.116232 line       435         model = AutoModelForMaskedLM.from_config(config)
New var:....... model = BertForMaskedLM(  (bert): BertModel(    (embeddings): BertEmbeddings(      (word_embeddings): Embedding(30522, 768, padding_idx=0)      (position_embeddings): Embedding(512, 768)      (token_type_embeddings): Embedding(2, 768)      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)      (dropout): Dropout(p=0.1, inplace=False)    )    (encoder): BertEncoder(      (layer): ModuleList(        (0-11): 12 x BertLayer(          (attention): BertAttention(            (self): BertSelfAttention(              (query): Linear(in_features=768, out_features=768, bias=True)              (key): Linear(in_features=768, out_features=768, bias=True)              (value): Linear(in_features=768, out_features=768, bias=True)              (dropout): Dropout(p=0.1, inplace=False)            )            (output): BertSelfOutput(              (dense): Linear(in_features=768, out_features=768, bias=True)              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)              (dropout): Dropout(p=0.1, inplace=False)            )          )          (intermediate): BertIntermediate(            (dense): Linear(in_features=768, out_features=3072, bias=True)            (intermediate_act_fn): GELUActivation()          )          (output): BertOutput(            (dense): Linear(in_features=3072, out_features=768, bias=True)            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)            (dropout): Dropout(p=0.1, inplace=False)          )        )      )    )  )  (cls): BertOnlyMLMHead(    (predictions): BertLMPredictionHead(      (transform): BertPredictionHeadTransform(        (dense): Linear(in_features=768, out_features=768, bias=True)        (transform_act_fn): GELUActivation()        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)      )      (decoder): Linear(in_features=768, out_features=30522, bias=True)    )  ))
08:31:08.068771 line       439     embedding_size = model.get_input_embeddings().weight.shape[0]
New var:....... embedding_size = 30522
08:31:08.070060 line       440     if len(tokenizer) > embedding_size:
08:31:08.071069 line       445     if training_args.do_train:
08:31:08.071965 line       446         column_names = list(raw_datasets["train"].features)
New var:....... column_names = ['text']
08:31:08.072944 line       449     text_column_name = "text" if "text" in column_names else column_names[0]
New var:....... text_column_name = 'text'
08:31:08.073870 line       451     if data_args.max_seq_length is None:
08:31:08.074761 line       452         max_seq_length = tokenizer.model_max_length
New var:....... max_seq_length = 512
08:31:08.075606 line       453         if max_seq_length > 1024:
08:31:08.076480 line       468     if data_args.line_by_line:
08:31:08.077350 line       507         def tokenize_function(examples):
New var:....... tokenize_function = <function main.<locals>.tokenize_function at 0x7fa99ed3d260>
08:31:08.078215 line       510         with training_args.main_process_first(desc="dataset map tokenization"):
08:31:08.079123 line       511             if not data_args.streaming:
08:31:08.079985 line       512                 tokenized_datasets = raw_datasets.map(
08:31:08.080838 line       513                     tokenize_function,
08:31:08.081705 line       514                     batched=True,
08:31:08.082551 line       515                     num_proc=data_args.preprocessing_num_workers,
08:31:08.083392 line       516                     remove_columns=column_names,
08:31:08.084226 line       517                     load_from_cache_file=not data_args.overwrite_cache,
08:31:08.085072 line       518                     desc="Running tokenizer on every text in dataset",
08:31:08.085903 line       512                 tokenized_datasets = raw_datasets.map(
New var:....... tokenized_datasets = DatasetDict({    test: Dataset({        features: ['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'],        num_rows: 4358    })    train: Dataset({        features: ['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'],        num_rows: 36718    })    validation: Dataset({        features: ['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'],        num_rows: 3760    })})
08:31:08.103850 line       510         with training_args.main_process_first(desc="dataset map tokenization"):
08:31:08.104805 line       529         def group_texts(examples):
New var:....... group_texts = <function main.<locals>.group_texts at 0x7fa99c5f3e20>
08:31:08.105711 line       550         with training_args.main_process_first(desc="grouping texts together"):
08:31:08.106631 line       551             if not data_args.streaming:
08:31:08.107499 line       552                 tokenized_datasets = tokenized_datasets.map(
08:31:08.108362 line       553                     group_texts,
08:31:08.109231 line       554                     batched=True,
08:31:08.110090 line       555                     num_proc=data_args.preprocessing_num_workers,
08:31:08.110950 line       556                     load_from_cache_file=not data_args.overwrite_cache,
08:31:08.111795 line       557                     desc=f"Grouping texts in chunks of {max_seq_length}",
08:31:08.112646 line       552                 tokenized_datasets = tokenized_datasets.map(
Modified var:.. tokenized_datasets = DatasetDict({    test: Dataset({        features: ['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'],        num_rows: 663    })    train: Dataset({        features: ['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'],        num_rows: 5602    })    validation: Dataset({        features: ['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'],        num_rows: 578    })})
08:31:08.126252 line       550         with training_args.main_process_first(desc="grouping texts together"):
08:31:08.127360 line       565     if training_args.do_train:
08:31:08.128235 line       566         if "train" not in tokenized_datasets:
08:31:08.129102 line       568         train_dataset = tokenized_datasets["train"]
New var:....... train_dataset = Dataset({    features: ['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'],    num_rows: 5602})
08:31:08.129965 line       569         if data_args.max_train_samples is not None:
08:31:08.130849 line       573     if training_args.do_eval:
08:31:08.131722 line       574         if "validation" not in tokenized_datasets:
08:31:08.132578 line       576         eval_dataset = tokenized_datasets["validation"]
New var:....... eval_dataset = Dataset({    features: ['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask'],    num_rows: 578})
08:31:08.133441 line       577         if data_args.max_eval_samples is not None:
08:31:08.134304 line       581         def preprocess_logits_for_metrics(logits, labels):
New var:....... preprocess_logits_for_metrics = <function main.<locals>.preprocess_logits_for_metrics at 0x7fa99c5f39c0>
08:31:08.135163 line       588         metric = evaluate.load("accuracy")
New var:....... metric = EvaluationModule(name: "accuracy", module_type: "metric", features: {'predictions': Value(dtype='int32', id=None), 'references': Value(dtype='int32', id=None)}, usage: """Args:    predictions (`list` of `int`): Predicted labels.    references (`list` of `int`): Ground truth labels.    normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.    sample_weight (`list` of `float`): Sample weights Defaults to None.Returns:    accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.Examples:    Example 1-A simple example        >>> accuracy_metric = evaluate.load("accuracy")        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])        >>> print(results)        {'accuracy': 0.5}    Example 2-The same as Example 1, except with `normalize` set to `False`.        >>> accuracy_metric = evaluate.load("accuracy")        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)        >>> print(results)        {'accuracy': 3.0}    Example 3-The same as Example 1, except with `sample_weight` set.        >>> accuracy_metric = evaluate.load("accuracy")        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])        >>> print(results)        {'accuracy': 0.8778625954198473}""", stored examples: 0)
08:31:08.348099 line       590         def compute_metrics(eval_preds):
New var:....... compute_metrics = <function main.<locals>.compute_metrics at 0x7fa99c5f3380>
08:31:08.349290 line       603     pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
New var:....... pad_to_multiple_of_8 = False
08:31:08.350253 line       604     data_collator = DataCollatorForLanguageModeling(
08:31:08.351177 line       605         tokenizer=tokenizer,
08:31:08.352075 line       606         mlm_probability=data_args.mlm_probability,
08:31:08.352980 line       607         pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
08:31:08.353879 line       604     data_collator = DataCollatorForLanguageModeling(
New var:....... data_collator = DataCollatorForLanguageModeling(tokenizer=BertTokenizerFast(name_or_path='skytree/tokenizer-bert-wiki-bookcorpus', vocab_size=30522, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True), mlm=True, mlm_probability=0.15, pad_to_multiple_of=None, tf_experimental_compile=False, return_tensors='pt')
08:31:08.354788 line       611     trainer = Trainer(
08:31:08.355716 line       612         model=model,
08:31:08.356621 line       613         args=training_args,
08:31:08.357524 line       614         train_dataset=train_dataset if training_args.do_train else None,
08:31:08.358430 line       615         eval_dataset=eval_dataset if training_args.do_eval else None,
08:31:08.359331 line       616         tokenizer=tokenizer,
08:31:08.360213 line       617         data_collator=data_collator,
08:31:08.361120 line       618         compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
08:31:08.361994 line       620         if training_args.do_eval and not is_torch_tpu_available()
08:31:08.362873 line       619         preprocess_logits_for_metrics=preprocess_logits_for_metrics
08:31:08.363748 line       620         if training_args.do_eval and not is_torch_tpu_available()
08:31:08.364622 line       619         preprocess_logits_for_metrics=preprocess_logits_for_metrics
08:31:08.365961 line       611     trainer = Trainer(
New var:....... trainer = <transformers.trainer.Trainer object at 0x7fa99c6a2790>
08:31:08.983271 line       625     if training_args.do_train:
08:31:08.984569 line       626         checkpoint = None
New var:....... checkpoint = None
08:31:08.985683 line       627         if training_args.resume_from_checkpoint is not None:
08:31:08.986746 line       628             checkpoint = training_args.resume_from_checkpoint
Modified var:.. checkpoint = '/home/xiaofeng.wu/prjs/ckpts/bert-base-uncased'
08:31:08.987737 line       631         train_result = trainer.train(resume_from_checkpoint=checkpoint)
New var:....... train_result = TrainOutput(global_step=264, training_loss=0.0, metrics={'train_runtime': 2.6466, 'train_samples_per_second': 6350.074, 'train_steps_per_second': 99.751, 'train_loss': 0.0, 'epoch': 3.0})
08:31:11.935920 line       632         trainer.save_model()  # Saves the tokenizer too for easy upload
Modified var:.. config = BertConfig {  "_name_or_path": "bert-base-uncased",  "architectures": [    "BertForMaskedLM"  ],  "attention_probs_dropout_prob": 0.1,  "classifier_dropout": null,  "gradient_checkpointing": false,  "hidden_act": "gelu",  "hidden_dropout_prob": 0.1,  "hidden_size": 768,  "initializer_range": 0.02,  "intermediate_size": 3072,  "layer_norm_eps": 1e-12,  "max_position_embeddings": 512,  "model_type": "bert",  "num_attention_heads": 12,  "num_hidden_layers": 12,  "pad_token_id": 0,  "position_embedding_type": "absolute",  "torch_dtype": "float32",  "transformers_version": "4.31.0.dev0",  "type_vocab_size": 2,  "use_cache": true,  "vocab_size": 30522}
08:31:12.394539 line       633         metrics = train_result.metrics
New var:....... metrics = {'train_runtime': 2.6466, 'train_samples_per_second': 6350.074, 'train_steps_per_second': 99.751, 'train_loss': 0.0, 'epoch': 3.0}
08:31:12.395824 line       636             data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
08:31:12.396758 line       635         max_train_samples = (
New var:....... max_train_samples = 5602
08:31:12.397670 line       638         metrics["train_samples"] = min(max_train_samples, len(train_dataset))
Modified var:.. train_result = TrainOutput(global_step=264, training_loss=0.0, metrics={'train_runtime': 2.6466, 'train_samples_per_second': 6350.074, 'train_steps_per_second': 99.751, 'train_loss': 0.0, 'epoch': 3.0, 'train_samples': 5602})
Modified var:.. metrics = {'train_runtime': 2.6466, 'train_samples_per_second': 6350.074, 'train_steps_per_second': 99.751, 'train_loss': 0.0, 'epoch': 3.0, 'train_samples': 5602}
08:31:12.398583 line       640         trainer.log_metrics("train", metrics)
08:31:12.400012 line       641         trainer.save_metrics("train", metrics)
08:31:12.401241 line       642         trainer.save_state()
08:31:12.402879 line       645     if training_args.do_eval:
08:31:12.403811 line       646         logger.info("*** Evaluate ***")
08:31:12.404853 line       648         metrics = trainer.evaluate()
Modified var:.. metrics = {'eval_loss': 6.545123100280762, 'eval_accuracy': 0.10951720951720952, 'eval_runtime': 7.1218, 'eval_samples_per_second': 81.159, 'eval_steps_per_second': 1.404, 'epoch': 3.0}
08:31:19.530030 line       650         max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
New var:....... max_eval_samples = 578
08:31:19.531283 line       651         metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
Modified var:.. metrics = {'eval_loss': 6.545123100280762, 'eval_accuracy': 0.10951720951720952, 'eval_runtime': 7.1218, 'eval_samples_per_second': 81.159, 'eval_steps_per_second': 1.404, 'epoch': 3.0, 'eval_samples': 578}
08:31:19.532268 line       652         try:
08:31:19.533255 line       653             perplexity = math.exp(metrics["eval_loss"])
New var:....... perplexity = 695.8423320639565
08:31:19.534173 line       656         metrics["perplexity"] = perplexity
Modified var:.. metrics = {'eval_loss': 6.545123100280762, 'eval_accuracy': 0.10951720951720952, 'eval_runtime': 7.1218, 'eval_samples_per_second': 81.159, 'eval_steps_per_second': 1.404, 'epoch': 3.0, 'eval_samples': 578, 'perplexity': 695.8423320639565}
08:31:19.535100 line       658         trainer.log_metrics("eval", metrics)
08:31:19.536531 line       659         trainer.save_metrics("eval", metrics)
08:31:19.537740 line       661     kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
New var:....... kwargs = {'finetuned_from': None, 'tasks': 'fill-mask'}
08:31:19.538673 line       662     if data_args.dataset_name is not None:
08:31:19.539656 line       663         kwargs["dataset_tags"] = data_args.dataset_name
Modified var:.. kwargs = {'finetuned_from': None, 'tasks': 'fill-mask', 'dataset_tags': 'wikitext'}
08:31:19.540558 line       664         if data_args.dataset_config_name is not None:
08:31:19.541533 line       665             kwargs["dataset_args"] = data_args.dataset_config_name
Modified var:.. kwargs = {'finetuned_from': None, 'tasks': 'fill-mask', 'dataset_tags': 'wikitext', 'dataset_args': 'wikitext-2-raw-v1'}
08:31:19.542435 line       666             kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
Modified var:.. kwargs = {'finetuned_from': None, 'tasks': 'fill-mask', 'dataset_tags': 'wikitext', 'dataset_args': 'wikitext-2-raw-v1', 'dataset': 'wikitext wikitext-2-raw-v1'}
08:31:19.543333 line       670     if training_args.push_to_hub:
08:31:19.544241 line       673         trainer.create_model_card(**kwargs)
08:31:19.813357 return     673         trainer.create_model_card(**kwargs)
Return value:.. None
Elapsed time: 00:00:13.332169
```
