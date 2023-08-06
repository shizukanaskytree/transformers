# load wiki dataset

Doc: https://huggingface.co/datasets/wikitext

```sh
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
```

```py
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
```