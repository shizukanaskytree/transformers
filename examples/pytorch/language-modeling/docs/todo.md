# TODO

1. dataset -> wiki and bookcorpus -> tokenizer
2. tokenizer, train, save
3. dataset -> wiki and bookcorpus -> training
4. inference -> /home/xiaofeng.wu/prjs/BertWithPretrained/Tasks/TaskForPretraining.py
5. lr scheduler:
   1. https://github.com/google-research/bert/issues/586
   2. BERT should be pretrained in 2 phases - 90% of training is done with sequence length 128 and 10% is done with sequence length 512 (c.f. "Pre-training tips and caveats" in README.md).
6. hyper param, table summary: https://stephanheijl.com/notes_on_bert.html
7. eval by glue
8. Deploy a HuggingFace model
   https://www.google.com/search?q=pipeline+huggingface+bert+inference+code&newwindow=1&rlz=1C5CHFA_enUS981US983&sxsrf=AB5stBiI9pLRYELAF4q4P0MfG6A_ZLqOMw:1691331059703&ei=86nPZIe5KoXt2roPieOomAo&start=20&sa=N&ved=2ahUKEwjH_oqYm8iAAxWFtlYBHYkxCqMQ8NMDegQIARAO&biw=1440&bih=708&dpr=2


