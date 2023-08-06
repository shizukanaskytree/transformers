import os
import logging
import sys
import torch

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

from transformers import pipeline



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.

    ##################
    train from scratch
    ##################
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


# @pysnooper.snoop(os.path.join(log_folder, f"pretty_print-{timestamp}.log"), color=False, max_variable_length=2000)
def pretty_print(token_ids, logits, pred_idx, itos, sentences, language):
    """
    格式化输出结果
    :param token_ids:   [src_len, batch_size]
    :param logits:  [src_len, batch_size, vocab_size]
    :param pred_idx:   二维列表，每个内层列表记录了原始句子中被mask的位置
    :param itos:
    :param sentences: 原始句子
    :return:
    """
    token_ids = token_ids.transpose(0, 1)   # [batch_size, src_len]
    logits = logits.transpose(0, 1)         # [batch_size, src_len, vocab_size]
    y_pred = logits.argmax(axis=2)          # [batch_size, src_len]
    sep = " " if language == 'en' else ""
    for token_id, sentence, y, y_idx in zip(token_ids, sentences, y_pred, pred_idx):
        sen = [itos[id] for id in token_id]
        sen_mask = sep.join(sen).replace(" ##", "").replace("[PAD]", "").replace(" ,", ",")
        sen_mask = sen_mask.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
        logging.info(f" ### 原始: {sentence}")
        logging.info(f"  ## 掩盖: {sen_mask}")
        for idx in y_idx:
            sen[idx] = itos[y[idx]].replace("##", "")
        sen = sep.join(sen).replace("[PAD]", "").replace(" ,", ",")
        sen = sen.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
        logging.info(f"  ## 预测: {sen}")
        logging.info("===============")


logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    (model_args, training_args) = parser.parse_args_into_dataclasses()

    #---------------------------------------------------------------------------

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    #---------------------------------------------------------------------------

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        # from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        # cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        # low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )

    model = model.to(training_args.device)
    model.eval()

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    ### inference tutorial: https://huggingface.co/docs/transformers/main/en/quicktour#autotokenizer

    pt_batch = tokenizer(
        [
            "We are very happy to show you the 🤗 Transformers library.",
            "We hope you don't hate it.",
            "We are very happy to [MASK] you the 🤗 Transformers library."
        ],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    pt_batch.to(training_args.device)

    # print(f"pt_batch: {pt_batch}")
    # print(pt_batch.keys())
    ### dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

    ### pt_batch: {
    ###     'input_ids': tensor([[    2, 25241, 25189, 26140, 29120, 25052, 25758, 25178, 25031, 19569, 29934, 25086, 27665,    18,     3],
    ###                          [    2, 25241, 28358, 25178, 26484,    11,    62, 29959, 19860, 25094,    18,     3,     0,     0,     0]], device='cuda:0'),
    ###     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0'),
    ###     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], device='cuda:0')
    ### }

    # encoding = tokenizer(**pt_batch)
    # print(encoding)
    ### {'input_ids': [[2, 51, 25353, 28006, 26392, 25134, 16, 28148, 16, 25211, 29183, 51, 26392, 25134, 18, 3], [2, 26392, 25092, 25311, 26185, 25050, 29347, 25080, 25053, 25311, 25667, 18, 3]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

    # outputs = model(**pt_batch)
    ### outputs type: MaskedLMOutput
    # print(f"outputs: {outputs}")

    # tokens = model.generate(pt_batch['input_ids'])
    # print(f"tokens: {tokens}")

    # for token_sequence, attention_mask in zip(tokens, pt_batch['attention_mask']):
    #     # Convert tensors to lists
    #     token_sequence = token_sequence.tolist()
    #     attention_mask = attention_mask.tolist()

    #     # Remove tokens where the corresponding attention_mask value is 0
    #     token_sequence = [token for token, mask in zip(token_sequence, attention_mask) if mask == 1]

    #     output = tokenizer.decode(token_sequence, skip_special_tokens=True)
    #     print(f"output: {output}")






if __name__ == '__main__':
    main()