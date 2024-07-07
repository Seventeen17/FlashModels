import itertools
import logging
import sys
import datasets
import torch
import transformers
from typing import Dict, Sequence
from functools import partial
from flashmodels.utils import jload
from torch.utils.data import Dataset

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union, List, Dict, Any
import torch.nn.functional as F

import copy

def data_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    padding_to: Optional[int] = None,
    bucket_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Args:
        batch(`List[Dict[str, Any]]`): The input data in batch
        tokenizer(`PreTrainedTokenizerBase`): The tokenizer of the model
        padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
            will be padded to the `longest`
        bucket_sizes(`List[int]`, optional): Bucket sizes of sequence for TorchAcc.
    """
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    labels = [torch.tensor(b['labels']) for b in batch]
    attention_mask = [
        torch.ones(len(input_ids[i]), dtype=torch.int64)
        for i in range(len(input_ids))
    ]


    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)


    longest_len = input_ids.shape[-1]
    print(f"longest_len ={longest_len}")
    bucket_data_length = _get_bucket(bucket_sizes, longest_len)
    print(f"padding to={bucket_data_length}")
    input_ids = F.pad(input_ids, (0, bucket_data_length), 'constant', tokenizer.pad_token_id)
    labels = F.pad(labels, (0, bucket_data_length), 'constant', -100)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def _get_bucket(bucket_sizes, data_length):
    cloest_length = sys.maxsize
    for b in bucket_sizes:
        if b == data_length or ((b < cloest_length) and (b > data_length)):
            cloest_length = b

    if cloest_length == sys.maxsize:
        bucket_sizes.append(data_length)
        cloest_length = data_length

    return cloest_length
def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 padding_strategy: str) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding_strategy,
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
def preprocess(sources: Sequence[str],
               tokenizer: transformers.PreTrainedTokenizer,
               padding_strategy: str) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s for s in sources]
    examples_tokenized = _tokenize_fn(examples, tokenizer, padding_strategy)

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, examples_tokenized["input_ids_lens"]):
        label[:source_len] = -100
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 padding_strategy: str,
                 max_seq_length: int):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        logging.warning("Loading data...")
        print(f'data_path={data_path}')
        import json
        sources = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_data = json.loads(line.strip())
                sources.append(line_data["text"])

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, tokenizer, padding_strategy)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def get_hf_dataset_loader(tokenizer, args):
    # if args.dataset_name_or_path.endswith(".json"):
    #     raw_datasets = datasets.load_dataset(
    #         "json", data_files=args.dataset_name_or_path)
    # else:
    #     raw_datasets = datasets.load_dataset(args.dataset_name_or_path,
    #                                          args.dataset_config)

    # column_names = list(raw_datasets["train"].features)
    # text_column_name = "text" if "text" in column_names else column_names[0]

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=args.dataset_name_or_path,
        padding_strategy=args.padding_strategy,
        max_seq_length=args.max_seq_length)

    # DataLoader creation
    train_sampler = None
    data_num_replicas = args.fsdp_num * args.dp_num
    if args.pp_num > 1:
        # disable sampler for now
        # the rank below should be:
        # config.get_mesh().get_dp_rank() * config.get_mesh().get_fsdp_num() \
        # + config.get_mesh().get_fsdp_rank()
        args.disable_train_sampler = True
    if (not args.disable_train_sampler) and (data_num_replicas > 1) \
            and (not args.tp_num > 1):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=(1 if args.tp_num > 1 else data_num_replicas),
            rank=(0 if args.tp_num > 1 else args.global_rank),
            shuffle=True)

    bs = args.micro_batch_size
    if args.tp_num > 1:
        bs *= args.dp_num
    if args.pp_num > 1:
        bs *= args.gradient_accumulation_steps
    if args.padding_strategy == "longest":
        bucket_sizes = [args.max_seq_length // 4 * (i + 1) for i in range(4)]
        data_collator = partial(
            data_collate_fn,
            tokenizer=tokenizer,
            padding_to=args.max_seq_length,
            bucket_sizes=bucket_sizes)
        collector = data_collator
    else:
        collector = transformers.default_data_collator
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        collate_fn=collector,
        sampler=train_sampler,
        drop_last=True)
    return train_dataloader
