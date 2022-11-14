from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from .utils import parse_label, load_jsonl


def load_seqs(
    texts_path: Path,
) -> List[str]:
    """
    Load sequences from JSONL file, parse labels, and labels that are rare
    or combination of multiple radicals are replaced with '…'.
    """
    texts = load_jsonl(texts_path)
    texts = [seq["text"] for seq in texts]
    print(f"Loaded {len(texts)} sequences.")
    texts = [
        "".join(
            [parse_label(c, True, comb_token="…", unk_token="…") for c in seq]
        )
        for seq in texts
    ]
    return texts


def get_train_examples(
    texts: List[str],
    tokenizer,
    block_size,
) -> List[Dict[str, torch.Tensor]]:
    print("====== examples ======")
    for i in range(12):
        if len(texts[i]) > 0:
            print(i, texts[i])
    print("======================")

    batch_encoding: Dict[str, list] = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=block_size,
    )
    input_ids: List[List[int]] = batch_encoding["input_ids"]
    examples = [
        {"input_ids": torch.tensor(e, dtype=torch.long)} for e in input_ids
    ]
    return examples


def load_test_seqs(path: Path):
    """
    Load test examples from JSONL file.
    """
    examples = load_jsonl(path)
    print(f"Loaded {len(examples)} test examples.")
    seqs: List[List[str]] = [example["sequence"] for example in examples]
    labels: List[List[str]] = [example["label"] for example in examples]
    texts = [
        "".join(
            [parse_label(c, True, comb_token="…", unk_token="…") for c in seq]
        )
        for seq in seqs
    ]
    label_texts = [
        "".join(
            [
                parse_label(c, True, comb_token="…", unk_token="…")
                for c in label
            ]
        )
        for label in labels
    ]
    return texts, label_texts


def get_test_examples(
    texts: List[str],
    label_texts: List[str],
    context_len: int,
    tokenizer,
) -> List[Dict[str, torch.Tensor]]:
    """
    Create test examples from texts and label_texts.

    This assumes that `texts` and `label_texts` are List[str],
    and contain the same sequences except but some characters in `texts`
    are replaced with [MASK].

    The sequences will be tokenized in blocks of `context_len` characters.
    """
    input_encoding: Dict[str, torch.Tensor] = tokenizer(
        texts,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=context_len,
        return_tensors="pt",
    )
    input_ids = input_encoding["input_ids"]
    att_mask = input_encoding["attention_mask"]
    label_ids: torch.Tensor = tokenizer(
        label_texts,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=context_len,
        return_tensors="pt",
    )["input_ids"]

    # Remove blocks without any [MASK]
    maskless_indices = torch.where(input_ids == tokenizer.mask_token_id)
    print(input_ids[:3])
    print(label_ids[:3])
    print("mask id:", tokenizer.mask_token_id)
    print("maskless_indices:", maskless_indices)
    exit()

    # Make padding ignored by the loss function (set to -100)
    label_ids[label_ids == 1] = -100
    num_examples = label_ids.shape[0]
    examples = [
        {
            "input_ids": input_ids[i],
            "labels": label_ids[i],
            "attention_mask": att_mask[i],
        }
        for i in range(num_examples)
    ]
    return examples


class ChujianMLMDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        is_training: bool,
        block_size: int = 10,  # Actually 8 because of [CLS] and [SEP]
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.is_training = is_training

        if is_training:
            """Assume that the texts are a list of strings"""
            self.texts = load_seqs(data_path)
            self.examples = get_train_examples(
                self.texts, self.tokenizer, self.block_size
            )
        else:
            # For evaluation
            self.texts, self.label_texts = load_test_seqs(data_path)
            self.examples = get_test_examples(
                self.texts, self.label_texts, block_size, tokenizer
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        return self.examples[i]
