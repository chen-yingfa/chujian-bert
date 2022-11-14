from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from .utils import parse_label, load_jsonl


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks of size `chunk_size`.
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def chunk_2d_list(lst: List[List], chunk_size: int) -> List[List]:
    """
    Split a 2D list into chunks of size `chunk_size`.
    """
    chunks = []
    for seq in lst:
        chunks.extend(chunk_list(seq, chunk_size))
    return chunks


def load_seqs(
    texts_path: Path,
) -> List[List[str]]:
    """
    Load sequences from JSONL file, parse labels, and labels that are rare
    or combination of multiple radicals are replaced with '…'.
    """
    texts = load_jsonl(texts_path)
    texts = [seq["text"] for seq in texts]
    print(f"Loaded {len(texts)} sequences.")
    texts = [
        [parse_label(c, True, comb_token="…", unk_token="…") for c in seq]
        for seq in texts
    ]
    return texts


def get_train_examples(
    texts: List[List[str]],
    tokenizer,
    context_len: int,
) -> List[Dict[str, torch.Tensor]]:
    print("====== train examples ======")
    for i in range(10):
        print(i, texts[i])
    print("============================")

    # Chunk sequences into blocks of `block_size` characters
    chunk_size = context_len - 2  # -2 for [CLS] and [SEP]
    texts = chunk_2d_list(texts, chunk_size)
    joined_texts = ["".join(x) for x in texts]

    batch_encoding: Dict[str, list] = tokenizer(
        joined_texts,
        add_special_tokens=True,
        truncation=True,
        max_length=context_len,
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
    # Don't join glyphs into a string because we need to
    # know the number of glyphs in each sequence.
    texts = [
        [parse_label(c, True, comb_token="…", unk_token="…") for c in seq]
        for seq in seqs
    ]
    label_texts = [
        [parse_label(c, True, comb_token="…", unk_token="…") for c in label]
        for label in labels
    ]
    return texts, label_texts


def get_test_examples(
    texts: List[List[str]],
    label_texts: List[List[str]],
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
    assert len(texts) == len(label_texts)
    # Split sequences (`texts` and `label_texts`) into chunks
    # of a length of `context_len`, then remove those without mask.
    print(f"# test examples before chunking: {len(texts)}")

    chunked_texts = []
    chunked_label_texts = []
    chunk_size = context_len - 2  # 2 for [CLS] and [SEP]
    for text, label_text in zip(texts, label_texts):
        if len(text) != len(label_text):
            print(
                f"Lengths of texts and label_texts must be the same"
                f"but got {len(text)} and {len(label_text)}."
            )
            print(text, label_text)
            exit()
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if "[MASK]" not in chunk:
                continue
            label_chunk = label_text[i : i + chunk_size]
            chunked_texts.append("".join(chunk))
            chunked_label_texts.append("".join(label_chunk))
    print(f"# test examples after chunking: {len(chunked_texts)}")
    print("====== Test examples ======")
    for i in range(5):
        print(i, "---", chunked_texts[i], "---", chunked_label_texts[i])
    print("===========================")

    def tokenize(texts):
        return tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=context_len,
            return_tensors="pt",
            padding="max_length",
        )

    input_encoding = tokenize(chunked_texts)
    input_ids = input_encoding["input_ids"]
    att_mask = input_encoding["attention_mask"]
    label_ids = tokenize(chunked_label_texts)["input_ids"]

    # Make padding ignored by the loss function (set to -100)
    mask_id = tokenizer.mask_token_id
    label_ids[input_ids != mask_id] = -100
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
