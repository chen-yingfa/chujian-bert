from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import parse_label, load_jsonl, dump_json


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


def load_train_texts(
    texts_path: Path,
) -> List[List[str]]:
    """
    Load sequences from JSONL file, parse labels, and labels that are rare
    or combination of multiple radicals are replaced with '…'.
    """
    texts = load_jsonl(texts_path)
    print(f"Loaded {len(texts)} sequences.")
    texts = [
        [parse_label(c, False, unk_token="…") for c in seq]
        for seq in texts
    ]
    return texts


def load_test_texts(path: Path):
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
        [parse_label(c, False, unk_token="…") for c in seq]
        for seq in seqs
    ]
    label_texts = [
        [parse_label(c, False, unk_token="…") for c in label]
        for label in labels
    ]
    return texts, label_texts


class ChujianMLMDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        is_training: bool,
        context_len: int = 8,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.is_training = is_training

        if is_training:
            """Assume that the texts are a list of strings"""
            self.texts = load_train_texts(data_path)
            self.examples = self.get_train_examples(
                self.texts, self.tokenizer, self.context_len
            )
        else:
            # For evaluation
            self.texts, self.label_texts = load_test_texts(data_path)
            self.examples = self.get_test_examples(
                self.texts, self.label_texts
            )

    def get_test_examples(
        self,
        texts: List[List[str]],
        label_texts: List[List[str]],
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
        chunk_size = self.context_len
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
            return self.tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                max_length=self.context_len + 2,  # +2 for [CLS] and [SEP]
                return_tensors="pt",
                padding="max_length",
            )

        input_encoding = tokenize(chunked_texts)
        input_ids = input_encoding["input_ids"]
        att_mask = input_encoding["attention_mask"]
        label_ids = tokenize(chunked_label_texts)["input_ids"]

        # Make padding ignored by the loss function (set to -100)
        mask_id = self.tokenizer.mask_token_id
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

    def get_train_examples(
        self,
        texts: List[List[str]],
    ) -> List[Dict[str, torch.Tensor]]:
        print("# of sequences before chunking:", len(texts))
        # Chunk sequences into blocks of `block_size` characters
        chunk_size = self.context_len
        texts = chunk_2d_list(texts, chunk_size)
        joined_texts = ["".join(x) for x in texts]
        print("# of sequences after chunking:", len(texts))

        print("====== train examples ======")
        for i in range(10):
            print(i, joined_texts[i])
        print("============================")

        batch_encoding: Dict[str, list] = self.tokenizer(
            joined_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.context_len + 2,  # +2 for [CLS] and [SEP]
        )
        input_ids: List[List[int]] = batch_encoding["input_ids"]
        examples = [
            {"input_ids": torch.tensor(e, dtype=torch.long)} for e in input_ids
        ]
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        return self.examples[i]


class ChujianMLMDatasetSmallVocab(Dataset):
    def __init__(
        self,
        vocab: List[str],
        data_path: Path,
        is_training: bool,
        context_len: int = 8,
        mask_prob: float = 0.15,
    ):
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + vocab
        self.vocab = vocab
        self.data_path = data_path
        self.is_training = is_training
        self.context_len = context_len
        self.mask_prob = mask_prob

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.pad_token_id = self.token_to_id["[PAD]"]
        self.unk_token_id = self.token_to_id["[UNK]"]
        self.cls_token_id = self.token_to_id["[CLS]"]
        self.sep_token_id = self.token_to_id["[SEP]"]
        self.mask_token_id = self.token_to_id["[MASK]"]

        dump_json(self.vocab, 'small_vocab.json')
        self.examples = self.get_examples()

    def get_examples(self) -> List[Dict[str, List]]:
        if self.is_training:
            texts = load_train_texts(self.data_path)
            examples = self.get_train_examples(texts)
        else:
            texts, label_texts = load_test_texts(self.data_path)
            examples = self.get_test_examples(texts, label_texts)
        return examples

    def get_train_examples(self, texts: List[List[str]]) -> List[Dict]:
        examples = []
        chunk_size = self.context_len
        texts = chunk_2d_list(texts, chunk_size)

        # Tokenize using custom vocab
        for text in texts:
            input_ids = [
                self.token_to_id.get(token, self.unk_token_id)
                for token in text
            ]
            examples.append(
                {
                    "input_ids": input_ids,
                }
            )
        return examples

    def add_special_tokens(self, input_ids: List[int], labels: List[int]):
        input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
        labels = [-100] + labels + [-100]
        pad_len = self.context_len + 2 - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids += [self.pad_token_id] * pad_len
        labels += [-100] * pad_len
        return input_ids, attention_mask, labels

    def get_test_examples(
        self,
        texts: List[List[str]],
        label_texts: List[List[str]],
    ) -> List[Dict]:
        chunk_size = self.context_len - 2  # -2 for [CLS] and [SEP]
        texts = chunk_2d_list(texts, chunk_size)
        label_texts = chunk_2d_list(label_texts, chunk_size)

        examples: List[Dict] = []
        for text, label_text in zip(texts, label_texts):
            assert len(text) == len(label_text)
            input_ids = [
                self.token_to_id.get(token, self.unk_token_id)
                for token in text
            ]
            labels = [
                self.token_to_id.get(token, self.unk_token_id)
                for token in label_text
            ]

            input_ids, attention_mask, labels = self.add_special_tokens(
                input_ids, labels)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        if self.is_training:
            # Random mask 15% of (non-padding) tokens
            input_ids = self.examples[i]["input_ids"]
            labels = input_ids[:]

            # Mask p% of tokens
            seq_len = len(input_ids)
            mask_cnt = max(1, int(seq_len * self.mask_prob))
            mask_indices = np.random.choice(
                np.arange(seq_len), mask_cnt, replace=False
            )
            for mask_index in mask_indices:
                input_ids[mask_index] = self.mask_token_id

            # Add special tokens: CLS, SEP and PAD
            input_ids, attention_mask, labels = self.add_special_tokens(
                input_ids, labels)

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print(labels.shape)
            # exit()

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        else:
            return self.examples[i]
