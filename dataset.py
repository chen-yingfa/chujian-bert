from typing import List, Dict

import torch
from torch.utils.data import Dataset


class ChujianMLMDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        is_training: bool,
        block_size: int = 10,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size

        if is_training:
            batch_encoding: dict = self.tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                max_length=block_size,
            )
            input_ids: List[List[int]] = batch_encoding["input_ids"]
            self.examples = [
                {"input_ids": torch.tensor(e, dtype=torch.long)}
                for e in input_ids
            ]
        else:
            encoding: Dict[str, torch.Tensor] = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=block_size,
            )
            num_examples = len(encoding['input_ids'])
            self.examples: List[Dict[str, torch.Tensor]] = [
                {
                    key: encoding[key][i] for key in encoding
                } for i in range(num_examples)
            ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.examples[i]
