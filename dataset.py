from typing import List, Dict

import torch
from torch.utils.data import Dataset


class ChujianMLMDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        block_size: int = 10,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size

        batch_encoding: dict = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
        )
        self.examples: List[int] = batch_encoding["input_ids"]
        self.examples = [
            {"input_ids": torch.tensor(e, dtype=torch.long)}
            for e in self.examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.examples[i]
