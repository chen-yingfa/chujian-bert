from typing import List, Dict

import torch
from torch.utils.data import Dataset


class ChujianMLMDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.inputs: Dict[str, torch.Tensor] = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        return {
            key: self.inputs[key][idx]
            for key in ['input_ids', 'attention_mask']
        }
