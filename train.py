from typing import List
from pathlib import Path
import random

from transformers import BertForMaskedLM, BertConfig
import torch
import numpy as np

from trainer import Trainer
from data.dataset import ChujianMLMDatasetSmallVocab
from utils import dump_json, load_json


def load_vocab() -> List[str]:
    glyphs_to_count_file = "../data/glyphs_k-0/glyph_to_count_sorted.json"
    vocab = load_json(glyphs_to_count_file)
    return list(vocab.keys())


def main():
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # TEXTS_PATH = Path(
    # '/data/private/chenyingfa/chujian/sequences/seq_texts.json')
    # TEXTS_PATH = Path('../data/sequences/seq_texts.json')
    TRAIN_PATH = Path("../data/sequences/train.jsonl")
    TEST_PATH = Path("../data/sequences/test.jsonl")

    print("(Randomly) Initializing model...")
    config = BertConfig.from_pretrained("configs/roberta-base-config.json")
    model = BertForMaskedLM(config)
    # print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {num_params}")

    # Hyperparameters
    num_epochs = 8
    lr = 5e-5
    batch_size = 128
    log_interval = 10
    mode = "train_test"

    # Output
    output_dir = Path(
        "result",
        # "temp",
        "bert_rand_init",
        f"lr{lr}-bs{batch_size}-ep{num_epochs}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    trainer = Trainer(
        model,
        output_dir,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        log_interval=log_interval,
    )

    vocab = load_vocab()
    if "train" in mode:
        train_data = ChujianMLMDatasetSmallVocab(
            vocab, TRAIN_PATH, is_training=True
        )
        test_data = ChujianMLMDatasetSmallVocab(
            vocab, TEST_PATH, is_training=False
        )
        trainer.train(train_data, test_data)
    if "test" in mode:
        test_output_dir = output_dir / "test"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        test_data = ChujianMLMDatasetSmallVocab(
            vocab, TEST_PATH, is_training=False
        )
        result = trainer.evaluate(test_data, test_output_dir)
        dump_json(result["preds"], test_output_dir / "preds.json")
        del result["preds"]
        dump_json(result, test_output_dir / "result.json")


if __name__ == "__main__":
    main()
