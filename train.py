import random
from typing import List, Tuple
from pathlib import Path

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from trainer import Trainer
from dataset import ChujianMLMDataset
from utils import load_json
from data.utils import parse_label


def get_dataset(
    tokenizer: AutoTokenizer,
    texts: List[str]
) -> Tuple[ChujianMLMDataset, ChujianMLMDataset, ChujianMLMDataset]:
    random.seed(0)
    split_idx = [int(len(texts) * 0.8), int(len(texts) * 0.9)]
    print(f'Splitting data into {split_idx}...')
    print(f'Train size: {split_idx[0]}')
    print(f'Dev size: {split_idx[1] - split_idx[0]}')
    print(f'Test size: {len(texts) - split_idx[1]}')
    random.shuffle(texts)
    train_texts = texts[:int(split_idx[0])]
    dev_texts = texts[int(split_idx[0]):int(split_idx[1])]
    test_texts = texts[int(split_idx[1]):]
    print('Building dataset...')
    train_data = ChujianMLMDataset(train_texts, tokenizer)
    dev_data = ChujianMLMDataset(dev_texts, tokenizer)
    test_data = ChujianMLMDataset(test_texts, tokenizer)
    return train_data, dev_data, test_data


def load_texts(min_len: int = 2) -> list:
    TEXTS_PATH = '../data/sequences/seq_texts.json'
    texts = load_json(TEXTS_PATH)
    texts = [seq['text'] for seq in texts]
    print(f'Loaded {len(texts)} sequences.')
    texts = [t for t in texts if len(t) > 0]  # Many sequences are empty
    print(f'# non-empty sequences: {len(texts)}')
    texts = [t for t in texts if len(t) >= min_len]
    print(f'Minimum length: {min_len}')
    print(f'# sequences with enough length: {len(texts)}')
    texts = [
        ''.join(
            [parse_label(c, True, comb_token='…', unk_token='…') for c in seq]
        ) for seq in texts
    ]
    return texts


def train(model):
    pass


def main():
    texts = load_texts()
    print('====== examples ======')
    for i in range(12):
        if len(texts[i]) > 0:
            print(i, texts[i])

    MODEL_NAME = "KoichiYasuoka/roberta-classical-chinese-base-char"
    TOKENIZER_PATH = 'tokenization/tokenizer'
    print('Loading tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    # Hyperparameters
    num_epochs = 2
    lr = 2e-5
    batch_size = 64
    log_interval = 10

    # Output
    output_dir = Path(
        'result/roberta-classical-chinese-base-char',
        f'lr{lr}-bs{batch_size}-ep{num_epochs}',
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model,
        output_dir,
        train_collate_fn=train_collate_fn,
        num_epochs=num_epochs,
        batch_size=batch_size,
        log_interval=log_interval
    )

    texts = load_texts()
    train_data, dev_data, test_data = get_dataset(tokenizer, texts)

    trainer.train(train_data, dev_data)

    # Test


if __name__ == '__main__':
    main()
