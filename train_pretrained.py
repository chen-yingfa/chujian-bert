import json
from pathlib import Path

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from trainer import Trainer
from data.dataset import ChujianMLMDataset


def dump_json(data: dict, path: Path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    # TEXTS_PATH = Path(
    # '/data/private/chenyingfa/chujian/sequences/seq_texts.json')
    # TEXTS_PATH = Path('../data/sequences/seq_texts.json')
    TRAIN_PATH = Path('../data/sequences/train.jsonl')
    TEST_PATH = Path('../data/sequences/test.jsonl')

    MODEL_NAME = "KoichiYasuoka/roberta-classical-chinese-large-char"
    TOKENIZER_PATH = 'tokenizer'
    print('Loading tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    # Hyperparameters
    num_epochs = 8
    lr = 5e-5
    batch_size = 128
    log_interval = 10
    mode = 'train_test'

    # Output
    output_dir = Path(
        'result',
        MODEL_NAME,
        f'lr{lr}-bs{batch_size}-ep{num_epochs}',
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model,
        output_dir,
        train_data_collator=train_data_collator,
        num_epochs=num_epochs,
        batch_size=batch_size,
        log_interval=log_interval
    )

    if 'train' in mode:
        train_data = ChujianMLMDataset(TRAIN_PATH, tokenizer, is_training=True)
        trainer.train(train_data)
    if 'test' in mode:
        test_output_dir = output_dir / 'test'
        test_output_dir.mkdir(parents=True, exist_ok=True)
        test_data = ChujianMLMDataset(TEST_PATH, tokenizer, is_training=False)
        result = trainer.evaluate(test_data, test_output_dir)
        dump_json(result['preds'], test_output_dir / 'preds.json')
        del result['preds']
        dump_json(result, test_output_dir / 'result.json')


if __name__ == '__main__':
    main()
