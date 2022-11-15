# Chujian BERT

Use BERT to perform Masked Language Model (MLM) on Chujian text.

We also try to use BERTs pretrained on classical Chinese texts, and finetuned on our Chujian data.

Pretrained models (on <https://www.huggingface.co>):

- KoichiYasuoka/roberta-classical-chinese-base-char

## Hyperparameters

MLM task:

- Context length: 8
- Training Mask Probability: 15%.

Model:

- Architecture: BERT base
- Training batch size: 512
- Learning rate: 1e-4

> We did not try many different hyperparameters, so there performance may have been better.

## Result

| Model | top-1 acc | top-3 acc | top-5 acc | top-10 acc |
| --- | --- | --- | --- | --- |
| BERT randomly initialized | 7.88 | 13.64 | 15.89 | 21.28 |
| BERT-Large (pretrained) | 12.61 | 18.13 | 21.39 | 26.27 |
