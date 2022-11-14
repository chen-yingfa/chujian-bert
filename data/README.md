# Data

## Dataset

[TODO]

## Data Preprocessing

We need to do two things:

1. Contruct train and test data.
2. Contruct the tokenizer vocabulary, by looping through the data.

### Train and Test Data Construction

This is done by `preprocess.py`, it will reformat the data of `chujian.json` (raw crawled data) into an array of arrays of glyphs labels, which is dumped to `sequences.json`.

> All output files are dumped to the directory specified by `DST_DIR`.

Then the data is split into train and test by a ratio of 9:1. 15% of glyphs will be masked in the test data (replaced with \[MASK\]). The result is dumped to `train.jsonl` and `test.jsonl`. Each line in `train.jsonl` is simply an array of glyph labels. Each line in `test.jsonl` is a dictionary with "sequence" and "label", which is identical except for the masked glyphs.

Example of `train.jsonl`:

```jsonl
["於", "西", "北", "亓（其）", "下", "高", "㠯（以）", "勥（強）", "○（地）", "不", "足", "於", "東", "南", "亓（其）", "上"]
["繏（纂）", "約", "紫"]
```

Example of `test.jsonl`:

```jsonl
{"sequence": ["[MASK]", "九"], "label": ["卅", "九"]}
{"sequence": ["[MASK]", "○", "占", "之", "曰", "吉"], "label": ["○", "○", "占", "之", "曰", "吉"]}
```

### Vocabulary Construction

This is done by `make_tokenizer.py`, which will dump the tokenizer with the new vocabulary to `../tokenizer`.
