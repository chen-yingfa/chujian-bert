import json
from pathlib import Path
from typing import List
import random


def dump_json(data, path):
    json.dump(
        data, open(path, "w", encoding="utf8"), indent=4, ensure_ascii=False
    )


def dump_jsonl(data, path):
    with open(path, "w", encoding="utf8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def get_sequences(meta_file: Path) -> list:
    seqs = []
    meta_data = json.load(open(meta_file, "r", encoding="utf8"))
    for ex in meta_data:
        words = ex["sequences"]
        images = ex["image_paths"]
        images = [meta_file.parent / image for image in images]
        name_to_image = {path.name: path for path in images}
        this_seq = []
        for word_idx, word in enumerate(words):
            glyph = word["jian_word"]
            if glyph is None:
                glyph = "None"
            glyph = glyph.replace("/", ",")
            name = Path(word["jian_word_image"]).name
            if name not in name_to_image:
                continue
            image = name_to_image[name]
            this_seq.append(
                {
                    "index": word_idx,
                    "glyph": glyph,
                    "image": str(image),
                }
            )
        seqs.append(
            {
                "slip_image": ex["jian_image_url"],
                "sequence": this_seq,
            }
        )
    return seqs


def split(seqs: list):
    random.shuffle(seqs)
    split_idx = [int(len(seqs) * 0.9), int(len(seqs) * 0.95)]
    train = seqs[:split_idx[0]]
    dev = seqs[split_idx[0]:split_idx[1]]
    test = seqs[split_idx[1]:]
    return train, dev, test


def create_test_examples(
    seqs: List[List[str]],
    mask_rate=0.15,
    min_cnt: int = 1,
):
    """
    Turn:
        ['○（莊）', '公', '䎽（問）', '爲']
    into:
        {
            "sequence": ['○（莊）', '[MASK]', '䎽（問）', '爲'],
            "label": ['○（莊）', '公', '䎽（問）', '爲']
        }
    """
    examples = []
    num_masks = 0
    num_tokens = 0
    for seq in seqs:
        label = seq[:]  # The unmasked sequence is the label.
        n = len(seq)
        indices = list(range(n))
        mask_indices = random.sample(indices, max(min_cnt, round(n * mask_rate)))
        num_masks += len(mask_indices)
        num_tokens += n
        for idx in mask_indices:
            seq[idx] = "[MASK]"
        examples.append(
            {
                "sequence": seq,
                "label": label,
            }
        )
    actual_mask_rate = num_masks / num_tokens
    print(f"Actual mask rate: {actual_mask_rate}")
    return examples


if __name__ == "__main__":
    # SRC_DIR = Path("/data/private/chenyingfa/chujian/data")
    # DST_DIR = Path("/data/private/chenyingfa/chujian/sequences")
    SRC_DIR = Path("E:/donny/code/school/research/chujian/data/data")
    DST_DIR = Path("E:/donny/code/school/research/chujian/data/text")
    meta_file = SRC_DIR / "chujian.json"

    print(f"Getting sequences from {meta_file}")
    seqs = get_sequences(meta_file)
    num_images = sum(len(seq) for seq in seqs)

    print(f"Found {len(seqs)} sequences")
    print(f"Found {num_images} images")
    print("======")

    seqs_file = DST_DIR / "all_sequences.json"
    print(f"Dumping to {seqs_file}")
    DST_DIR.mkdir(exist_ok=True, parents=True)
    dump_json(seqs, seqs_file)

    seqs = [
        {
            "slip_image": seq["slip_image"],
            "text": [word["glyph"] for word in seq["sequence"]],
        }
        for seq in seqs
    ]
    min_len = 2
    seqs = [s for s in seqs if len(s["text"]) >= 2]
    print(
        f"Found {len(seqs)} sequences with at least {min_len} (labeled) words"
    )

    seqs_file = DST_DIR / "texts.json"
    print(f"Dumping to {seqs_file}")
    dump_json(seqs, seqs_file)

    # Split data into train and test
    random.seed(0)
    texts = [seq["text"] for seq in seqs]
    train_texts, dev_texts, test_texts = split(texts)
    dev_texts = create_test_examples(dev_texts)
    test_texts = create_test_examples(test_texts)
    print(test_texts[:10])
    print(f"Dumping train and test data to {DST_DIR}")
    dump_jsonl(train_texts, DST_DIR / "train.jsonl")
    dump_jsonl(dev_texts, DST_DIR / "dev.jsonl")
    dump_jsonl(test_texts, DST_DIR / "test.jsonl")
