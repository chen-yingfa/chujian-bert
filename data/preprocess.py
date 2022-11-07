import json
from pathlib import Path


def dump_json(data, path):
    json.dump(
        data, open(path, "w", encoding="utf8"), indent=4, ensure_ascii=False
    )


def get_sequences(meta_file: Path) -> list:
    seqs = []
    meta_data = json.load(open(meta_file, 'r', encoding='utf8'))
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
        seqs.append({
            "slip_image": ex["jian_image_url"],
            "sequence": this_seq,
        })
    return seqs


SRC_DIR = Path("/data/private/chenyingfa/chujian/data")
DST_DIR = Path("/data/private/chenyingfa/chujian/sequences")
meta_file = SRC_DIR / "chujian.json"

print(f'Getting sequences from {meta_file}')
seqs = get_sequences(meta_file)
num_images = sum(len(seq) for seq in seqs)

print(f"Found {len(seqs)} sequences")
print(f"Found {num_images} images")
print("======")

seqs_file = DST_DIR / 'sequences.json'
print(f"Dumping to {seqs_file}")
DST_DIR.mkdir(exist_ok=True, parents=True)
dump_json(seqs, seqs_file)

seq_texts = []
for seq in seqs:
    seq_texts.append({
        "slip_image": seq["slip_image"],
        "text": [word['glyph'] for word in seq["sequence"]]
    })
seq_texts_file = DST_DIR / 'seq_texts.json'
print(f'Dumping seq texts to {seq_texts_file}')
dump_json(seq_texts, seq_texts_file)
