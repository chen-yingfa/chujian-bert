'''
This is loop through all text sequences, and count the number of times
each character appears. Those appearing at least k=10 times will be added
to the vocabulary of the tokenizer.

The tokenizer is dumped to `../tokenizer`.
'''
import json
from collections import defaultdict
from utils import parse_label
from typing import List
from transformers import AutoTokenizer


def dump_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def iter_seqs(seqs: List[List[str]]):
    for seq in seqs:
        for c in seq:
            yield c


def get_new_tokens(
    seqs: List[List[str]],
    orig_vocab: dict,
) -> List[str]:
    # Make new vocab
    word_cnt = defaultdict(int)
    for c in iter_seqs(seqs):
        word_cnt[c] += 1

    k = 10
    new_tokens = {"[COMB]"}
    for c in iter_seqs(seqs):
        if c not in orig_vocab and word_cnt[c] >= k:
            new_tokens.add(c)
    new_tokens = list(new_tokens)
    print(f"Num of new tokens: {len(new_tokens)}")
    return new_tokens


# seq_file = (
#     '/data/private/chenyingfa/chujian/sequences/seq_texts.json')
seq_file = (
    "E:/donny/code/school/research/chujian/data/sequences/sequences.json"
)
seqs = json.load(open(seq_file, "r", encoding="utf-8"))
seqs = [seq["text"] for seq in seqs]
seqs = [[parse_label(c, use_comb_token=False) for c in seq] for seq in seqs]
print("====== Example sequences")
for seq in seqs[:5]:
    print(seq)
print("======")

# Load tokenizer
model_name = "KoichiYasuoka/roberta-classical-chinese-base-char"
# model_name = "ethanyt/guwenbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get new tokens
new_tokens = get_new_tokens(seqs, tokenizer.get_vocab())
dump_json(new_tokens, "new_vocab.json")

# Update the vocab of tokenizer
tokenizer.add_tokens(new_tokens)
print("Saving tokenizer to ../tokenizer")
tokenizer.save_pretrained("../tokenizer")

# Test the tokenizer
text = "綊墿？箄{竹巫口}弩弓[COMB]𡄹{弓口二}？？[COMB][MASK][SEP][CLS]"
print(tokenizer.tokenize(text))
