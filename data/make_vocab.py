import json
from collections import defaultdict
from utils import parse_label
from typing import List
from transformers import AutoTokenizer


def dump_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


seq_file = (
    '/data/private/chenyingfa/chujian/sequences/seq_texts.json')
seqs = json.load(open(seq_file, 'r', encoding='utf-8'))
seqs = [seq['text'] for seq in seqs]
seqs = [[parse_label(c, use_comb_token=False) for c in seq] for seq in seqs]
print(seqs[:10])

# Load tokenizer
model_name = "KoichiYasuoka/roberta-classical-chinese-base-char"
# model_name = "ethanyt/guwenbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def iter_seqs(seqs: List[List[str]]):
    for seq in seqs:
        for c in seq:
            yield c


word_cnt = defaultdict(int)
for c in iter_seqs(seqs):
    word_cnt[c] += 1

k = 10
orig_vocab = tokenizer.vocab
new_tokens = {'[COMB]'}
for c in iter_seqs(seqs):
    if c not in orig_vocab and word_cnt[c] >= k:
        new_tokens.add(c)
new_tokens = list(new_tokens)
print(f"Num of new tokens: {len(new_tokens)}")
dump_json(new_tokens, 'new_vocab.json')

# Update the vocab of tokenizer
tokenizer.add_tokens(new_tokens)
print('Saving tokenizer to ../tokenizer')
tokenizer.save_pretrained('../tokenizer')

# Test the tokenizer
text = "綊墿？箄{竹巫口}弩弓[COMB]𡄹{弓口二}？？[COMB]"
print(tokenizer.tokenize(text))
