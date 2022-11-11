
import json
from collections import defaultdict

from transformers import AutoTokenizer


def dump_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def parse_label(label: str) -> str:
    UNK_TOKEN = '?'
    # Normalize the glyph label
    RM_STRS = [
        '=', 'None'
    ]
    for c in RM_STRS:
        label = label.replace(c, '')

    # Replace brackets
    for c in ['（', '〈', '[']:
        label = label.replace(c, '(')
    for c in ['）', '〉', ']']:
        label = label.replace(c, ')')

    if label == '':
        return UNK_TOKEN

    if label[-1] == ')':
        for i in range(len(label) - 2, -1, -1):
            if label[i] == '(':
                # "（*）"
                if label[i] == '(':
                    if label[i+1:-1] == '○':
                        label = label[:i]
                    else:
                        label = label[i+1:-1]
                else:
                    # "*}（*）"
                    if label[i-1] == '}':
                        label = label[i+1:-1]
                    # "A（*）" -> "A"
                    else:
                        label = label[0]
                break
        else:
            label = label[:-1]
    # "A→B"
    if '→' in label:
        label = label.split('→')[1]
    if label == '𬨭':
        label = '將'
    if label == '𫵖':
        label = '尸示'

    # if len(label) != 1:
    #     return UNK_TOKEN

    DISCARD_CHARS = [
        '?'
        '□', '■',
        '○', '●',
        '△', '▲',
        '☆', '★',
        '◇', '◆',
        '□'
    ]
    if any(c in label for c in DISCARD_CHARS):
        return UNK_TOKEN
    return label


model_name = "KoichiYasuoka/roberta-classical-chinese-base-char"
# model_name = "ethanyt/guwenbert-base"
cache_dir = "E:/.cache/huggingface"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)

seq_file = (
    'E:/donny/code/school/research/chujian/data/sequences/seq_texts.json')
seqs = json.load(open(seq_file, 'r', encoding='utf-8'))
seqs = [seq['text'] for seq in seqs]
seqs = [[parse_label(c) for c in seq] for seq in seqs]


def get_rare_words(seqs: list, k: int) -> dict:
    '''Get rare words (occurring less than k times) in the sequences.'''
    word_cnt = defaultdict(int)
    for seq in seqs:
        for c in seq:
            word_cnt[c] += 1
    rare_words = {w: cnt for w, cnt in word_cnt.items() if cnt < k}
    return rare_words


rare_words = get_rare_words(seqs, 10)
print('# rare words:', len(rare_words))

vocab: dict = tokenizer.vocab
orig_vocab_size = len(vocab)
new_vocab = {}
print('Original vocab size:', orig_vocab_size)
print('Num new words:', len(new_vocab))
print('Final vocab size', len(vocab))
dump_json(new_vocab, 'new_vocab.json')
dump_json(new_words_cnt, 'new_words_cnt.json')

discard_cnt = 0
print('======')
for k in range(10):
    cnt = 0
    for word in new_vocab:
        if new_words_cnt[word] == k:
            cnt += 1
    print(f'Num words with count {k}: {cnt}')
    discard_cnt += cnt
print('======')
print(discard_cnt)
print('Final number of new words:', len(new_vocab) - discard_cnt)
print('Final vocab size:', len(vocab) - discard_cnt)

# %%
