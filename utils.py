import json


def dump_json(obj, path, indent=4):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
