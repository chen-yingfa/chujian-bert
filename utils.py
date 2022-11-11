import json


def load_json(path):
    return json.load(open(path, 'r', encoding='utf-8'))


def dump_json(data, path):
    json.dump(
        data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
