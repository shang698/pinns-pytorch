import json
from pathlib import Path


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    parent = config.pop("extends", None)
    if parent:
        config = {**load_config(path.parent / parent), **config}
    return config


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
