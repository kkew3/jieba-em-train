import contextlib
from pathlib import Path

import yaml

from .vocab import Vocab


def load_jieba_dict(jieba_dict_path: str):
    v = Vocab()
    with open(jieba_dict_path, encoding="utf-8") as infile:
        for line in infile:
            word, freq, *tag = line.rstrip("\n").split()
            freq = int(freq)
            if freq > 0 and tag:
                v.add(word, freq, tag[0])
            elif freq > 0:
                v.add(word, freq)
    return v


def parse_rime_dict_header(header: str):
    obj = yaml.safe_load(header)
    if "import_tables" not in obj:
        obj["import_tables"] = []
    if "columns" not in obj:
        obj["columns"] = ["text", "code", "weight"]
    return obj


def load_rime_dict_into_vocab(v: Vocab, rime_dict_path):
    header_lines = []
    stage = "before_header"
    columns = None
    with open(rime_dict_path, encoding="utf-8") as infile:
        for line in infile:
            line_trim = line.rstrip()
            if not line_trim or line_trim.startswith("#"):
                continue
            # State transitions.
            if stage == "before_header":
                if line_trim == "---":
                    stage = "header"
                    continue
            elif stage == "header" and line_trim == "...":
                # Parse header and load sub-tables.
                header_obj = parse_rime_dict_header("".join(header_lines))
                columns = header_obj["columns"]
                for name in header_obj["import_tables"]:
                    table_path = Path(rime_dict_path).parent / (
                        name + ".dict.yaml"
                    )
                    load_rime_dict_into_vocab(v, table_path)
                stage = "payload"
                continue
            if stage == "before_header":
                continue
            if stage == "header":
                header_lines.append(line)
                continue
            # Else, stage is "payload".
            assert columns is not None
            word = None
            freq = None
            for token_name, token in zip(columns, line_trim.split("\t")):
                if token_name == "text":
                    word = token
                elif token_name == "weight":
                    freq = int(token)
            if word is not None and freq is not None and freq > 0:
                with contextlib.suppress(ValueError):
                    v.add(word, freq)
