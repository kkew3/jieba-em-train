import math
from typing import TypeAlias

VocabKey: TypeAlias = str | int  # noqa: UP040


class Vocab:
    def __init__(self):
        self.tok2idx = {}
        self.idx2tok = []
        self.idx2freq = []
        self.idx2tag = []
        self.idx2main = []
        self.total_freq = 0

    def __len__(self):
        return len(self.idx2tok)

    def __eq__(self, value):
        if isinstance(value, Vocab):
            return (
                self.tok2idx == value.tok2idx
                and self.idx2tok == value.idx2tok
                and self.idx2freq == value.idx2freq
                and self.idx2tag == value.idx2tag
                and self.idx2main == value.idx2main
                and self.total_freq == value.total_freq
            )
        return NotImplemented

    def __contains__(self, key: VocabKey):
        if isinstance(key, int):
            return 0 <= key < len(self)
        if isinstance(key, str):
            return key in self.idx2tok
        raise TypeError("key")

    def __getitem__(self, key: VocabKey):
        if isinstance(key, int):
            return self.idx2tok[key]
        if isinstance(key, str):
            return self.tok2idx[key]
        raise TypeError("key")

    def __iter__(self):
        yield from self.idx2tok

    def items(self):
        return self.tok2idx.items()

    def copy(self):
        v = Vocab()
        v.tok2idx = self.tok2idx.copy()
        v.idx2tok = self.idx2tok.copy()
        v.idx2freq = self.idx2freq.copy()
        v.idx2tag = self.idx2tag.copy()
        v.total_freq = self.total_freq
        return v

    def freq(self, key: VocabKey):
        if isinstance(key, int):
            return self.idx2freq[key]
        if isinstance(key, str):
            return self.idx2freq[self.tok2idx[key]]
        raise TypeError("key")

    def uniform_freq_(self, value: int = 1):
        for i in range(len(self.idx2freq)):
            self.update_freq(i, value)

    def tag(self, key: VocabKey):
        if isinstance(key, int):
            return self.idx2tag[key]
        if isinstance(key, str):
            return self.idx2tag[self.tok2idx[key]]
        raise TypeError("key")

    def is_main(self, key: VocabKey):
        """Check if `key` belongs to the *main* vocab."""
        if isinstance(key, int):
            return self.idx2main[key]
        if isinstance(key, str):
            return self.idx2main[self.tok2idx[key]]
        raise TypeError("key")

    def update_freq(self, key: VocabKey, freq: int):
        if isinstance(key, int):
            old_freq = self.idx2freq[key]
            self.idx2freq[key] = freq
            self.total_freq = self.total_freq - old_freq + freq
        elif isinstance(key, str):
            old_freq = self.idx2freq[self.tok2idx[key]]
            self.idx2freq[self.tok2idx[key]] = freq
            self.total_freq = self.total_freq - old_freq + freq
        else:
            raise TypeError("key")

    def __setitem__(self, key, freq):
        raise RuntimeError("use self.update_freq(key, freq) instead")

    def append(
        self,
        word: str,
        freq: int,
        tag: str | None = None,
        is_main: bool = True,
    ):
        if word in self:
            raise ValueError("word already exists")
        idx = len(self)
        self.idx2tok.append(word)
        self.idx2freq.append(freq)
        self.idx2tag.append(tag)
        self.idx2main.append(is_main)
        self.tok2idx[word] = idx
        self.total_freq += freq
        return idx

    def log_prob(self, key: VocabKey, default: float = float("-inf")):
        if key not in self:
            return default
        if self.freq(key) == 0:
            return float("-inf")
        return math.log(self.freq(key)) - math.log(self.total_freq)

    def as_logprob_array(self):
        import numpy as np

        logprob = np.full(len(self), -math.log(self.total_freq))
        for idx, freq in enumerate(self.idx2freq):
            if freq == 0:
                logprob[idx] = float("-inf")
                continue
            logprob[idx] += math.log(freq)
        return logprob

    def dump(self, path: str, jieba_compatible: bool = True):
        with open(path, "w", encoding="utf-8") as outfile:
            self.dumpfile(outfile, jieba_compatible=jieba_compatible)

    def dumpfile(self, outfile, jieba_compatible: bool = True):
        for word, freq, tag, main in zip(
            self.idx2tok, self.idx2freq, self.idx2tag, self.idx2main
        ):
            if jieba_compatible:
                main_marker = ""
            elif main:
                main_marker = "+"
            else:
                main_marker = "-"
            if tag is None:
                outfile.write(f"{main_marker}{word} {freq}\n")
            else:
                outfile.write(f"{main_marker}{word} {freq} {tag}\n")

    @classmethod
    def load(cls, path: str, jieba_compatible: bool = True):
        with open(path, encoding="utf-8") as infile:
            return cls.loadfile(infile, jieba_compatible=jieba_compatible)

    @classmethod
    def loadfile(cls, infile, jieba_compatible: bool = True):
        v = cls()
        for line in infile:
            word, freq, *tag = line.rstrip("\n").split()
            freq = int(freq)
            if jieba_compatible:
                is_main = True
            else:
                main_marker, word = word[0], word[1:]
                is_main = main_marker == "+"
            if tag:
                v.append(word, freq, tag[0], is_main=is_main)
            else:
                v.append(word, freq, is_main=is_main)
        return v
