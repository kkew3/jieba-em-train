import math
from typing import TypeAlias

VocabKey: TypeAlias = str | int  # noqa: UP040


class Vocab:
    def __init__(self):
        self.tok2idx = {}
        self.idx2tok = []
        self.idx2freq = []
        self.idx2tag = []
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

    def append(self, word: str, freq: int, tag: str | None = None):
        if word in self:
            raise ValueError("word already exists")
        idx = len(self)
        self.idx2tok.append(word)
        self.idx2freq.append(freq)
        self.idx2tag.append(tag)
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
            logprob[idx] += math.log(freq)
        return logprob

    def dump(self, path: str):
        with open(path, "w", encoding="utf-8") as outfile:
            self.dumpfile(outfile)

    def dumpfile(self, outfile):
        for word, freq, tag in zip(self.idx2tok, self.idx2freq, self.idx2tag):
            if tag is None:
                outfile.write(f"{word} {freq}\n")
            else:
                outfile.write(f"{word} {freq} {tag}\n")

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as infile:
            return cls.loadfile(infile)

    @classmethod
    def loadfile(cls, infile):
        v = cls()
        for line in infile:
            word, freq, *tag = line.rstrip("\n").split()
            freq = int(freq)
            if tag:
                v.append(word, freq, tag[0])
            else:
                v.append(word, freq)
        return v
