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
            return key in self.tok2idx
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

    def _normalize_key(self, key: VocabKey) -> int:
        if isinstance(key, int):
            return key
        if isinstance(key, str):
            return self.tok2idx[key]
        raise TypeError("key")

    def freq(self, key: VocabKey):
        return self.idx2freq[self._normalize_key(key)]

    def uniform_freq_(self, value: int = 1):
        for i in range(len(self.idx2freq)):
            self.update_freq(i, value)

    def tag(self, key: VocabKey):
        return self.idx2tag[self._normalize_key(key)]

    def is_main(self, key: VocabKey):
        """Check if `key` belongs to the *main* vocab."""
        return self.idx2main[self._normalize_key(key)]

    def update_freq(self, key: VocabKey, freq: int):
        idx = self._normalize_key(key)
        old_freq = self.idx2freq[idx]
        self.idx2freq[idx] = freq
        self.total_freq = self.total_freq - old_freq + freq

    def __setitem__(self, key, freq):
        raise RuntimeError("use self.update_freq(key, freq) instead")

    def append(
        self,
        word: str,
        freq: int,
        tag: str | None = None,
        is_main: bool = True,
    ):
        """Append word to self. Raise `ValueError` on duplicate word."""
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

    def upsert(
        self,
        word: str,
        freq: int,
        tag: str | None = None,
        is_main: bool = True,
        update: bool = False,
    ) -> bool:
        """
        Update/append a word. If there is duplicate word with exactly the
        same (frequency, tag and is_main marker), skip silently. Otherwise,
        update the word if `update` is True. Return whether an upsert has been
        performed.
        """
        try:
            self.append(word, freq, tag, is_main)
            return True
        except ValueError:
            orig_freq = self.freq(word)
            orig_tag = self.tag(word)
            orig_main = self.is_main(word)
            if freq == orig_freq and tag == orig_tag and is_main == orig_main:
                return False
            if not update:
                return False
            idx = self.tok2idx[word]
            self.update_freq(idx, freq)
            self.idx2tag = tag
            self.idx2main = is_main
            return True

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

    def as_main_mask_array(self):
        """True means the word belongs to the "main" vocab."""
        import numpy as np

        mask = np.ones(len(self), dtype=np.bool_)
        for idx, is_main in enumerate(self.idx2main):
            mask[idx] = is_main
        return mask

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
