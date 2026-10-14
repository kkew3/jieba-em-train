import math


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

    def __contains__(self, key):
        if isinstance(key, int):
            return 0 <= key < len(self)
        if isinstance(key, str):
            return key in self.idx2tok
        raise TypeError("key")

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.idx2tok[key]
        if isinstance(key, str):
            return self.tok2idx[key]
        raise TypeError("key")

    def __iter__(self):
        yield from self.idx2tok

    def items(self):
        return self.tok2idx.items()

    def freq(self, key):
        if isinstance(key, int):
            return self.idx2freq[key]
        if isinstance(key, str):
            return self.idx2freq[self.tok2idx[key]]
        raise TypeError("key")

    def tag(self, key):
        if isinstance(key, int):
            return self.idx2tag[key]
        if isinstance(key, str):
            return self.idx2tag[self.tok2idx[key]]
        raise TypeError("key")

    def __setitem__(self, key, freq):
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

    def add(self, word, freq, tag=None, update=False):
        if word in self.tok2idx:
            idx = self.tok2idx[word]
            if (
                self.idx2freq[idx] != freq or self.idx2tag[idx] != tag
            ) and not update:
                raise ValueError(
                    "word already exists with a different freq/tag"
                )
            self[word] = freq
            self.idx2tag[self.tok2idx[word]] = tag
            return None
        idx = len(self)
        self.idx2tok.append(word)
        self.idx2freq.append(freq)
        self.idx2tag.append(tag)
        self.tok2idx[word] = idx
        self.total_freq += freq
        return idx

    def log_prob(self, word, default="-inf"):
        if word not in self:
            return float(default)
        if self.freq(word) == 0:
            return float("-inf")
        return math.log(self.freq(word)) - math.log(self.total_freq)

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
    def load(cls, path):
        with open(path, encoding="utf-8") as infile:
            return cls.loadfile(infile)

    @classmethod
    def loadfile(cls, infile):
        v = cls()
        for line in infile:
            word, freq, *tag = line.rstrip("\n").split()
            freq = int(freq)
            if tag:
                v.add(word, freq, tag[0])
            else:
                v.add(word, freq)
        return v
