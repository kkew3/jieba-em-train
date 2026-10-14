import copy
import math
from pathlib import Path

import numpy as np
import rjieba
from bs4 import BeautifulSoup

from .vocab import Vocab


def count_pred_stats(tp, fp, fn, jb, segmented_sentences):
    for seg_sent_true in segmented_sentences:
        sent = "".join(seg_sent_true)
        seg_sent = jb.cut(sent, hmm=True)
        seg_idx_true = np.cumsum([len(w) for w in seg_sent_true])
        seg_idx = np.cumsum([len(w) for w in seg_sent])
        i = 0  # cursor for true seg
        m = seg_idx_true.shape[0]
        j = 0  # cursor for pred seg
        n = seg_idx.shape[0]
        while i < m and j < n:
            if seg_idx_true[i] > seg_idx[j]:
                fp += 1
                j += 1
            elif seg_idx_true[i] < seg_idx[j]:
                fn += 1
                i += 1
            else:
                tp += 1
                i += 1
                j += 1
        while i < m:
            fn += 1
            i += 1
        while j < n:
            fp += 1
            j += 1
    return tp, fp, fn


def to_freq(log_theta, c=None):
    if c is None:
        c = math.log(0xFFFFFFFF >> 2)
    elif isinstance(c, int):
        c = math.log(c)
    freqs = [int(x) for x in np.round(np.exp(log_theta + c), 0).tolist()]
    return freqs


def export_vocab(v: Vocab, log_theta, c=None) -> Vocab:
    freqs = to_freq(log_theta, c=c)
    v = copy.deepcopy(v)
    for word in v:
        idx = v[word]
        v.idx2freq[idx] = freqs[idx]
    return v


def configure_jieba(v: Vocab, log_theta=None, freq_c=None):
    jb = rjieba.Jieba()
    jb.clear()
    if log_theta is None:
        for word in v:
            jb.add_word(word, freq=v.freq(word), tag=v.tag(word))
            return jb

    freqs = to_freq(log_theta, freq_c)
    for i in range(len(v)):
        word = v[i]
        f = freqs[i]
        jb.add_word(word, freq=f, tag=v.tag(word))
    return jb


def compute_f_beta(tp, fp, fn, beta):
    beta2 = beta**2
    fbeta = (1 + beta2) * tp / ((1 + beta2) * tp + fp + beta2 * fn)
    return fbeta


class Fbeta:
    def __init__(self, tp, fp, fn):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def get(self, beta=1):
        return compute_f_beta(self.tp, self.fp, self.fn, beta)


def iter_segmented_sentences(path: Path):
    if path.suffix == ".xml":
        # MSRA bakeoff xml.
        with open(path, "rb") as infile:
            soup = BeautifulSoup(infile, "lxml-xml")
        for sent in soup.find_all("SENTENCE"):
            segmented_sent = [token.text for token in sent.find_all("w")]
            yield segmented_sent
    else:
        # Parsed plaintext.
        with open(path, encoding="utf-8") as infile:
            for line in infile:
                yield line.rstrip("\n").split()


def run_eval(segmented_sents_path: str, v: Vocab, log_theta, freq_c=None):
    jb = configure_jieba(v, log_theta, freq_c=freq_c)
    tp, fp, fn = count_pred_stats(
        0, 0, 0, jb, iter_segmented_sentences(Path(segmented_sents_path))
    )
    return Fbeta(tp, fp, fn)


def pprint_cut(segmented_sent: list[str], sep="/") -> str:
    return sep.join(segmented_sent).replace(f"{sep}\n{sep}", "\n")
