from argparse import ArgumentParser
from dataclasses import dataclass

import jieba
from bs4 import BeautifulSoup
from tdigest import TDigest


def split_blocks(sentence: str):
    """Split `sentence` into jieba-cuttable blocks."""
    sentence = jieba.strdecode(sentence)
    for blk in jieba.re_han_default.split(sentence):
        if blk and jieba.re_han_default.match(blk):
            yield blk


@dataclass
class CorpusStats:
    # Number of sentences.
    n: int
    # Number of chars.
    c: int
    # Number of unique chars.
    v: int
    # Max number of chars in a sentence.
    s: int
    # The 99th percentile of the number of chars in a sentence.
    s99: int
    # The 95th percentile of the number of chars in a sentence.
    s95: int
    # The cumulative percentiles of the number of chars in a sentence.
    _sq: TDigest
    # The cumulative unique chars in the corpus.
    _v: set

    @classmethod
    def zero(cls):
        return cls(n=0, c=0, v=0, s=0, s99=0, s95=0, _sq=TDigest(), _v=set())

    def update(self, sentence, flush=False):
        l_ = len(sentence)
        self._sq.update(l_)
        self.s = max(self.s, l_)
        self.n += 1
        self._v.update(sentence)
        self.c += l_
        if flush:
            self.finish()

    def finish(self):
        self.v = len(self._v)
        self.s99 = int(self._sq.percentile(99))
        self.s95 = int(self._sq.percentile(95))

    def write_header(self, outfile):
        outfile.write(f"#N {self.n}\n")
        outfile.write(f"#C {self.c}\n")
        outfile.write(f"#V {self.v}\n")
        outfile.write(f"#S {self.s}\n")
        outfile.write(f"#S99 {self.s99}\n")
        outfile.write(f"#S95 {self.s95}\n")


def process_html_data(html_infile, dataset_outfile):
    soup = BeautifulSoup(html_infile, "lxml")
    corpus_stats = CorpusStats.zero()
    for sent in soup.find_all("p"):
        for blk in split_blocks(sent.text):
            corpus_stats.update(blk)
    corpus_stats.write_header(dataset_outfile)
    for sent in soup.find_all("p"):
        for blk in split_blocks(sent.text):
            dataset_outfile.write(f"{blk}\n")


def process_html_data_main():
    parser = ArgumentParser(
        description="Process an html corpus into dataset file."
    )
    parser.add_argument("html_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    with (
        open(args.html_path, "rb") as infile,
        open(args.output_path, "w", encoding="utf-8") as outfile,
    ):
        process_html_data(infile, outfile)
