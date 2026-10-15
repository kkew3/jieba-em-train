import gzip
import itertools
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import datasets
import jieba
from bs4 import BeautifulSoup
from tdigest import TDigest
from tqdm import tqdm


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
    # The cumulative unique chars in the corpus.
    _v: set | None

    @classmethod
    def zero(cls):
        return cls(n=0, c=0, v=0, s=0, _v=set())

    def update(self, sentence, flush=False):
        l_ = len(sentence)
        self.s = max(self.s, l_)
        self.n += 1
        self._v.update(sentence)
        self.c += l_
        if flush:
            self.finish()

    def finish(self):
        self.v = len(self._v)
        self._v = None

    def write_header(self, outfile):
        outfile.write(f"#N {self.n}\n")
        outfile.write(f"#C {self.c}\n")
        outfile.write(f"#V {self.v}\n")
        outfile.write(f"#S {self.s}\n")

    @classmethod
    def read_header(cls, infile):
        header = {}
        for line in itertools.islice(infile, 4):
            d, _, value = line.partition(" ")
            value = int(value)
            header[d] = value
        return cls(
            n=header["#N"],
            c=header["#C"],
            v=header["#V"],
            s=header["#S"],
            _v=None,
        )


def iter_sentences(corpos_infile, **tqdm_kwargs):
    """tqdm kwargs other than `total` can be supplied via `tqdm_kwargs`."""
    stats = CorpusStats.read_header(corpos_infile)
    for line in tqdm(corpos_infile, total=stats.n, **tqdm_kwargs):
        yield line.rstrip("\n")


def process_wiki_data_main():
    parser = ArgumentParser(
        description=(
            "Download preprocessed fjcanyue/wikipedia-zh-cn from huggingface."
        )
    )
    parser.add_argument("tofile")
    parser.add_argument(
        "-t",
        "--total",
        type=int,
        default=3186425,
        help=(
            "An estimate of the dataset length, "
            "used to display the progress bar. "
            "Set to 0 to disable progress bar."
        ),
    )
    args = parser.parse_args()

    ds = datasets.load_dataset(
        "fjcanyue/wikipedia-zh-cn", split="train", streaming=True
    )
    tofile = Path(ds).resolve()
    corpus_stats = CorpusStats.zero()
    if tofile.suffix == ".gz":
        _open = gzip.open
        _mode = "wt"
    else:
        _open = open
        _mode = "w"
    with NamedTemporaryFile(
        mode="w+", dir=tofile.parent, prefix="tmp", encoding="utf-8"
    ) as tmpout:
        for text in tqdm(ds["text"], total=args.total, disable=args.total == 0):
            for blk in split_blocks(text):
                corpus_stats.update(blk)
                tmpout.write(f"{blk}\n")
        corpus_stats.finish()
        tmpout.seek(0)
        with _open(tofile, _mode, encoding="utf-8") as outfile:
            corpus_stats.write_header(outfile)
            shutil.copyfileobj(tmpout, outfile)


def process_html_data(html_infile, dataset_outfile):
    soup = BeautifulSoup(html_infile, "lxml")
    corpus_stats = CorpusStats.zero()
    for sent in soup.find_all("p"):
        for blk in split_blocks(sent.text):
            corpus_stats.update(blk)
    corpus_stats.finish()
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
