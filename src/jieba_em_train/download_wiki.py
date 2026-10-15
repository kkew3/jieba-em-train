import shutil
from argparse import ArgumentParser
from pathlib import Path
from tempfile import NamedTemporaryFile

import datasets
from tqdm import tqdm

from .preprocess_data import split_blocks, CorpusStats


def main():
    parser = ArgumentParser(
        description="Download preprocessed fjcanyue/wikipedia-zh-cn from huggingface."
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
    with NamedTemporaryFile(
        mode="w+", dir=tofile.parent, prefix="tmp", encoding="utf-8"
    ) as tmpout:
        for text in tqdm(ds["text"], total=args.total, disable=args.total == 0):
            for blk in split_blocks(text):
                corpus_stats.update(blk)
                tmpout.write(f"{blk}\n")
        tmpout.seek(0)
        with open(tofile, "w", encoding="utf-8") as outfile:
            corpus_stats.write_header(outfile)
            shutil.copyfileobj(tmpout, outfile)
