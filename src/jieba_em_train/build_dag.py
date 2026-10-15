from collections import defaultdict
from collections.abc import Iterable

import ahocorasick
import numpy as np

from .vocab import Vocab


def build_dag(v: Vocab, sentences: Iterable[str], name: str):
    """
    Create two files:

    - {name}.dag: the DAG file, alternating between backward and forward DAGs.
    - {name}.dag.offset: the starting byte offset of each sentence.

    Return the file paths created.
    """
    dag_path = name + ".dag"
    index_path = name + ".dag.offset"

    faho = ahocorasick.Automaton()
    baho = ahocorasick.Automaton()
    for word in v:
        faho.add_word(word, (len(word), v[word]))
        baho.add_word(word[::-1], (len(word), v[word]))
    faho.make_automaton()
    baho.make_automaton()

    sz = np.array(0, dtype=np.int64)  # offset in bytes
    with open(dag_path, "wb") as dag_out, open(index_path, "wb") as idx_out:
        for sent in sentences:
            idx_out.write(sz.tobytes())
            n = len(sent)

            # Build backward adjacency list.
            adjl = defaultdict(list)
            for end, (len_word, v_idx) in baho.iter(sent[::-1]):
                start = n - (end + 1)
                j = start + len_word
                adjl[start].append((j, v_idx))
            bbuf = [n]
            for start in reversed(range(n)):
                adjl[start].sort()
                for j, v_idx in adjl[start]:
                    bbuf.extend([j, v_idx])
                bbuf.append(-1)
            dag_out.write(np.array(bbuf, dtype=np.int32).tobytes())

            # Build forward adjacency list.
            adjl = defaultdict(list)
            for end, (len_word, v_idx) in faho.iter(sent):
                end = end + 1
                i = end - len_word
                adjl[end].append((i, v_idx))
            fbuf = [n]
            for end in range(1, n + 1):
                adjl[end].sort()
                for i, v_idx in adjl[end]:
                    fbuf.extend([i, v_idx])
                fbuf.append(-1)
            dag_out.write(np.array(fbuf, dtype=np.int32).tobytes())

            assert len(bbuf) == len(fbuf)
            sz += len(bbuf) * 4
            sz += len(fbuf) * 4
    return dag_path, index_path


def main():
    from argparse import ArgumentParser

    from tqdm import tqdm

    from .preprocess_data import CorpusStats

    def iter_sentences(infile):
        stats = CorpusStats.read_header(infile)
        for line in tqdm(infile, total=stats.n, mininterval=5):
            yield line.rstrip("\n")

    parser = ArgumentParser(description="Create DAG files.")
    parser.add_argument("vocab_file", help="the vocabulary to load")
    parser.add_argument("sentences_file", help="the sentences file")
    parser.add_argument("out_name", help="the output stem")
    args = parser.parse_args()

    v = Vocab.load(args.vocab_file)
    with open(args.sentences_file, encoding="utf-8") as infile:
        paths = build_dag(v, iter_sentences(infile), args.out_name)
    for path in paths:
        print("Created:", path)
