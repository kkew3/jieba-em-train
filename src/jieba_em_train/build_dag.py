"""
# DAG format spec

File extension:

- `*.f.dag` for forward dag, where the neighbors in the adjacency list are in-neighbors.
- `*.b.dag` for backward dag, where the neighbors in the adjacency list are out-neighbors.

Data types:

- All tokens are `i32`.

Spec:

```
FILE           ::= ( DAG_SEGMENT )+
DAG_SEGMENT    ::= SEGMENT_LENGTH . NUM_DAG_NODE . ADJACENCY_LIST
ADJACENCY_LIST ::= ( NODE )+                                        // NUM_DAG_NODE repetition of NODE_LIST
NODE           ::= EDGE_LIST . TERMINAL
EDGE_LIST      ::= ( EDGE )+
EDGE           ::= OPPOSITE_END_NODE_IDX . VOCAB_IDX
TERMINAL       ::= "-1"
```

For example, let the vocabulary `V` be:

```
[0] 甲
[1] 甲乙
[2] 甲乙丙
[3] 乙
```

Let the sentences be:

```
[0] 甲乙甲乙
[1] 甲乙丙乙甲
```

Then the forward dag file is a flattened array of the following:

```c
18             // there are 18 numbers in this segment, including "18" itself;
4              // there are 4 nodes in the adjacency list;
0 0 -1         // current edge end is node 1;
               // (0, .): the word spans [0, 1); (., 0): the word is V[0]="甲";
0 1 1 3 -1     // current edge end is node 2;
               // (0, .): the word spans [0, 2); (., 1): the word is V[1]="甲乙";
               // (1, .): the word spans [1, 2); (., 3): the word is V[3]="乙";
2 0 -1         // current edge end is node 3;
               // (2, .): the word spans [2, 3); (., 0): the word is V[0]="甲";
2 1 3 3 -1     // current edge end is node 4;
               // (2, .): the word spans [2, 4); (., 1): the word is V[1]="甲乙";
               // (3, .): the word spans [3, 4); (., 3): the word is V[3]="乙";

19             // there are 18 numbers in this segment, including "19" itself;
5              // there are 5 nodes in the adjacency list;
0 0 -1         // current edge is node 1;
               // (0, .): the word spans [0, 1); (., 0): the word is V[0]="甲";
0 1 1 3 -1     // current edge end is node 2;
               // (0, .): the word spans [0, 2); (., 1): the word is V[1]="甲乙";
               // (1, .): the word spans [1, 2); (., 3): the word is V[3]="乙";
0 2 -1         // current edge end is node 3;
               // (0, .): the word spans [0, 3); (., 2): the word is V[2]="甲乙丙";
3 3 -1         // current edge end is node 4;
               // (3, .): the word spans [3, 4); (., 3): the word is V[3]="乙";
4 0 -1         // current edge end is node 5;
               // (4, .): the word spans [4, 5); (., 0): the word is V[0]="甲";
```

The backward dag file is a flattened array of the following:

```c
18             // there are 18 numbers in this segment, including "18" itself;
4              // there are 4 nodes in the adjacency list;
4 3 -1         // current edge start is node 3;
               // (4, .): the word spans [3, 4); (., 3): the word is V[3]="乙";
3 0 4 1 -1     // current edge start is node 2;
               // (3, .): the word spans [2, 3); (., 0): the word is V[0]="甲";
               // (3, .): the word spans [2, 4); (., 1): the word is V[1]="甲乙";
2 3 -1         // current edge start is node 1;
               // (2, .): the word spans [1, 2); (., 3): the word is V[3]="乙";
1 0 2 1 -1     // current edge start is node 0;
               // (1, .): the word spans [0, 1); (., 0): the word is V[0]="甲";
               // (2, .): the word spans [0, 2): (., 1): the word is V[1]="甲乙";

19             // there are 18 numbers in this segment, including "18" itself;
5              // there are 5 nodes in the adjacency list;
5 0 -1         // current edge start is node 4;
               // (5, .): the word spans [4, 5); (., 0): the word is V[0]="甲";
4 3 -1         // current edge start is node 3;
               // (4, .): the word spans [3, 4); (., 3): the word is V[3]="乙";
-1             // current edge start is node 2, and there is no edge;
2 3 -1         // current edge start is node 1;
               // (2, .): the word spans [1, 2); (., 3): the word is V[3]="乙";
1 0 2 1 3 2 -1 // current edge start is node 0;
               // (1, .): the word spans [0, 1); (., 0): the word is V[0]="甲";
               // (2, .): the word spans [0, 2); (., 1): the word is V[1]="甲乙";
               // (3, .): the word spans [0, 3): (., 2): the word is V[2]="甲乙丙";
```
"""

from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable

import ahocorasick
import numpy as np

from .preprocess_data import iter_sentences
from .vocab import Vocab


def build_dag_naive(v: Vocab, sentences, name):
    """A deprecated naive referential implementation not used in production."""
    fwd_path = name + ".f.dag"
    bwd_path = name + ".b.dag"
    adjl = [0, 0]  # the buffer for adjacency list
    with open(fwd_path, "wb") as fout, open(bwd_path, "wb") as bout:
        for sent in sentences:
            n = len(sent)
            adjl[1] = n

            # Build forward adjacency list.
            del adjl[2:]
            for end in range(1, n + 1):
                for i in range(end):
                    if sent[i:end] in v:
                        v_idx = v[sent[i:end]]
                        adjl.extend([i, v_idx])
                adjl.append(-1)
            adjl[0] = len(adjl)
            fout.write(np.array(adjl, dtype=np.int32).tobytes())

            # Build backward adjacency list.
            del adjl[2:]
            for start in reversed(range(n)):
                for j in range(start + 1, n + 1):
                    if sent[start:j] in v:
                        v_idx = v[sent[start:j]]
                        adjl.extend([j, v_idx])
                adjl.append(-1)
            adjl[0] = len(adjl)
            bout.write(np.array(adjl, dtype=np.int32).tobytes())
    return fwd_path, bwd_path


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
    parser = ArgumentParser(description="Create DAG files.")
    parser.add_argument("vocab_file", help="the vocabulary to load")
    parser.add_argument("sentences_file", help="the sentences file")
    parser.add_argument("out_name", help="the output stem")
    args = parser.parse_args()

    v = Vocab.load(args.vocab_file)
    with open(args.sentences_file, encoding="utf-8") as infile:
        paths = build_dag(
            v, iter_sentences(infile, mininterval=5), args.out_name
        )
    for path in paths:
        print("Created:", path)
