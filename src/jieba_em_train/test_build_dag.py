import tempfile
from pathlib import Path

import numpy as np

from .build_dag import build_dag
from .vocab import Vocab


def test_build_dag():
    v = Vocab()
    for word in ["甲", "甲乙", "甲乙丙", "乙"]:
        v.append(word, 2)
    sentences = ["甲乙甲乙", "甲乙丙乙甲"]

    with tempfile.TemporaryDirectory(
        suffix=".tmp", prefix="test_build_dag", dir="."
    ) as tmpdir:
        dag_path, idx_path = build_dag(v, sentences, str(Path(tmpdir) / "test"))
        adjl = np.fromfile(dag_path, dtype=np.int32)
        # fmt: off
        assert np.all(
            adjl
            == np.array(
                [
                    4,
                    4, 3, -1,
                    3, 0, 4, 1, -1,
                    2, 3, -1,
                    1, 0, 2, 1, -1,
                    4,
                    0, 0, -1,
                    0, 1, 1, 3, -1,
                    2, 0, -1,
                    2, 1, 3, 3, -1,

                    5,
                    5, 0, -1,
                    4, 3, -1,
                    -1,
                    2, 3, -1,
                    1, 0, 2, 1, 3, 2, -1,
                    5,
                    0, 0, -1,
                    0, 1, 1, 3, -1,
                    0, 2, -1,
                    3, 3, -1,
                    4, 0, -1,
                ],
                dtype=np.int32,
            )
        )
        # fmt: on
        offset = np.fromfile(idx_path, dtype=np.int64)
        assert np.all(offset == np.array([0, 34 * 4], dtype=np.int64))
