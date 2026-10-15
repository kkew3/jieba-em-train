import math
import time
from datetime import datetime
from typing import Literal

import numpy as np

from jieba_em_train import _core

from .eval_jieba import run_eval
from .vocab import Vocab


def now():
    return datetime.now().isoformat()


def val_against_freq_c(val_msra_path: str, v: Vocab, log_theta):
    best_freq_c = None
    best_score = None
    freq_c_shift = [2, 10, 18, 26]
    for cs in freq_c_shift:
        freq_c = 0xFFFFFFFF >> cs
        f1 = run_eval(val_msra_path, v, log_theta, freq_c).get()
        if best_score is None or f1 > best_score:
            best_freq_c = freq_c
            best_score = f1
    assert best_score is not None
    assert best_freq_c is not None
    return best_freq_c, best_score


def get_splits(dag_name: str, n_jobs: int):
    offset_file = f"{dag_name}.dag.offset"
    offset_arr = np.memmap(offset_file, dtype=np.int64, mode="r")
    step = int(offset_arr.shape[0] / n_jobs) + 1
    offsets = []
    for j in range(0, offset_arr.shape[0], step):
        offsets.append(offset_arr[j].tolist())
    sizes = []
    assert offsets
    for j in range(len(offsets) - 1):
        sizes.append(offsets[j + 1] - offsets[j])
    sizes.append(0)
    return offsets, sizes


def train_loops(
    v: Vocab,
    kappa: float,
    n_iter: int,
    dag_name: str,
    val_msra_path: str,
    n_jobs: int = 1,
    return_option: Literal["best", "all"] = "best",
):
    log_theta = v.as_logprob_array()
    log_theta0 = v.as_logprob_array()
    tau = 0.1
    hat_logprobs = log_theta + math.log(1 - tau)
    results = {}
    splits = get_splits(dag_name, n_jobs) if n_jobs > 1 else None

    freq_c, f1 = val_against_freq_c(val_msra_path, v, log_theta)
    results[-1] = (f1, log_theta, tau, freq_c)
    print(f"[{now()}] ITER -1 F1: {f1}")
    # Sleep a bit so that `print` above gets flushed before GIL is released
    # during `_core` call.
    time.sleep(1)
    for train_iter in range(n_iter):
        if kappa <= 0:
            pseudo_word_counts = 0
        else:
            pseudo_word_counts = np.exp(log_theta0 + math.log(kappa))
        if splits is None:
            word_counts, seqlen, cnt = _core.compute_values(
                hat_logprobs.tolist(), dag_name
            )
        else:
            word_counts, seqlen, cnt = _core.compute_values_parallel(
                hat_logprobs.tolist(), dag_name, *splits
            )
        word_counts = np.array(word_counts) + pseudo_word_counts
        log_theta = np.log(word_counts) - np.log(np.sum(word_counts))
        tau = cnt / (cnt + seqlen)
        hat_logprobs = log_theta + math.log(1 - tau)
        freq_c, f1 = val_against_freq_c(val_msra_path, v, log_theta)
        results[train_iter] = (f1, log_theta, tau, freq_c)
        print(f"[{now()}] ITER {train_iter} F1: {f1}")
        time.sleep(1)
    results_list = list(results.items())
    best_iter, (best_f1, best_log_theta, best_tau, best_freq_c) = max(
        results_list, key=lambda x: x[1][0]
    )
    print(f"Best iter: {best_iter} at F1: {best_f1}")
    if return_option == "all":
        return results
    return best_f1, (best_log_theta, best_tau, best_freq_c)
