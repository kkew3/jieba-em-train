import collections
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from scipy import special as sp_special

from jieba_em_train import _core

from .eval_jieba import run_eval
from .preprocess_data import CorpusStats
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


def suggest_tau(corpus_file: str):
    """
    Suggest an initial tau based on median sentence length. In general the
    initial tau should be no smaller than this value.
    """
    bins = collections.defaultdict(int)
    with open(corpus_file, encoding="utf-8") as infile:
        stats = CorpusStats.read_header(infile)
        for line in infile:
            bins[len(line.rstrip("\n"))] += 1
    _n = stats.n // 2
    b_star = 0
    for b in sorted(bins):
        b_star = b
        if _n < bins[b]:
            break
        _n -= bins[b]
    return 1 / (1 + b_star)


def log(x):
    return np.log(np.maximum(x, 1e-20))


@dataclass
class GlobalPrior:
    """
    A global pseudocount applied to all words. The fields are pseudocount
    scales; set to zero to disable the corresponding prior.

    Parameters
    ----------
    kappa : float
        The pseudocount scale.
    """

    kappa: float

    def apply_prior(self, word_counts: list[float], v: Vocab):
        """Return log_theta."""
        logit_theta = np.logaddexp(
            log(np.array(word_counts)),
            log(self.kappa) + v.as_logprob_array(),
        )
        return logit_theta - sp_special.logsumexp(logit_theta)


@dataclass
class MainSubsetPrior:
    """
    A pseudocount prior applied separately to main and non-main words. The
    fields are pseudocount scales; set to zero to disable the corresponding
    prior.

    Parameters
    ----------
    mix_prior : np.ndarray, shape [2]
        The pseudocounts applied to the mixing distribution [main, non_main].
    kappa_main : float
        The pseudocount scale applied to the main words.
    kappa_non_main : float
        The pseudocount scale applied to the non-main words, default 0.
    """

    mix_prior: np.ndarray
    kappa_main: float
    kappa_non_main: float = 1e-20

    def apply_prior(self, word_counts: list[float], v: Vocab):
        """Return log_theta."""
        word_counts_ = np.array(word_counts)
        main_mask = v.as_main_mask_array()
        word_counts_partition_arr = (
            np.array(
                [
                    np.sum(word_counts_[main_mask]),
                    np.sum(word_counts_[~main_mask]),
                ]
            )
            + self.mix_prior
        )
        log_pi = log(word_counts_partition_arr) - log(
            np.sum(word_counts_partition_arr)
        )
        log_prior = v.as_logprob_array()
        log_word_counts = log(word_counts_)
        log_prior_main = log_prior - sp_special.logsumexp(log_prior[main_mask])
        logit_theta_main = np.logaddexp(
            log_word_counts, log(self.kappa_main) + log_prior_main
        )
        log_theta_main = logit_theta_main - sp_special.logsumexp(
            logit_theta_main
        )
        log_prior_non_main = log_prior - sp_special.logsumexp(
            log_prior[~main_mask]
        )
        logit_theta_non_main = np.logaddexp(
            log_word_counts, log(self.kappa_non_main) + log_prior_non_main
        )
        log_theta_non_main = logit_theta_non_main - sp_special.logsumexp(
            logit_theta_non_main
        )
        return np.where(
            main_mask,
            log_pi[0] + log_theta_main,
            log_pi[1] + log_theta_non_main,
        )


def train_loops(
    v: Vocab,
    prior: GlobalPrior | MainSubsetPrior,
    tau: float,
    n_iter: int,
    dag_name: str,
    val_msra_path: str,
    n_jobs: int = 1,
    return_option: Literal["best", "all"] = "best",
    verbose: bool = True,
):
    log_theta = v.as_logprob_array()
    hat_logprobs = log_theta + math.log(1 - tau)
    results = {}

    freq_c, f1 = val_against_freq_c(val_msra_path, v, log_theta)
    results[-1] = (f1, log_theta, tau, freq_c)
    if verbose:
        print(f"[{now()}] ITER -1 F1: {f1}")
    for train_iter in range(n_iter):
        word_counts, seqlen, cnt = _core.compute_values_parallel(
            hat_logprobs.tolist(), dag_name, n_jobs=n_jobs
        )
        log_theta = prior.apply_prior(word_counts, v)
        tau = cnt / (cnt + seqlen)
        hat_logprobs = log_theta + math.log(1 - tau)
        freq_c, f1 = val_against_freq_c(val_msra_path, v, log_theta)
        results[train_iter] = (f1, log_theta, tau, freq_c)
        if verbose:
            print(f"[{now()}] ITER {train_iter} F1: {f1}")
    results_list = list(results.items())
    best_iter, (best_f1, best_log_theta, best_tau, best_freq_c) = max(
        results_list, key=lambda x: x[1][0]
    )
    if verbose:
        print(f"Best iter: {best_iter} at F1: {best_f1}")
    if return_option == "all":
        return results
    return best_f1, (best_log_theta, best_tau, best_freq_c)
