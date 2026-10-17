import numpy as np
from sklearn.exceptions import NotFittedError

from . import em_training, eval_jieba
from .vocab import Vocab


class EmDictionaryEstimator:
    def __init__(
        self,
        mix_prior_scale=1.0,
        mix_prior_main_fraction=0.5,
        main_prior_scale=1.0,
        non_main_prior_scale=1.0,
        vocab_path=None,
        vocab_jieba_compatible=False,
        init_tau=0.125,
        freq_c=None,
        training_budget=50,
        val_msra_path=None,
        n_jobs=1,
        verbose=False,
        **kwargs,
    ):
        self.mix_prior_scale = mix_prior_scale
        self.mix_prior_main_fraction = mix_prior_main_fraction
        self.main_prior_scale = main_prior_scale
        self.non_main_prior_scale = non_main_prior_scale
        self.vocab_path = vocab_path
        self.vocab_jieba_compatible = vocab_jieba_compatible
        self.init_tau = init_tau
        self.freq_c = freq_c
        self.training_budget = training_budget
        self.val_msra_path = val_msra_path
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.is_fitted_ = False

    def set_params(self, **params):
        if "mix_prior_scale" in params:
            self.mix_prior_scale = params["mix_prior_scale"]
        if "mix_prior_main_fraction" in params:
            self.mix_prior_main_fraction = params["mix_prior_main_fraction"]
        if "main_prior_scale" in params:
            self.main_prior_scale = params["main_prior_scale"]
        if "non_main_prior_scale" in params:
            self.non_main_prior_scale = params["non_main_prior_scale"]
        if "vocab_path" in params:
            self.vocab_path = params["vocab_path"]
        if "vocab_jieba_compatible" in params:
            self.vocab_jieba_compatible = params["vocab_jieba_compatible"]
        if "init_tau" in params:
            self.init_tau = params["init_tau"]
        if "freq_c" in params:
            self.freq_c = params["freq_c"]
        if "training_budget" in params:
            self.training_budget = params["training_budget"]
        if "val_msra_path" in params:
            self.val_msra_path = params["val_msra_path"]
        if "n_jobs" in params:
            self.n_jobs = params["n_jobs"]
        if "verbose" in params:
            self.verbose = params["verbose"]
        self._configure_jieba()
        return self

    def get_params(self, deep=True):
        return {
            "mix_prior_scale": self.mix_prior_scale,
            "mix_prior_main_fraction": self.mix_prior_main_fraction,
            "main_prior_scale": self.main_prior_scale,
            "non_main_prior_scale": self.non_main_prior_scale,
            "vocab_path": self.vocab_path,
            "vocab_jieba_compatible": self.vocab_jieba_compatible,
            "init_tau": self.init_tau,
            "freq_c": self.freq_c,
            "training_budget": self.training_budget,
            "val_msra_path": self.val_msra_path,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
        }

    def fit(self, X: str, y=None, **params):
        self.set_params(**params)
        dag_name = X
        assert self.vocab_path and self.val_msra_path
        v = Vocab.load(
            self.vocab_path, jieba_compatible=self.vocab_jieba_compatible
        )
        prior = em_training.MainSubsetPrior(
            mix_prior=self.mix_prior_main_fraction
            * np.array(
                [self.mix_prior_main_fraction, 1 - self.mix_prior_main_fraction]
            ),
            kappa_main=self.main_prior_scale,
            kappa_non_main=self.non_main_prior_scale,
        )
        best_f1, (log_theta, _tau, _freq_c) = em_training.train_loops(
            v,
            prior,
            tau=self.init_tau,
            n_iter=self.training_budget,
            dag_name=dag_name,
            val_msra_path=self.val_msra_path,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.best_f1_ = best_f1
        self.log_theta_ = log_theta
        self.is_fitted_ = True
        self._configure_jieba()
        return self

    def _configure_jieba(self):
        if not self.is_fitted_:
            return
        v = Vocab.load(
            self.vocab_path, jieba_compatible=self.vocab_jieba_compatible
        )
        freq_c = (
            -float(np.min(self.log_theta_)) + 1e-6
            if self.freq_c is None
            else self.freq_c
        )
        self.jb_ = eval_jieba.configure_jieba(v, self.log_theta_, freq_c)

    def score(self, X: str, y=None):
        if not self.is_fitted_:
            raise NotFittedError

        segmented_sents_path = X

        v = Vocab.load(
            self.vocab_path, jieba_compatible=self.vocab_jieba_compatible
        )
        freq_c = (
            -float(np.min(self.log_theta_)) + 1e-6
            if self.freq_c is None
            else self.freq_c
        )

        return eval_jieba.run_eval(
            segmented_sents_path, v, self.log_theta_, freq_c
        ).get()

    def predict(self, X: str):
        if not self.is_fitted_:
            raise NotFittedError

        text = X

        if getattr(self, "jb_", None) is None:
            self._configure_jieba()

        return self.jb_.cut(text)
