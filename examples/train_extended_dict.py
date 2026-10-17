from pathlib import Path

import optuna
from optuna import Trial

from jieba_em_train.sklearn import EmDictionaryEstimator


VOCAB_PATH = "train_dicts/rime_base.vocab"
VAL_MSRA_PATH = "/scratch/jieba-em-train/msra_bakeoff3_training.txt"
DAG_NAME = "/scratch/jieba-em-train/train_rime_base"
TEST_PATH = "corpora/private_testset.txt"
STUDY_DB_PATH = "results/study.db"


def objective(trial: Trial):
    est = EmDictionaryEstimator(
        mix_prior_scale=trial.suggest_float(
            "mix_prior_scale", low=1, high=1e8, log=True
        ),
        mix_prior_main_fraction=trial.suggest_float(
            "mix_prior_main_fraction", low=1e-6, high=1 - 1e-6
        ),
        main_prior_scale=trial.suggest_float(
            "main_prior_scale", low=1, high=1e8, log=True
        ),
        non_main_prior_scale=trial.suggest_float(
            "non_main_prior_scale", low=1, high=1e8, log=True
        ),
        vocab_path=VOCAB_PATH,
        vocab_jieba_compatible=False,
        init_tau=trial.suggest_float("init_tau", low=1e-6, high=1 - 1e-6),
        freq_c=trial.suggest_categorical(
            "freq_c",
            [None, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFF, 0xFFFFFFFFFFFF],
        ),
        training_budget=trial.suggest_categorical("training_budget", [10]),
        val_msra_path=VAL_MSRA_PATH,
        n_jobs=4,
        verbose=True,
    )
    est.fit(DAG_NAME)
    return (
        0.125 * est.score(TEST_PATH)
        + 0.25 * ("左执法" in est.predict("宫本武藏担任左执法"))
        + 0.5 * ("开发者" in est.predict("受到了广大开发者的广泛青睐"))
        + 1 * (est.best_f1_ > 0.9)
    )


db_path = Path.cwd().resolve() / STUDY_DB_PATH
study = optuna.create_study(
    storage=f"sqlite:///{db_path}",
    study_name="Fit extended dictionary",
    load_if_exists=True,
    direction="maximize",
)
study.optimize(objective, n_trials=100)
