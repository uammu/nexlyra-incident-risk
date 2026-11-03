import numpy as np

from incident_risk.cv import assert_chronological, time_folds
from incident_risk.metrics import bootstrap_ci


def test_cv_folds_respect_time(fake_df, cfg):
    folds = time_folds(fake_df, cfg, n_splits=4)
    assert len(folds) == 4
    assert_chronological(fake_df, folds, cfg)


def test_bootstrap_ci_ordering():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    ci = bootstrap_ci(y, p, 50, n_boot=100)
    assert ci["pr_auc"]["lo95"] <= ci["pr_auc"]["mean"] <= ci["pr_auc"]["hi95"]
