# tests/test_permutation.py
import numpy as np
from MLstatkit import Permutation_test

def test_permutation_detects_difference():
    rng = np.random.RandomState(0)
    y = rng.randint(0,2,400)
    pA = y*0.8 + rng.rand(400)*0.2  # better
    pB = rng.rand(400)              # worse
    ma, mb, p, bench, mean, std = Permutation_test(
        y, pA, pB, metric_str="roc_auc", n_bootstraps=200, random_state=1
    )
    assert p < 0.05
    assert ma > mb

def test_permutation_no_signal_has_large_p():
    rng = np.random.RandomState(1)
    y = rng.randint(0,2,300)
    pA = rng.rand(300)
    pB = rng.rand(300)
    _, _, p, *_ = Permutation_test(y, pA, pB, metric_str="roc_auc", n_bootstraps=150, random_state=2)
    assert p > 0.1  # 不做太嚴格門檻，避免偶發
