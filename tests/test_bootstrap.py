# tests/test_bootstrap.py
import numpy as np
import pytest
from MLstatkit import Bootstrapping

def test_bootstrap_runs_and_reproducible():
    y = np.array([0,1,0,1,0,1,1,0,1,0])
    p = np.array([0.1,0.9,0.2,0.8,0.3,0.7,0.6,0.4,0.9,0.05])
    s1, lo1, hi1 = Bootstrapping(y, p, metric_str="roc_auc", n_bootstraps=200, random_state=42)
    s2, lo2, hi2 = Bootstrapping(y, p, metric_str="roc_auc", n_bootstraps=200, random_state=42)
    assert (s1,lo1,hi1) == (s2,lo2,hi2)
    assert 0.0 <= lo1 <= s1 <= hi1 <= 1.0

def test_bootstrap_degenerate_raises():
    y = np.zeros(50, dtype=int)
    p = np.random.RandomState(0).rand(50)
    with pytest.raises(RuntimeError):
        Bootstrapping(y, p, metric_str="roc_auc", n_bootstraps=50, random_state=0)
