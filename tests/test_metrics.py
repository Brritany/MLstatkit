# tests/test_metrics.py
import numpy as np
import pytest
from MLstatkit import get_metric_fn

def test_basic_classification_metrics_binary_threshold():
    y = np.array([0,1,0,1,0,1,1,0])
    p = np.array([0.1,0.9,0.2,0.8,0.3,0.7,0.6,0.4])
    for m in ["f1","accuracy","recall","precision"]:
        fn = get_metric_fn(m, threshold=0.5, average="binary")
        val = fn(y, p)
        assert 0.0 <= val <= 1.0

def test_roc_auc_and_ap_like():
    y = np.array([0,1,0,1,0,1,1,0])
    p = np.array([0.1,0.9,0.2,0.8,0.3,0.7,0.6,0.4])
    roc = get_metric_fn("roc_auc")(y, p)
    ap  = get_metric_fn("average_precision")(y, p)
    pr  = get_metric_fn("pr_auc")(y, p)
    assert 0.0 <= roc <= 1.0
    assert 0.0 <= ap <= 1.0
    assert 0.0 <= pr <= 1.0
    # AP 與 PR-AUC 通常不同，但應大致相關
    assert abs(ap - pr) < 0.2

def test_unsupported_metric_raises():
    with pytest.raises(ValueError):
        get_metric_fn("not_a_metric")
