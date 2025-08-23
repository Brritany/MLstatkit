# tests/test_conversions.py
import math
import pytest
from MLstatkit import AUC2OR

def test_auc2or_identity_at_point5():
    assert AUC2OR(0.5) == pytest.approx(1.0, rel=1e-6)

def test_auc2or_monotonic():
    assert AUC2OR(0.7) > AUC2OR(0.6) > AUC2OR(0.5)

def test_auc2or_high_auc_reasonable():
    or95 = AUC2OR(0.95)
    assert or95 > 10  # 很高，但不應為 inf/NaN
    assert math.isfinite(or95)
