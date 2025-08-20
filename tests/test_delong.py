import numpy as np
from MLstatkit.stats import Delong_test


def test_delong_basic_and_ci_output_types():
    """
    Test basic functionality of Delong_test and ensure correct output types.
    """
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    pa = np.array([.1, .9, .2, .8, .3, .7, .4, .6])   # Model A: perfect separation → AUC=1
    pb = np.array([.2, .8, .3, .7, .4, .6, .5, .5])   # Model B: high score but not perfect

    # Test return_ci=False
    z, p = Delong_test(y, pa, pb, return_ci=False)
    assert isinstance(z, float)
    assert isinstance(p, float)

    # Test return_ci=True
    z2, p2, ciA, ciB = Delong_test(y, pa, pb, return_ci=True, alpha=0.95)
    assert isinstance(z2, float)
    assert isinstance(p2, float)
    assert isinstance(ciA, (tuple, list)) and len(ciA) == 2
    assert isinstance(ciB, (tuple, list)) and len(ciB) == 2


def test_delong_ci_ranges_and_perfect_separation():
    """
    Check CI ranges and behavior when model has perfect separation.
    """
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    pa = np.array([.1, .9, .2, .8, .3, .7, .4, .6])   # Perfect separation
    pb = np.array([.2, .8, .3, .7, .4, .6, .5, .5])

    _, _, ciA, ciB = Delong_test(y, pa, pb, return_ci=True, alpha=0.95)

    # For model A, CI should collapse to (1.0, 1.0) due to zero variance
    assert ciA == (1.0, 1.0)

    # For model B, CI should be within [0, 1] and lower <= upper
    assert 0.0 <= ciB[0] <= ciB[1] <= 1.0


def test_delong_ci_alpha_monotonicity():
    """
    Check that narrower alpha gives a narrower CI (90% CI <= 95% CI).
    """
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)

    # Generate synthetic data: model A slightly better than B
    pa = np.clip(0.5 + 0.4 * (y - 0.5) + rng.normal(0, 0.15, y.size), 0, 1)
    pb = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.20, y.size), 0, 1)

    _, _, ciA95, ciB95 = Delong_test(y, pa, pb, return_ci=True, alpha=0.95)
    _, _, ciA90, ciB90 = Delong_test(y, pa, pb, return_ci=True, alpha=0.90)

    width = lambda ci: ci[1] - ci[0]

    # 90% CI should be narrower or equal to 95% CI
    assert width(ciA90) <= width(ciA95)
    assert width(ciB90) <= width(ciB95)
