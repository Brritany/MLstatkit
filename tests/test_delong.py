import numpy as np
from MLstatkit import Delong_test


def test_delong_basic_and_ci_output_types():
    """
    Test basic functionality of Delong_test and ensure correct output types.
    """
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    pa = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])  # Model A: nearly perfect separation
    pb = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])  # Model B: high but not perfect

    # return_ci=False, return_auc=False -> only z and p
    z, p = Delong_test(y, pa, pb, return_ci=False, return_auc=False, verbose=0)
    assert isinstance(z, float)
    assert isinstance(p, float)

    # return_ci=True, return_auc=False -> z, p, ci_A, ci_B
    z2, p2, ciA, ciB = Delong_test(y, pa, pb, return_ci=True, return_auc=False, alpha=0.95, verbose=0)
    assert isinstance(z2, float)
    assert isinstance(p2, float)
    assert isinstance(ciA, (tuple, list)) and len(ciA) == 2
    assert isinstance(ciB, (tuple, list)) and len(ciB) == 2


def test_delong_ci_ranges_and_perfect_separation():
    """
    Check CI ranges and behavior when a model has perfect (or near-perfect) separation.
    For perfect separation under DeLong diagonals, CI can collapse to (1.0, 1.0).
    """
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    pa = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])  # Perfect ordering for A
    pb = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])

    _, _, ciA, ciB = Delong_test(y, pa, pb, return_ci=True, return_auc=False, alpha=0.95, verbose=0)

    # For model A, CI should be within [0, 1] and quite tight near 1.0; in many cases it collapses to (1.0, 1.0).
    assert 0.0 <= ciA[0] <= ciA[1] <= 1.0
    # Accept both the collapsed (1.0,1.0) and near-1 intervals.
    if ciA != (1.0, 1.0):
        assert ciA[1] == 1.0 and ciA[0] >= 0.9

    # For model B, CI should be within [0, 1] and lower <= upper
    assert 0.0 <= ciB[0] <= ciB[1] <= 1.0


def test_delong_ci_alpha_monotonicity():
    """
    Check that narrower alpha gives a narrower CI (90% CI <= 95% CI).
    """
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)

    # Synthetic data: model A slightly better than B
    pa = np.clip(0.5 + 0.4 * (y - 0.5) + rng.normal(0, 0.15, y.size), 0, 1)
    pb = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.20, y.size), 0, 1)

    _, _, ciA95, ciB95 = Delong_test(y, pa, pb, return_ci=True, return_auc=False, alpha=0.95, verbose=0)
    _, _, ciA90, ciB90 = Delong_test(y, pa, pb, return_ci=True, return_auc=False, alpha=0.90, verbose=0)

    width = lambda ci: ci[1] - ci[0]

    # 90% CI should be narrower or equal to 95% CI
    assert width(ciA90) <= width(ciA95)
    assert width(ciB90) <= width(ciB95)


def test_delong_no_signal_p_not_small():
    """
    When both models are random with no signal, p-value should not be small.
    """
    rng = np.random.default_rng(42)
    y = np.array([0, 1] * 100)
    pa = rng.random(y.size)
    pb = rng.random(y.size)

    z, p = Delong_test(y, pa, pb, return_ci=False, return_auc=False, verbose=0)
    assert isinstance(z, float)
    assert p > 0.05


def test_signed_z_and_symmetry():
    """
    Verify z carries the correct sign and p-value symmetry upon swapping models.
    By construction, B > A; thus z > 0 for (A,B) and z < 0 for (B,A). Two-sided p is identical.
    """
    rng = np.random.default_rng(2025)
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:600] = 1
    rng.shuffle(y)

    # B better than A
    A = 1.2 * y + rng.normal(0, 1.0, size=n)
    B = 1.6 * y + rng.normal(0, 1.0, size=n)

    z_ab, p_ab = Delong_test(y, A, B, return_ci=False, return_auc=False, verbose=0)
    z_ba, p_ba = Delong_test(y, B, A, return_ci=False, return_auc=False, verbose=0)

    assert z_ab > 0
    assert z_ba < 0
    # Two-sided p-values should be identical when swapping models
    assert abs(p_ab - p_ba) < 1e-12


def test_fallback_to_bootstrap_on_degeneracy():
    """
    Force a degenerate situation (near-perfect separation/opposition) to trigger bootstrap.
    Check that method == 'bootstrap' and outputs are well-formed.
    """
    # Construct a strongly degenerate case
    y = np.array([0, 0, 0, 1, 1, 1] * 60, dtype=int)  # 360 samples
    A = y.astype(float)           # perfect for A
    B = 1.0 - y.astype(float)     # strictly opposite for B

    # Request full output (need 'info' to inspect method)
    z, p, ciA, ciB, aucA, aucB, info = Delong_test(
        y, A, B,
        return_ci=True, return_auc=True,
        n_boot=1000, random_state=7,
        verbose=0
    )

    assert info["method"] == "bootstrap"
    assert isinstance(z, float) or (np.isinf(z))
    assert 0.0 <= ciA[0] <= ciA[1] <= 1.0
    assert 0.0 <= ciB[0] <= ciB[1] <= 1.0
    assert 0.0 <= aucA <= 1.0 and 0.0 <= aucB <= 1.0
    # In this extreme setting, p-value should be extremely small
    assert p <= 1e-6


def test_verbose_levels_no_error():
    """
    Ensure that verbose levels 0, 1, and 2 execute without error.
    Do not assert on printed content; only check the function runs and returns.
    """
    rng = np.random.default_rng(11)
    n = 500
    y = np.zeros(n, dtype=int)
    y[:300] = 1
    rng.shuffle(y)

    A = 1.5 * y + rng.normal(0, 1.0, size=n)
    B = 1.3 * y + rng.normal(0, 1.0, size=n)

    # verbose=0
    _ = Delong_test(y, A, B, return_ci=True, return_auc=True, verbose=0)

    # verbose=1
    _ = Delong_test(y, A, B, return_ci=True, return_auc=True, verbose=1)

    # verbose=2 (no bootstrap progress expected here; just ensure no errors)
    _ = Delong_test(y, A, B, return_ci=True, return_auc=True, verbose=2, progress_every=0)
