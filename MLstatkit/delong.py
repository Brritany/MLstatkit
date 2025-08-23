# MLstatkit/delong.py
from typing import Tuple, Union
import math
import numpy as np
import scipy.stats

ArrayLike = Union[list, np.ndarray]

def Delong_test(
    true: ArrayLike,
    prob_A: ArrayLike,
    prob_B: ArrayLike,
    return_ci: bool = False,
    alpha: float = 0.95
):
    """
    Perform DeLong's test for comparing the AUCs of two models.

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        True binary labels in {0, 1}.
    prob_A : array-like of shape (n_samples,)
        Predicted probabilities by model A.
    prob_B : array-like of shape (n_samples,)
        Predicted probabilities by model B.
    return_ci : bool, default=False
        If True, also return (ci_A, ci_B) for AUCs.
    alpha : float, default=0.95
        Confidence level for AUC CIs.

    Returns
    -------
    z_score : float
    p_value : float
    (ci_A, ci_B) : optional, tuple(lower, upper) for each model AUC.
    """
    y_true = np.asarray(true)
    pA = np.asarray(prob_A, dtype=float)
    pB = np.asarray(prob_B, dtype=float)

    def compute_midrank(x: np.ndarray) -> np.ndarray:
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        T2[J] = T + 1
        return T2

    def compute_ground_truth_statistics(y_true: np.ndarray):
        uniq = np.unique(y_true)
        assert uniq.size == 2 and np.all(uniq == np.array([0, 1])), "Ground truth must be binary in {0,1}."
        order = (-y_true).argsort()         # positives first
        label_1_count = int(y_true.sum())   # m
        return order, label_1_count

    def auc_ci(auc_val: float, var: float, alpha_val: float):
        z = scipy.stats.norm.ppf(1.0 - (1.0 - alpha_val) / 2.0)
        lower = auc_val - z * np.sqrt(var)
        upper = auc_val + z * np.sqrt(var)
        return max(0.0, float(lower)), min(1.0, float(upper))

    # ---- DeLong fast AUC and covariance ----
    order, m = compute_ground_truth_statistics(y_true)
    sorted_probs = np.vstack((pA, pB))[:, order]
    n = sorted_probs.shape[1] - m
    k = sorted_probs.shape[0]  # 2 models

    tx, ty, tz = [np.empty([k, size], dtype=np.float64) for size in [m, n, m + n]]
    for r in range(k):
        pos = sorted_probs[r, :m]
        neg = sorted_probs[r, m:]
        tx[r, :], ty[r, :], tz[r, :] = [
            compute_midrank(arr) for arr in (pos, neg, sorted_probs[r, :])
        ]

    # AUCs for A and B
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)

    # DeLong covariance
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    # z & p-value for difference in AUCs
    diff = (aucs[1] - aucs[0])
    denom = float((np.array([[1.0, -1.0]]) @ delongcov @ np.array([[1.0], [-1.0]])).flatten()[0])
    z = abs(diff) / math.sqrt(denom)
    p_value = float(scipy.stats.norm.sf(z) * 2.0)
    z_score = float(-z)  # keep sign convention from your original code

    if not return_ci:
        return z_score, p_value

    var_auc_A = float(delongcov[0, 0])
    var_auc_B = float(delongcov[1, 1])
    ci_A = auc_ci(float(aucs[0]), var_auc_A, alpha)
    ci_B = auc_ci(float(aucs[1]), var_auc_B, alpha)
    return z_score, p_value, ci_A, ci_B
