from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import math
import scipy.stats
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score,
    roc_auc_score, average_precision_score, precision_recall_curve, auc
)

def Delong_test(true, prob_A, prob_B, return_ci: bool = False, alpha: float = 0.95):
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
        If True, also return the confidence intervals (CIs) of AUCs for both models.
    alpha : float, default=0.95
        Confidence level for AUC CIs (e.g., 0.95 for 95% CI).

    Returns
    -------
    z_score : float
    p_value : float
    ci_A : tuple(float, float), optional
    ci_B : tuple(float, float), optional

    Example
    -------
    >>> true = [0, 1, 0, 1]
    >>> prob_A = [0.1, 0.4, 0.35, 0.8]
    >>> prob_B = [0.2, 0.3, 0.4, 0.7]
    >>> z, p = Delong_test(true, prob_A, prob_B)
    >>> z, p, ci_A, ci_B = Delong_test(true, prob_A, prob_B, return_ci=True)
    """
    true = np.asarray(true)
    prob_A = np.asarray(prob_A, dtype=float)
    prob_B = np.asarray(prob_B, dtype=float)

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
        assert np.array_equal(np.unique(y_true), [0, 1]), "Ground truth must be binary."
        order = (-y_true).argsort()         # positives first
        label_1_count = int(y_true.sum())   # m
        return order, label_1_count

    def auc_ci(auc_val: float, var: float, alpha_val: float):
        # Normal approximation CI using variance from DeLong covariance diagonal
        z = scipy.stats.norm.ppf(1.0 - (1.0 - alpha_val) / 2.0)
        lower = auc_val - z * np.sqrt(var)
        upper = auc_val + z * np.sqrt(var)
        return max(0.0, float(lower)), min(1.0, float(upper))

    # ---- DeLong fast AUC and covariance ----
    order, m = compute_ground_truth_statistics(true)
    sorted_probs = np.vstack((prob_A, prob_B))[:, order]
    n = sorted_probs.shape[1] - m
    k = sorted_probs.shape[0]  # 2 models

    tx, ty, tz = [np.empty([k, size], dtype=np.float64) for size in [m, n, m + n]]
    for r in range(k):
        pos = sorted_probs[r, :m]
        neg = sorted_probs[r, m:]
        tx[r, :], ty[r, :], tz[r, :] = [
            compute_midrank(arr) for arr in (pos, neg, sorted_probs[r, :])
        ]

    # AUCs for A and B (vector of length 2). Important: name is 'aucs' (not sklearn.metrics.auc)
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)

    # DeLong covariance
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    # z & p-value for difference in AUCs
    L = np.array([[1.0, -1.0]])
    denom = float((L @ delongcov @ L.T).flatten()[0])
    # z = np.abs(np.diff(aucs)) / np.sqrt(denom)
    # p_value = float(scipy.stats.norm.sf(abs(z)) * 2.0)
    # z_score = -float(z[0])  # sign convention consistent with original code

    diff = (aucs[1] - aucs[0])
    z = abs(diff) / math.sqrt(denom)
    p_value = float(scipy.stats.norm.sf(z) * 2.0)
    z_score = float(-z)

    if not return_ci:
        return z_score, p_value

    # Confidence intervals using diagonal variances
    var_auc_A = float(delongcov[0, 0])
    var_auc_B = float(delongcov[1, 1])
    ci_A = auc_ci(float(aucs[0]), var_auc_A, alpha)
    ci_B = auc_ci(float(aucs[1]), var_auc_B, alpha)

    # ci_A = auc_ci(auc[0], var_auc_A, alpha)
    # ci_B = auc_ci(auc[1], var_auc_B, alpha)

    return z_score, p_value, ci_A, ci_B


def get_metric_fn(metric_str: str, threshold: float = 0.5, average: str = 'macro'):
    """
    Return a callable metric function based on the specified metric string.

    Parameters
    ----------
    metric_str : {'f1','accuracy','recall','precision','roc_auc','pr_auc','average_precision'}
    threshold : float, default=0.5
        Threshold for converting probabilities to binary labels for certain metrics.
    average : str, default='macro'
        Averaging method for multi-class/multi-label scenarios.

    Returns
    -------
    function
        A function f(y_true, y_pred_prob) -> float
    """
    metric_str = (metric_str or "").lower()

    if metric_str == 'f1':
        return lambda y, y_pred: f1_score(y, (np.asarray(y_pred) >= threshold).astype(int), average=average)

    if metric_str == 'accuracy':
        return lambda y, y_pred: accuracy_score(y, (np.asarray(y_pred) >= threshold).astype(int))

    if metric_str == 'recall':
        return lambda y, y_pred: recall_score(y, (np.asarray(y_pred) >= threshold).astype(int), average=average)

    if metric_str == 'precision':
        return lambda y, y_pred: precision_score(y, (np.asarray(y_pred) >= threshold).astype(int), average=average)

    if metric_str == 'roc_auc':
        return lambda y, y_pred: roc_auc_score(y, np.asarray(y_pred))

    if metric_str == 'average_precision':
        return lambda y, y_pred: average_precision_score(y, np.asarray(y_pred))

    if metric_str == 'pr_auc':
        def _pr_auc(y, y_pred):
            precision, recall, _ = precision_recall_curve(y, np.asarray(y_pred))
            return float(auc(recall, precision))
        return _pr_auc

    raise ValueError(f"Unsupported metric_str: {metric_str!r}")
