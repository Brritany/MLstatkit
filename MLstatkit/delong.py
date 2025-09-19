# MLstatkit/delong.py
from typing import Union, Tuple, Optional, Dict, Any, List
import math
import numpy as np
import scipy.stats

ArrayLike = Union[list, np.ndarray]

def Delong_test(
    true: ArrayLike,
    prob_A: ArrayLike,
    prob_B: ArrayLike,
    *,
    alpha: float = 0.95,
    return_ci: bool = True,
    return_auc: bool = True,
    n_boot: int = 5000,
    random_state: Optional[int] = None,
    verbose: int = 0,            # 0: silent, 1: key steps, 2: detailed
    progress_every: int = 0       # only used when verbose >= 2
) -> Tuple:
    """
    DeLong test for comparing AUCs of two models with automatic fallback to bootstrap
    when degeneracy is detected (non-positive or invalid variance of AUC difference).

    The z statistic is signed as z = (AUC_B - AUC_A) / SE.
    A two-sided p-value is reported, aligned with R pROC::roc.test.

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        Binary ground truth labels in {0,1}.
    prob_A : array-like of shape (n_samples,)
        Scores or probabilities from model A.
    prob_B : array-like of shape (n_samples,)
        Scores or probabilities from model B.
    alpha : float, default=0.95
        Confidence level for per-model AUC CIs (normal approximation, clipped to [0,1]).
    return_ci : bool, default=True
        Whether to return (ci_A, ci_B).
    return_auc : bool, default=True
        Whether to return (auc_A, auc_B).
    n_boot : int, default=5000
        Number of bootstrap resamples used when fallback path is triggered.
    random_state : int or None
        RNG seed for reproducibility in bootstrap.
    verbose : {0,1,2}, default=0
        0: no output;
        1: key steps (samples, method, z/p, CIs);
        2: detailed (also shows var_diff, diff, optional bootstrap progress).
    progress_every : int, default=0
        When verbose >= 2, print bootstrap progress every N iterations (0 disables).

    Returns
    -------
    Depending on flags (return_ci, return_auc), returns:
        - if return_ci=False and return_auc=False:
            (z, p_value)
        - if return_ci=True and return_auc=False:
            (z, p_value, ci_A, ci_B)
        - if return_ci=False and return_auc=True:
            (z, p_value, auc_A, auc_B)
        - if return_ci=True and return_auc=True:
            (z, p_value, ci_A, ci_B, auc_A, auc_B, info)

    info (only returned in the last case) is a dict with keys:
        method: 'delong' or 'bootstrap'
        var_diff: variance of AUC difference (if DeLong path)
        tie_rate_A, tie_rate_B: tie proportions in A and B
        n_pos, n_neg: class counts
        n_boot: number of bootstrap samples used (if bootstrap)
        messages: list of (level, message) tuples recorded during the run
    """
    # ---------- logger ----------
    logs: List[Tuple[int, str]] = []
    def log(level: int, msg: str):
        if verbose >= level:
            print(msg)
        logs.append((level, msg))

    # ---------- input checks ----------
    y_true = np.asarray(true).ravel().astype(int)
    pA = np.asarray(prob_A, dtype=np.float64).ravel()
    pB = np.asarray(prob_B, dtype=np.float64).ravel()

    if y_true.shape[0] != pA.shape[0] or pA.shape[0] != pB.shape[0]:
        raise ValueError("true, prob_A, prob_B must have the same length.")
    if y_true.ndim != 1 or pA.ndim != 1 or pB.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays.")
    if not np.isfinite(pA).all() or not np.isfinite(pB).all():
        raise ValueError("prob_A and prob_B must be finite.")
    uniq = np.unique(y_true)
    if uniq.size != 2 or not np.array_equal(uniq, np.array([0, 1])):
        raise ValueError("Ground truth must be binary in {0,1}.")

    m = int(y_true.sum())
    n = y_true.size - m
    if m == 0 or n == 0:
        raise ValueError("At least one positive and one negative sample are required.")

    rng = np.random.default_rng(random_state)

    # ---------- quick diagnostics ----------
    def tie_rate(x: np.ndarray) -> float:
        return 1.0 - (len(np.unique(x)) / len(x))
    trA = tie_rate(pA); trB = tie_rate(pB)

    if verbose >= 1:
        log(1, "=== DeLong Test ===")
        log(1, f"Samples: total={y_true.size}, pos={m}, neg={n}")
        log(1, f"Tie rate: A={trA:.3f}, B={trB:.3f}")

    # ---------- helpers ----------
    def compute_midrank(x: np.ndarray) -> np.ndarray:
        # Return 1-based midranks (stable under ties)
        J = np.argsort(x, kind="mergesort")
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
        T2[J] = T + 1.0
        return T2

    def auc_ci(auc_val: float, var: float, alpha_val: float) -> Tuple[float, float]:
        # Normal-approx CI clipped to [0,1]; guard against tiny negative var
        var = max(0.0, float(var))
        zc = scipy.stats.norm.ppf(1.0 - (1.0 - alpha_val) / 2.0)
        half = zc * math.sqrt(var)
        lo = max(0.0, float(auc_val - half))
        hi = min(1.0, float(auc_val + half))
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    def delong_core(scoresA: np.ndarray, scoresB: np.ndarray):
        # Compute AUCs and covariance via fast DeLong with midranks
        order = (-y_true).argsort(kind="mergesort")  # positives first
        sp = np.vstack((scoresA, scoresB))[:, order]  # (2, m+n)
        k = sp.shape[0]
        tx, ty, tz = [np.empty([k, size], dtype=np.float64) for size in [m, n, m + n]]
        for r in range(k):
            pos = sp[r, :m]; neg = sp[r, m:]
            tx[r, :], ty[r, :], tz[r, :] = [
                compute_midrank(arr) for arr in (pos, neg, sp[r, :])
            ]
        aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
        auc_A, auc_B = float(aucs[0]), float(aucs[1])

        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01, bias=False)
        sy = np.cov(v10, bias=False)
        cov = sx / m + sy / n  # 2x2 covariance matrix

        contrast = np.array([-1.0, 1.0])  # B - A
        var_diff = float(contrast @ cov @ contrast.T)
        return auc_A, auc_B, cov, var_diff

    def bootstrap_diff(y: np.ndarray, sA: np.ndarray, sB: np.ndarray, n_boot: int):
        # Bootstrap distribution of AUC_B - AUC_A
        idx = np.arange(y.shape[0])
        diffs = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            bb = rng.choice(idx, size=idx.size, replace=True)
            yb, a, b_ = y[bb], sA[bb], sB[bb]
            m_ = int(yb.sum()); n_ = yb.size - m_
            if m_ == 0 or n_ == 0:
                diffs[b] = np.nan
                continue
            ord_ = (-yb).argsort(kind="mergesort")
            concat = np.vstack((a, b_))[:, ord_]
            tx, ty, tz = [np.empty([2, size], dtype=np.float64) for size in [m_, n_, m_ + n_]]
            for r in range(2):
                pos = concat[r, :m_]; neg = concat[r, m_:]
                tx[r, :], ty[r, :], tz[r, :] = [
                    compute_midrank(arr) for arr in (pos, neg, concat[r, :])
                ]
            aucs_b = tz[:, :m_].sum(axis=1) / (m_ * n_) - (m_ + 1.0) / (2.0 * n_)
            diffs[b] = float(aucs_b[1] - aucs_b[0])
            if verbose >= 2 and progress_every and (b + 1) % progress_every == 0:
                print(f"[bootstrap] progress: {b+1}/{n_boot}")

        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            raise FloatingPointError("Bootstrap failed: no valid resamples.")

        mean = float(np.mean(diffs))
        std = float(np.std(diffs, ddof=1))
        # Treat near-zero std as degeneracy to avoid gigantic finite z
        eps = 1e-12
        if std < eps:
            z_boot = np.inf * np.sign(mean) if mean != 0.0 else 0.0
        else:
            z_boot = mean / std
        # Symmetric two-sided p
        p_boot = float(2 * min((diffs <= 0).mean(), (diffs > 0).mean()))
        return z_boot, p_boot, diffs.size

    # ---------- try DeLong first ----------
    if verbose >= 1:
        log(1, "Step 1) Attempt standard DeLong.")
    auc_A, auc_B, cov, var_diff = delong_core(pA, pB)
    diff = auc_B - auc_A

    info: Dict[str, Any] = {
        "method": "delong",
        "var_diff": var_diff,
        "tie_rate_A": trA,
        "tie_rate_B": trB,
        "n_pos": m,
        "n_neg": n,
        "n_boot": 0,
        "messages": logs,
    }

    if verbose >= 2:
        log(2, f"AUC_A = {auc_A:.6f}, AUC_B = {auc_B:.6f}, diff(B-A) = {diff:.6f}")
        log(2, f"var_diff = {var_diff:.6e}")

    if not np.isfinite(var_diff) or var_diff <= 0.0:
        # Fallback to bootstrap
        if verbose >= 1:
            log(1, "Degeneracy detected (non-positive or invalid var_diff).")
            log(1, f"Step 2) Fallback to bootstrap (n_boot={n_boot}, seed={random_state}).")
        z_boot, p_boot, n_eff = bootstrap_diff(y_true, pA, pB, n_boot=n_boot)
        info.update({
            "method": "bootstrap",
            "n_boot": int(n_eff),
            "var_diff": None,
        })
        # Per-model AUC CIs via DeLong diagonals (optionally could bootstrap per-AUC)
        ci_A = auc_ci(auc_A, float(max(0.0, cov[0, 0])), alpha)
        ci_B = auc_ci(auc_B, float(max(0.0, cov[1, 1])), alpha)
        if verbose >= 1:
            log(1, f"[bootstrap] z = {z_boot:.6f}, p = {p_boot:.3e}, effective resamples = {n_eff}/{n_boot}")
        if verbose >= 2:
            log(2, f"CI_A = {ci_A}, CI_B = {ci_B}")

        # -------- assemble return by flags (compatibility) --------
        if not return_ci and not return_auc:
            return float(z_boot), float(p_boot)
        if return_ci and not return_auc:
            return float(z_boot), float(p_boot), ci_A, ci_B
        if (not return_ci) and return_auc:
            return float(z_boot), float(p_boot), float(auc_A), float(auc_B)
        return float(z_boot), float(p_boot), ci_A, ci_B, float(auc_A), float(auc_B), info

    # ---------- standard DeLong path ----------
    z = float(diff / math.sqrt(var_diff))
    p_value = float(scipy.stats.norm.sf(abs(z)) * 2.0)
    ci_A = auc_ci(auc_A, float(cov[0, 0]), alpha)
    ci_B = auc_ci(auc_B, float(cov[1, 1]), alpha)

    if verbose >= 1:
        log(1, f"[delong] z = {z:.6f}, p = {p_value:.3e}")
    if verbose >= 2:
        log(2, f"CI_A = {ci_A}, CI_B = {ci_B}")

    # -------- assemble return by flags (compatibility) --------
    if not return_ci and not return_auc:
        return z, p_value
    if return_ci and not return_auc:
        return z, p_value, ci_A, ci_B
    if (not return_ci) and return_auc:
        return z, p_value, float(auc_A), float(auc_B)
    return z, p_value, ci_A, ci_B, float(auc_A), float(auc_B), info
