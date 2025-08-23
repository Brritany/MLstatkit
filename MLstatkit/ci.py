# MLstatkit/ci.py
from typing import Tuple
import numpy as np
from .metrics import get_metric_fn

try:
    from tqdm import tqdm
except Exception:  # 讓 tqdm 非強制
    def tqdm(x, **kwargs):
        return x

def Bootstrapping(
    y_true,
    y_prob,
    metric_str: str = "f1",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    threshold: float = 0.5,
    average: str = "macro",
    random_state: int = 0
) -> Tuple[float, float, float]:
    """
    Return (original_score, ci_lower, ci_upper).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    assert y_true.shape[0] == y_prob.shape[0]

    rng = np.random.RandomState(random_state)
    metric_fn = get_metric_fn(metric_str, threshold, average)

    original_score = metric_fn(y_true, y_prob)

    scores = []
    n = len(y_true)
    for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {metric_str}"):
        idx = rng.randint(0, n, n)
        if np.unique(y_true[idx]).size < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_prob[idx]))

    if len(scores) == 0:
        raise RuntimeError("All bootstrap samples were degenerate (single class).")

    alpha = (1 - confidence_level) / 2.0
    lo = float(np.percentile(scores, 100 * alpha))
    hi = float(np.percentile(scores, 100 * (1 - alpha)))
    return float(original_score), lo, hi
