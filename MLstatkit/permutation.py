# MLstatkit/permutation.py
from typing import Tuple
import numpy as np
from .metrics import get_metric_fn

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def Permutation_test(
    y_true,
    prob_model_A,
    prob_model_B,
    metric_str: str = "f1",
    n_bootstraps: int = 1000,
    threshold: float = 0.5,
    average: str = "macro",
    random_state: int = 0
) -> Tuple[float, float, float, float, float, float]:
    """
    Return: metric_a, metric_b, p_value, benchmark, samples_mean, samples_std
    """
    y_true = np.asarray(y_true)
    a = np.asarray(prob_model_A)
    b = np.asarray(prob_model_B)
    assert y_true.shape[0] == a.shape[0] == b.shape[0]

    rng = np.random.RandomState(random_state)
    metric_fn = get_metric_fn(metric_str, threshold, average)

    metric_a = metric_fn(y_true, a)
    metric_b = metric_fn(y_true, b)
    benchmark = abs(metric_a - metric_b)

    samples = np.zeros(n_bootstraps, dtype=float)
    for i in tqdm(range(n_bootstraps), desc=f"Computing {metric_str} Permutation Test p-value"):
        msk = rng.rand(len(y_true)) < 0.5
        a_perm = np.where(msk, a, b)
        b_perm = np.where(msk, b, a)
        samples[i] = abs(metric_fn(y_true, a_perm) - metric_fn(y_true, b_perm))

    p_value = float(np.mean(samples >= benchmark))
    return float(metric_a), float(metric_b), p_value, float(benchmark), float(samples.mean()), float(samples.std(ddof=0))
