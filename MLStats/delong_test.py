import pandas as pd
import numpy as np
import scipy.stats

def Delong_test(ground_truth, predictions_one, predictions_two):
    """
    Perform DeLong's test for comparing the AUCs of two models.

    Parameters
    ----------
    ground_truth : array-like of shape (n_samples,)
        True binary labels in range {0, 1}.
    predictions_one : array-like of shape (n_samples,)
        Predicted probabilities by the first model.
    predictions_two : array-like of shape (n_samples,)
        Predicted probabilities by the second model.

    Returns
    -------
    z_score : float
        The z score from comparing the AUCs of two models.
    p_value : float
        The p value from comparing the AUCs of two models.

    Example
    -------
    >>> ground_truth = [0, 1, 0, 1]
    >>> predictions_one = [0.1, 0.4, 0.35, 0.8]
    >>> predictions_two = [0.2, 0.3, 0.4, 0.7]
    >>> z_score, p_value = delong_test(ground_truth, predictions_one, predictions_two)
    >>> print(f"Z-Score: {z_score}, P-Value: {p_value}")
    """
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        T2[J] = T + 1
        return T2

    def compute_ground_truth_statistics(ground_truth):
        assert np.array_equal(np.unique(ground_truth), [0, 1]), "Ground truth must be binary."
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        return order, label_1_count

    order, label_1_count = compute_ground_truth_statistics(np.array(ground_truth))
    predictions_sorted_transposed = np.vstack((np.array(predictions_one), np.array(predictions_two)))[:, order]

    # Fast DeLong computation starts here
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    tx, ty, tz = [np.empty([k, size], dtype=np.float64) for size in [m, n, m + n]]
    for r in range(k):
        positive_examples = predictions_sorted_transposed[r, :m]
        negative_examples = predictions_sorted_transposed[r, m:]
        tx[r, :], ty[r, :], tz[r, :] = [compute_midrank(examples) for examples in [positive_examples, negative_examples, predictions_sorted_transposed[r, :]]]

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    # Calculating p-value
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, delongcov), l.T)).flatten()
    p_value = scipy.stats.norm.sf(abs(z)) * 2

    z_score = -z[0].item()
    p_value = p_value[0].item()

    return z_score, p_value
