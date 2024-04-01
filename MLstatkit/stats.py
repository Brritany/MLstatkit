import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def Delong_test(true, prob_A, prob_B):
    """
    Perform DeLong's test for comparing the AUCs of two models.

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        True binary labels in range {0, 1}.
    prob_A : array-like of shape (n_samples,)
        Predicted probabilities by the first model.
    prob_B : array-like of shape (n_samples,)
        Predicted probabilities by the second model.

    Returns
    -------
    z_score : float
        The z score from comparing the AUCs of two models.
    p_value : float
        The p value from comparing the AUCs of two models.

    Example
    -------
    >>> true = [0, 1, 0, 1]
    >>> prob_A = [0.1, 0.4, 0.35, 0.8]
    >>> prob_B = [0.2, 0.3, 0.4, 0.7]
    >>> z_score, p_value = Delong_test(true, prob_A, prob_B)
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

    def compute_ground_truth_statistics(true):
        assert np.array_equal(np.unique(true), [0, 1]), "Ground truth must be binary."
        order = (-true).argsort()
        label_1_count = int(true.sum())
        return order, label_1_count

    order, label_1_count = compute_ground_truth_statistics(np.array(true))
    sorted_probs = np.vstack((np.array(prob_A), np.array(prob_B)))[:, order]

    # Fast DeLong computation starts here
    m = label_1_count
    n = sorted_probs.shape[1] - m
    k = sorted_probs.shape[0]
    tx, ty, tz = [np.empty([k, size], dtype=np.float64) for size in [m, n, m + n]]
    for r in range(k):
        positive_examples = sorted_probs[r, :m]
        negative_examples = sorted_probs[r, m:]
        tx[r, :], ty[r, :], tz[r, :] = [
            compute_midrank(examples) for examples in [positive_examples, negative_examples, sorted_probs[r, :]]
        ]

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


def Bootstrapping(y_true, y_prob, score_func_str, n_bootstraps=1000, confidence_level=0.95, threshold=0.5, average='macro'):
    """
    Calculate confidence intervals for specified performance metrics using bootstrapping,
    and returns the original score along with the confidence interval. Supports 'auroc', 'auprc', and 'f1'.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels in range {0, 1}.
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities or binary predictions depending on the score function.
    score_func_str : str
        Scoring function identifier: 'auroc', 'auprc', or 'f1'.
    n_bootstraps : int
        Number of bootstrapping samples to use.
    confidence_level : float
        The confidence interval level (e.g., 0.95 for 95% confidence interval).
    threshold : float
        Threshold to convert probabilities to binary labels for 'f1' scoring function.
    average : str, optional
        This parameter is required for multiclass/multilabel targets. 
        If None, the scores for each class are returned. Otherwise, this 
        determines the type of averaging performed on the data.

    Returns:
    --------
    original_score : float
        The original score calculated without bootstrapping.
    confidence_lower : float
        The lower bound of the confidence interval.
    confidence_upper : float
        The upper bound of the confidence interval.
    
    Example
    -------
    >>> y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])
    >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.3, 0.4, 0.7, 0.05])
    >>> for score in ['auroc', 'auprc', 'f1']:
        >>> original_score, conf_lower, conf_upper = Bootstrapping(y_true, y_prob, score, threshold=0.5)
        >>> print(f"{score.upper()} original score: {original_score:.3f}, confidence interval: [{conf_lower:.3f} - {conf_upper:.3f}]")   
    """
    if score_func_str == 'auroc':
        original_score = roc_auc_score(y_true, y_prob, average=average)
    elif score_func_str == 'auprc':
        original_score = average_precision_score(y_true, y_prob, average=average)
    elif score_func_str == 'f1':
        y_pred = (y_prob >= threshold).astype(int)
        original_score = f1_score(y_true, y_pred, average=average)
    else:
        raise ValueError(f"Unsupported score function: {score_func_str}")
    
    bootstrapped_scores = []
    rng = np.random.RandomState(42)  # Control reproducibility
    
    for _ in range(n_bootstraps):
        # Randomly sample with replacement
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip this loop iteration if sample is not valid
        
        if score_func_str == 'auroc':
            score = roc_auc_score(y_true[indices], y_prob[indices], average=average)
        elif score_func_str == 'auprc':
            score = average_precision_score(y_true[indices], y_prob[indices], average=average)
        elif score_func_str == 'f1':
            y_pred = (y_prob[indices] >= threshold).astype(int)
            score = f1_score(y_true[indices], y_pred, average=average)
        
        bootstrapped_scores.append(score)
    
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2.0 * 100
    upper_percentile = 100 - (alpha / 2.0 * 100)
    confidence_lower = np.percentile(bootstrapped_scores, lower_percentile)
    confidence_upper = np.percentile(bootstrapped_scores, upper_percentile)
    
    return original_score, confidence_lower, confidence_upper
