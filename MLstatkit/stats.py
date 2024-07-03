from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import math
import scipy.stats
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score, 
    roc_auc_score, average_precision_score, precision_recall_curve, auc
)

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

def get_metric_fn(metric_str, threshold=0.5, average='macro'):
    """
    Returns a callable metric function based on the specified metric string.

    Parameters:
    -----------
    metric_str : str
        Identifier for the scoring function: 'f1', 'accuracy', 'recall', 'precision', 'roc_auc', 'pr_auc', 'average_precision'.
    threshold : float, default=0.5
        Threshold for converting probabilities to binary labels for certain metrics.
    average : str, default='macro'
        Averaging method for multi-class/multi-label scenarios.

    Returns:
    --------
    function
        A function that computes the specified metric.
    """
    if metric_str == 'f1':
        return lambda y, y_pred: f1_score(y, (y_pred >= threshold).astype(int), average=average)
    elif metric_str == 'accuracy':
        return lambda y, y_pred: accuracy_score(y, (y_pred >= threshold).astype(int))
    elif metric_str == 'recall':
        return lambda y, y_pred: recall_score(y, (y_pred >= threshold).astype(int), average=average)
    elif metric_str == 'precision':
        return lambda y, y_pred: precision_score(y, (y_pred >= threshold).astype(int), average=average)
    elif metric_str == 'roc_auc':
        return roc_auc_score
    elif metric_str == 'pr_auc':
        # Directly using precision_recall_curve and auc to calculate PR AUC
        return lambda y, y_pred: auc(*precision_recall_curve(y, y_pred)[1::-1])
    elif metric_str == 'average_precision':
        return lambda y, y_pred: average_precision_score(y, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric_str}")

def Bootstrapping(y_true, y_prob, metric_str='f1', n_bootstraps=1000, 
                  confidence_level=0.95, threshold=0.5, average='macro', random_state=0):
    """
    Calculate confidence intervals for specified performance metrics using bootstrapping,
    returning the original score along with the confidence interval. Supports 'f1', 'accuracy',
    'recall', 'precision', 'roc_auc', and 'pr_auc'.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities or labels, depending on the score function.
    metric_str : str, default='f1'
        Identifier for the scoring function: 'f1', 'accuracy', 'recall', 'precision', 'roc_auc', 'pr_auc', 'average_precision'.
    n_bootstraps : int, default=1000
        Number of bootstrapping samples to use.
    confidence_level : float, default=0.95
        The confidence interval level (e.g., 0.95 for 95% confidence interval).
    random_state : int, default=0
        Seed for the random number generator for reproducibility.
    threshold : float, default=0.5
        Threshold for converting probabilities to binary labels for 'f1' scoring function.
    average : str, default='macro'
        Determines the type of averaging performed on the data for multi-class/multi-label targets.

    Returns:
    --------
    original_score : float
        The original score calculated without bootstrapping.
    confidence_lower : float
        The lower bound of the confidence interval.
    confidence_upper : float
        The upper bound of the confidence interval.
    """
    assert y_true.shape[0] == y_prob.shape[0]
    
    rng = np.random.RandomState(random_state)  # Use RandomState for reproducibility
    
    metric_fn = get_metric_fn(metric_str, threshold, average)

    original_score = metric_fn(y_true, y_prob)

    bootstrapped_scores = []
    for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {metric_str}"):
        indices = rng.randint(0, len(y_true), len(y_true))
        if np.unique(y_true[indices]).size < 2:
            continue  # Ensure both classes are present
        score = metric_fn(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
    
    # Calculate confidence intervals
    confidence_lower = np.percentile(bootstrapped_scores, (1 - confidence_level) / 2 * 100)
    confidence_upper = np.percentile(bootstrapped_scores, (1 + confidence_level) / 2 * 100)
    
    return original_score, confidence_lower, confidence_upper

def Permutation_test(y_true, prob_model_A, prob_model_B, metric_str='f1', n_bootstraps=1000, 
                     threshold=0.5, average='macro', random_state=0):
    """
    Conducts a permutation test to determine if there is a statistically significant 
    difference between the metrics of two models.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    prob_model_A : array-like of shape (n_samples,)
        Predicted probabilities from the first model.
    prob_model_B : array-like of shape (n_samples,)
        Predicted probabilities from the second model.
    metric_str : str, default='f1'
        The metric for comparison ('f1', 'accuracy', 'recall', 'precision', 'roc_auc', 'pr_auc', 'average_precision').
    n_bootstraps : int, default=1000
        The number of bootstrap samples used for the test.
    random_state : int, default=0
        Controls the randomness of the permutation samples.
    threshold : float, default=0.5
        The threshold for converting probabilities to binary labels.
    average : str, default='macro'
        The averaging method for the metric calculation.
    
    Returns:
    --------
    metric_a, metric_b, pvalue, benchmark, samples_mean, samples_std : float
        The metric values for models A and B, the p-value of the test, the observed metric difference,
        and the mean and standard deviation of the differences in the permutation samples.
    """
    assert y_true.shape[0] == prob_model_A.shape[0] == prob_model_B.shape[0]

    np.random.seed(random_state)  # For reproducibility
    metric_fn = get_metric_fn(metric_str, threshold, average)

    # Calculate original metrics
    metric_a = metric_fn(y_true, prob_model_A)
    metric_b = metric_fn(y_true, prob_model_B)
    benchmark = np.abs(metric_a - metric_b)  # The observed difference

    # Initialize array for storing metric differences from permutations
    samples = np.zeros(n_bootstraps)

    for i in tqdm(range(n_bootstraps), desc=f"Computing {metric_str} Permutation Test p-value"):
        # Randomly shuffle the predictions between models A and B
        msk = np.random.rand(len(y_true)) < 0.5
        y_pred_a_perm = np.where(msk, prob_model_A, prob_model_B)
        y_pred_b_perm = np.where(msk, prob_model_B, prob_model_A)

        # Calculate metrics for the permuted predictions
        sample_metric_a = metric_fn(y_true, y_pred_a_perm)
        sample_metric_b = metric_fn(y_true, y_pred_b_perm)

        # Store the absolute difference in metrics for this permutation
        samples[i] = np.abs(sample_metric_a - sample_metric_b)

    # Calculate p-value: proportion of permuted differences greater than or equal to observed
    p_value = np.mean(samples >= benchmark)

    # Optionally calculate mean and standard deviation of permuted differences
    samples_mean = np.mean(samples)
    samples_std = np.std(samples)

    return metric_a, metric_b, p_value, benchmark, samples_mean, samples_std

def AUC2OR(AUC, return_all=False):
    """
    Converts Area Under the Curve (AUC) to Odds Ratio (OR) and optionally returns intermediate values.
    
    Parameters:
    -----------
    AUC : float
        The Area Under the Curve (AUC) value to be converted.
    return_all : bool, default=False
        If True, returns intermediate values t, z, d, and ln_OR in addition to OR.
    
    Returns:
    --------
    OR : float
        The calculated Odds Ratio (OR) from the given AUC value.
    t : float, optional
        Intermediate value calculated from AUC.
    z : float, optional
        Intermediate value calculated from t.
    d : float, optional
        Intermediate value calculated from z.
    ln_OR : float, optional
        The natural logarithm of the Odds Ratio.
    
    Notes:
    ------
    The function calculates several intermediate values:
    - t : derived from the AUC using a logarithmic transformation.
    - z : calculated from t using a polynomial approximation.
    - d : a scaling of z.
    - ln_OR : the natural logarithm of OR, derived from d.
    """
    
    def calculate_t(AUC):
        return math.sqrt(math.log(1 / ((1 - AUC) ** 2)))

    def calculate_z(AUC):
        t = calculate_t(AUC)
        numerator = 2.515517 + 0.802853 * t + 0.0103328 * (t ** 2)
        denominator = 1 + 1.432788 * t + 0.189269 * (t ** 2) + 0.001308 * (t ** 3)
        z = t - (numerator / denominator)
        return z

    def calculate_d(AUC):
        z = calculate_z(AUC)
        d = z * math.sqrt(2)
        return d

    t = calculate_t(AUC)
    z = calculate_z(AUC)
    d = calculate_d(AUC)
    ln_OR = (math.pi * d) / math.sqrt(3)
    OR = math.exp(ln_OR)
    
    if return_all:
        return t, z, d, ln_OR, OR
    else:
        return OR
