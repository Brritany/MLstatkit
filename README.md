# MLstatkit

![PyPI - Version](https://img.shields.io/pypi/v/MLstatkit)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Brritany/MLstatkit)
![PyPI - Status](https://img.shields.io/pypi/status/MLstatkit)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/MLstatkit)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/MLstatkit)
![PyPI - Download](https://img.shields.io/pypi/dm/MLstatkit)
[![Downloads](https://static.pepy.tech/badge/MLstatkit)](https://pepy.tech/project/MLstatkit)

**MLstatkit** is a Python library that integrates established statistical methods into modern machine learning workflows.  
It provides a set of core functions widely used for model evaluation and statistical inference:

- **DeLong's test** (`Delong_test`) for comparing the AUCs of two correlated ROC curves.  

- **Bootstrapping** (`Bootstrapping`) for estimating confidence intervals of metrics such as ROC-AUC, F1-score, accuracy, precision, recall, and PR-AUC.  

- **Permutation test** (`Permutation_test`) for evaluating whether performance differences between two models are statistically significant.  

- **AUC to Odds Ratio conversion** (`AUC2OR`) for interpreting ROC-AUC values in terms of odds ratios and related effect size statistics.  

Since v0.1.9, the library has been **modularized** into dedicated files (`ci.py`, `conversions.py`, `delong.py`, `metrics.py`, `permutation.py`), while keeping a unified import interface through `stats.py`. This improves readability, maintainability, and extensibility for future methods.

## Installation

Install MLstatkit directly from PyPI using pip:

```bash
pip install MLstatkit
```

## Usage

### Delong's Test for ROC Curve

`Delong_test` function enables a statistical evaluation of the differences between the **areas under two correlated Receiver Operating Characteristic (ROC) curves derived from distinct models**. This facilitates a deeper understanding of comparative model performance.  
Since version `0.1.8`, the function also supports returning **confidence intervals (CIs)** for the AUCs of both models, similar to the functionality of `roc.test` in R.

#### Parameters (DeLong’s Test)

- **true** : array-like of shape (n_samples,)
    Binary ground truth labels in {0, 1}.

- **prob_A**, **prob_B** : array-like of shape (n_samples,)
    Scores or probabilities for the positive class from models A and B.

- **alpha** : float, default=0.95
    Confidence level for the AUC confidence intervals (normal approximation, clipped to [0, 1]).

- **return_ci** : bool, default=True
    If `True`, return `(ci_A, ci_B)` for model A and B AUCs.

- **return_auc** : bool, default=True
    If `True`, include `(auc_A, auc_B)` in the return tuple in addition to `z` and `p` (and optionally `(ci_A, ci_B)` if `return_ci=True`).

- **n_boot** : int, default=5000
    Number of bootstrap resamples used if the fallback path is triggered.

- **random_state** : int or None, default=None
    RNG seed for reproducibility in the bootstrap path (not used for standard DeLong).

- **verbose** : {0, 1, 2}, default=0
  - `0`: Silent
  - `1`: Key steps (sample counts, method, z/p, CIs)
  - `2`: Detailed (includes var_diff, raw differences, and optional bootstrap progress)

- **progress_every** : int, default=0
    When `verbose >= 2`, print bootstrap progress every N iterations; `0` disables progress output.

#### Returns (DeLong's test)

Depending on `return_ci` and `return_auc`, the function returns different tuples:

- `return_ci=False`, `return_auc=False` → `(z, p_value)`
- `return_ci=True`, `return_auc=False` → `(z, p_value, ci_A, ci_B)`
- `return_ci=False`, `return_auc=True` → `(z, p_value, auc_A, auc_B)`
- `return_ci=True`, `return_auc=True` → `(z, p_value, ci_A, ci_B, auc_A, auc_B, info)`

Where **info** (present only when both `return_ci` and `return_auc` are `True`) is a `dict` containing:

- `method`: `"delong"` or `"bootstrap"`
- `var_diff`: variance of the AUC difference (if DeLong was used)
- `tie_rate_A`, `tie_rate_B`: tie proportions in model A and B scores
- `n_pos`, `n_neg`: class counts
- `n_boot`: number of effective bootstrap samples (if bootstrap)
- `messages`: list of `(level, message)` logs captured during the run

#### Example (DeLong’s Test)

- Example 1 --- Minimal usage (z and p only)

``` python
from MLstatkit import Delong_test
import numpy as np

true   = np.array([0, 1, 0, 1])
prob_A = np.array([0.10, 0.40, 0.35, 0.80])
prob_B = np.array([0.20, 0.30, 0.40, 0.70])

z, p = Delong_test(true, prob_A, prob_B, return_ci=False, return_auc=False, verbose=0)
print(f"z = {z:.6f}, p = {p:.3e}")
```

- Example 2 --- AUCs and 95% CIs (with method info)

``` python
z, p, ci_A, ci_B, auc_A, auc_B, info = Delong_test(
    true, prob_A, prob_B,
    alpha=0.95, return_ci=True, return_auc=True, verbose=1
)

print(f"Method   : {info['method']}")
print(f"AUC_A    : {auc_A:.4f}, CI_A = {ci_A}")
print(f"AUC_B    : {auc_B:.4f}, CI_B = {ci_B}")
print(f"z-score  : {z:.4f}, p-value = {p:.3e}")
```

- Example 3 — Degenerate case (forces bootstrap fallback)

```python
# Perfect separation for A, completely reversed scores for B
true   = np.array([0, 1] * 50)
prob_A = true.astype(float)        # Model A: perfect AUC = 1.0
prob_B = 1 - true.astype(float)    # Model B: worst AUC = 0.0

z, p, ci_A, ci_B, auc_A, auc_B, info = Delong_test(
    true, prob_A, prob_B,
    alpha=0.95, return_ci=True, return_auc=True,
    n_boot=2000, random_state=42, verbose=2, progress_every=500
)

print("--- Bootstrap fallback example ---")
print(f"Method   : {info['method']} (auto-fallback expected)")
print(f"AUC_A    : {auc_A:.4f}, CI_A = {ci_A}")
print(f"AUC_B    : {auc_B:.4f}, CI_B = {ci_B}")
print(f"z-score  : {z}, p-value = {p:.3e}")
```

### Bootstrapping for Confidence Intervals

The `Bootstrapping` function calculates **confidence intervals (CIs)** for specified performance metrics using bootstrapping, providing a measure of the estimation's reliability. It supports calculation for AUROC (area under the ROC curve), AUPRC (area under the precision-recall curve), and F1 score metrics.

#### Parameters（Bootstrapping）

- **true** : array-like of shape (n_samples,)  
    True binary labels, where the labels are either {0, 1}.

- **prob** : array-like of shape (n_samples,)  
    Predicted probabilities, as returned by a classifier's predict_proba method, or binary predictions based on the specified scoring function and threshold.

- **metric_str** : str, default='f1'  
    Identifier for the scoring function to use. Supported values include 'f1', 'accuracy', 'recall', 'precision', 'roc_auc', 'pr_auc', and 'average_precision'.

- **n_bootstraps** : int, default=1000  
    The number of bootstrap iterations to perform. Increasing this number improves the reliability of the confidence interval estimation but also increases computational time.

- **confidence_level** : float, default=0.95  
    The confidence level for the interval estimation. For instance, 0.95 represents a 95% confidence interval.

- **threshold** : float, default=0.5  
    A threshold value used for converting probabilities to binary labels for metrics like 'f1', where applicable.

- **average** : str, default='macro'  
    Specifies the method of averaging to apply to multi-class/multi-label targets. Other options include 'micro', 'samples', 'weighted', and 'binary'.

- **random_state** : int, default=0  
    Seed for the random number generator. This parameter ensures reproducibility of results.

#### Returns（Bootstrapping）

- **original_score** : float  
    Metric score on the original (non-resampled) dataset.

- **confidence_lower** : float  
    Lower bound of the bootstrap confidence interval.

- **confidence_upper** : float  
    Upper bound of the bootstrap confidence interval.

#### Examples（Bootstrapping）

```python
from MLstatkit import Bootstrapping

# Example data
y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])
y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.3, 0.4, 0.7, 0.05])

# Calculate confidence intervals for AUROC
original_score, confidence_lower, confidence_upper = Bootstrapping(y_true, y_prob, 'roc_auc')
print(f"AUROC: {original_score:.3f}, Confidence interval: [{confidence_lower:.3f} - {confidence_upper:.3f}]")

# Calculate confidence intervals for AUPRC
original_score, confidence_lower, confidence_upper = Bootstrapping(y_true, y_prob, 'pr_auc')
print(f"AUPRC: {original_score:.3f}, Confidence interval: [{confidence_lower:.3f} - {confidence_upper:.3f}]")

# Calculate confidence intervals for F1 score with a custom threshold
original_score, confidence_lower, confidence_upper = Bootstrapping(y_true, y_prob, 'f1', threshold=0.5)
print(f"F1 Score: {original_score:.3f}, Confidence interval: [{confidence_lower:.3f} - {confidence_upper:.3f}]")

# Loop through multiple metrics
for score in ['roc_auc', 'pr_auc', 'f1']:
    original_score, conf_lower, conf_upper = Bootstrapping(y_true, y_prob, score, threshold=0.5)
    print(f"{score.upper()} original score: {original_score:.3f}, confidence interval: [{conf_lower:.3f} - {conf_upper:.3f}]")
```

### Permutation Test for Statistical Significance

The `Permutation_test` function evaluates whether the observed difference in performance between two models is **statistically significant**.  
It works by randomly shuffling the predictions between the models and recalculating the chosen metric many times to generate a null distribution of differences.  
This approach makes no assumptions about the underlying distribution of the data, making it a robust method for model comparison.

#### Parameters

- **y_true** : array-like of shape (n_samples,)  
  True binary labels in {0, 1}.  

- **prob_model_A** : array-like of shape (n_samples,)  
  Predicted probabilities from the first model.  

- **prob_model_B** : array-like of shape (n_samples,)  
  Predicted probabilities from the second model.  

- **metric_str** : str, default=`'f1'`  
  Metric to compare. Supported: `'f1'`, `'accuracy'`, `'recall'`, `'precision'`, `'roc_auc'`, `'pr_auc'`, `'average_precision'`.  

- **n_bootstraps** : int, default=`1000`  
  Number of permutation samples to generate.  

- **threshold** : float, default=`0.5`  
  Threshold for converting probabilities into binary predictions (used for metrics such as F1, precision, recall).  

- **average** : str, default=`'macro'`  
  Averaging strategy for multi-class/multi-label tasks. Options: `'binary'`, `'micro'`, `'macro'`, `'weighted'`, `'samples'`.  

- **random_state** : int, default=`0`  
  Random seed for reproducibility.  

#### Returns

- **metric_a** : float  
  Metric value for model A on the original data.  

- **metric_b** : float  
  Metric value for model B on the original data.  

- **p_value** : float  
  The p-value from the permutation test, i.e., the probability of observing a difference as extreme as the actual one under the null hypothesis.  

- **benchmark** : float  
  The observed absolute difference between the metrics of model A and model B.  

- **samples_mean** : float  
  Mean of the metric differences from permutation samples.  

- **samples_std** : float  
  Standard deviation of the metric differences from permutation samples.  

#### Example

```python
import numpy as np
from MLstatkit import Permutation_test

y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])
prob_model_A = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.3, 0.4, 0.7, 0.05])
prob_model_B = np.array([0.2, 0.3, 0.25, 0.85, 0.15, 0.35, 0.45, 0.65, 0.01])

# Compare models using a permutation test on F1 score
metric_a, metric_b, p_value, benchmark, samples_mean, samples_std = Permutation_test(
    y_true, prob_model_A, prob_model_B, metric_str='f1'
)

print(f"F1 Score Model A: {metric_a:.5f}, Model B: {metric_b:.5f}")
print(f"Observed Difference: {benchmark:.5f}, p-value: {p_value:.5f}")
print(f"Permutation Samples Mean: {samples_mean:.5f}, Std: {samples_std:.5f}")
```

### Conversion of AUC to Odds Ratio (OR)

The `AUC2OR` function converts an **Area Under the ROC Curve (AUC)** value into an **Odds Ratio (OR)** under the binormal model.  
This transformation helps interpret classification performance in terms of effect sizes commonly used in statistics.  

- Under the binormal model:  

$$
AUC = \Phi\left(\frac{d}{\sqrt{2}}\right), \quad d \text{ is Cohen's } d
$$

$$
\ln(OR) = \frac{\pi}{\sqrt{3}} \times d
$$

Since version `0.1.9`, `AUC2OR` uses the exact **inverse normal CDF** (`scipy.stats.norm.ppf`) to compute \(z = \Phi^{-1}(AUC)\), improving accuracy over older approximations.

#### Parameters （AUC to OR）

- **AUC** : float  
  Area Under the ROC Curve, must be in (0, 1).  

- **return_all** : bool, default=`False`  
  If True, returns intermediate values `(z, d, ln_or, OR)` in addition to OR:  
  - **z** : probit (inverse normal CDF of AUC)  
  - **d** : effect size, `sqrt(2) * z`  
  - **ln_or** : natural logarithm of the Odds Ratio  
  - **OR** : Odds Ratio  

#### Returns （AUC to OR）

- **OR** : float  
  Odds Ratio corresponding to the given AUC.  

- **(z, d, ln_or, OR)** if `return_all=True`.  

#### Example （AUC to OR）

```python
from MLstatkit import AUC2OR

auc = 0.7  # Example AUC value

# Convert AUC to OR and retrieve intermediate values
z, d, ln_or, OR = AUC2OR(auc, return_all=True)
print(f"z: {z:.5f}, d: {d:.5f}, ln_OR: {ln_or:.5f}, OR: {OR:.5f}")

# Convert AUC to OR without intermediate values
OR = AUC2OR(auc)
print(f"OR: {OR:.5f}")
```

## References

### Delong's Test

The implementation of `Delong_test` in MLstatkit is based on the following publication:

- Xu Sun and Weichao Xu, "Fast implementation of DeLong’s algorithm for comparing the areas under correlated receiver operating characteristic curves," in *IEEE Signal Processing Letters*, vol. 21, no. 11, pp. 1389-1393, 2014, IEEE.

### Bootstrapping

The `Bootstrapping` method for calculating confidence intervals does not directly reference a single publication but is a widely accepted statistical technique for estimating the distribution of a metric by resampling with replacement. For a comprehensive overview of bootstrapping methods, see:

- B. Efron and R. Tibshirani, "An Introduction to the Bootstrap," Chapman & Hall/CRC Monographs on Statistics & Applied Probability, 1994.

### Permutation Test

The `Permutation_test` are utilized to assess the significance of the difference in performance metrics between two models by randomly reallocating observations to groups and computing the metric. This approach does not make specific distributional assumptions, making it versatile for various data types. For a foundational discussion on permutation tests, refer to:

- P. Good, "Permutation Tests: A Practical Guide to Resampling Methods for Testing Hypotheses," Springer Series in Statistics, 2000.

These references lay the groundwork for the statistical tests and methodologies implemented in MLstatkit, providing users with a deep understanding of their scientific basis and applicability.

### AUC2OR

The `AUC2OR` function converts the Area Under the Receiver Operating Characteristic Curve (AUC) into an **Odds Ratio (OR)** under the binormal model.  
When `return_all=True`, it also provides intermediate values:

- **z** : probit (Φ⁻¹ of AUC)  
- **d** : Cohen’s d effect size (`sqrt(2) * z`)  
- **ln_or** : natural logarithm of the odds ratio  
- **OR** : odds ratio  

This conversion is useful for interpreting ROC-AUC values in terms of effect sizes commonly used in statistical research.

- Salgado, J. F. (2018). *Transforming the area under the normal curve (AUC) into Cohen’s d, Pearson’s rpb, odds-ratio, and natural log odds-ratio: Two conversion tables.* European Journal of Psychology Applied to Legal Context, 10(1), 35–47.

## Contributing

We welcome contributions to MLstatkit! Please see our contribution guidelines for more details.

## License

MLstatkit is distributed under the MIT License. For more information, see the LICENSE file in the GitHub repository.

### Update log

- `0.1.91`
  - DeLong’s Test: fixed z-score sign convention; added automatic bootstrap fallback when variance degenerates.
  - Added verbose levels (0–2) and progress_every for bootstrap monitoring.
  - Expanded return options with rich info metadata.
  - Docs & tests: updated README (new examples, clearer parameters/returns) and extended unit tests; all passed.
- `0.1.9`  
  - **Refactor & modularization**: split `stats.py` into multiple modules (`ci.py`, `conversions.py`, `delong.py`, `metrics.py`, `permutation.py`) for better maintainability, while preserving a unified import interface.  
  - **Functions restored**: `Bootstrapping`, `Permutation_test`, and `AUC2OR` now available again after refactor.  
  - **AUC2OR** updated to use binormal model with exact `norm.ppf`, improving accuracy over the earlier polynomial approximation. Supports `return_all=True` to retrieve intermediate values `(z, d, ln_or, OR)`.  
  - **Improved testing**: added dedicated `tests/` for all core functions (Delong, Bootstrapping, Permutation test, AUC2OR, metrics, imports). Achieved full test coverage (`pytest` 16 passed).  
  - **README.md** updated with revised usage examples and clearer documentation.  
- `0.1.8`   Add return_ci option to Delong_test for AUC confidence intervals. Add `pyproject.toml`.
- `0.1.7`   Update `README.md`
- `0.1.6`   Debug.
- `0.1.5`   Update `README.md`, Add `AUC2OR` function.
- `0.1.4`   Update `README.md`, Add `Permutation_tests` function, Re-do `Bootstrapping` Parameters.
- `0.1.3`   Update `README.md`.
- `0.1.2`   Add `Bootstrapping` operation process progress display.
- `0.1.1`   Update `README.md`, `setup.py`. Add `CONTRIBUTING.md`.
- `0.1.0`   First edition.
