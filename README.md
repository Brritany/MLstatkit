# MLStats

MLStats is a comprehensive Python library designed to seamlessly integrate established statistical methods into machine learning projects. It encompasses a variety of tools, including Delong's test for comparing AUCs and bootstrapping for calculating confidence intervals, among others. With its modular design, MLStats offers researchers and data scientists a flexible and powerful toolkit to augment their analyses and model evaluations, catering to a broad spectrum of statistical testing needs within the domain of machine learning.

## Installation

Install MLStats directly from TestPyPI using pip:

```bash
pip install -i https://test.pypi.org/simple/ MLStats
```

## Usage

### Delong's Test

`Delong_test` function allows for statistical comparison of AUCs from two different models, providing insights into their performance differences.

#### Parameters:
- **true** : array-like of shape (n_samples,)  
    True binary labels in range {0, 1}.

- **prob_A** : array-like of shape (n_samples,)  
    Predicted probabilities by the first model.

- **prob_B** : array-like of shape (n_samples,)  
    Predicted probabilities by the second model.

#### Returns:
- **z_score** : float  
    The z score from comparing the AUCs of two models.

- **p_value** : float  
    The p value from comparing the AUCs of two models.

#### Example:

```python
from MLStats.stats import Delong_test

# Example data
true = np.array([0, 1, 0, 1])
prob_A = np.array([0.1, 0.4, 0.35, 0.8])
prob_B = np.array([0.2, 0.3, 0.4, 0.7])

# Perform DeLong's test
z_score, p_value = Delong_test(true, prob_A, prob_B)

print(f"Z-Score: {z_score}, P-Value: {p_value}")
```

This demonstrates the usage of `Delong_test` to statistically compare the AUCs of two models based on their predictions and the ground truth labels. The returned z-score and p-value help in understanding if the difference in model performances is statistically significant.

## Contributing

We welcome contributions to MLStats! Please see our contribution guidelines for more details.

## License

MLStats is distributed under the MIT License. For more information, see the LICENSE file in the GitHub repository.


