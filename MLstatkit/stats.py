"""
Compatibility shim for MLstatkit<=0.1.x users.
Do NOT add new features here. Will be removed in 0.2.0.
"""
import warnings

warnings.warn(
    "Importing from MLstatkit.stats is deprecated and will be removed in v0.2.0. "
    "Please use `from MLstatkit import ...` instead.",
    UserWarning,
    stacklevel=2
)

from .metrics import get_metric_fn
from .ci import Bootstrapping
from .permutation import Permutation_test
from .conversions import AUC2OR
from .delong import Delong_test

__all__ = [
    "get_metric_fn",
    "Bootstrapping",
    "Permutation_test",
    "AUC2OR",
    "Delong_test",
]
