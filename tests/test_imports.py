# tests/test_imports.py
def test_public_imports_root():
    from MLstatkit import (
        get_metric_fn, Delong_test, Bootstrapping, Permutation_test, AUC2OR
    )

def test_public_imports_stats_shim():
    from MLstatkit.stats import (
        get_metric_fn, Delong_test, Bootstrapping, Permutation_test, AUC2OR
    )
