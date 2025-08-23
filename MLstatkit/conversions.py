# MLstatkit/conversions.py
import math
from typing import Tuple
from scipy.stats import norm

def AUC2OR(AUC: float, return_all: bool = False):
    """
    Convert AUC to Odds Ratio under the binormal model:
      AUC = Phi(d / sqrt(2))  =>  d = sqrt(2) * Phi^{-1}(AUC)
      ln(OR) = (pi / sqrt(3)) * d

    Parameters
    ----------
    AUC : float
        Area Under the ROC Curve in (0, 1).
    return_all : bool, default False
        If True, return (z, d, ln_or, OR) where:
          z = Phi^{-1}(AUC)
          d = sqrt(2) * z
          ln_or = (pi / sqrt(3)) * d
          OR = exp(ln_or)

    Returns
    -------
    OR : float
        Odds ratio corresponding to the given AUC.
    (z, d, ln_or, OR) if return_all=True
    """
    # guard and clip to avoid inf at exact 0 or 1 due to ppf
    eps = 1e-12
    a = float(AUC)
    if not (0.0 < a < 1.0):
        # clip to open interval (0,1)
        a = min(max(a, eps), 1.0 - eps)

    z = norm.ppf(a)                 # exact inverse CDF
    d = math.sqrt(2.0) * z
    ln_or = (math.pi / math.sqrt(3.0)) * d
    OR = math.exp(ln_or)

    if return_all:
        return z, d, ln_or, OR
    return OR
