"""
frequentist.py: Classical Frequentist estimators and confidence intervals for comparison.
"""

import numpy as np
from src.exact_posterior import log_comb


def umvue(m, k):
    """
    Uniformly Minimum Variance Unbiased Estimator (UMVUE):
    N_hat = m * (1 + 1/k) - 1
    """
    m = np.asarray(m, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    return m * (1.0 + 1.0 / k) - 1.0


def umvue_variance(n_true, k):
    """
    Exact variance of UMVUE:
    Var(N_hat) = (N + 1)(N - k) / (k * (k + 2))
    """
    return (n_true + 1.0) * (n_true - k) / (k * (k + 2.0))


def mle(m):
    """
    Maximum Likelihood Estimator:
    N_hat = m
    """
    return np.asarray(m, dtype=np.float64)


def frequentist_ci(m, k, confidence=0.95):
    """
    Frequentist confidence interval based on the sampling distribution of M.
    P(M <= m | N) = binom(m, k) / binom(N, k).
    Standard one-sided upper (1 - alpha) bound: N <= m / alpha^(1/k)
    For central 95%:
      Lower bound: m (since N >= m with probability 1)
      Upper bound: smallest N such that binom(m, k) / binom(N, k) <= alpha
    """
    m = int(m)
    k = int(k)
    alpha = 1.0 - confidence

    # Exact discrete search for upper bound
    target_log = log_comb(m, k) - np.log(alpha)
    low, high = m, max(m * 20, m + 50000)
    while low < high:
        mid = (low + high) // 2
        if log_comb(mid, k) >= target_log:
            high = mid
        else:
            low = mid + 1
    u_bound = int(low)
    return (m, u_bound)
