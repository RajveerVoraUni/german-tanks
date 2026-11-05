"""
bayes_factors.py: Hypothesis testing, Bayes factors, and Lindley-Jeffreys prior-width diagnostics.
"""

import numpy as np
from src.exact_posterior import log_comb


def legal_cap_bayes_factor(m, k, L, omega=100000, prior_type="uniform"):
    """
    Computes Bayes Factor BF_10 for:
      H0: N <= L
      H1: N > L
    Under a proper prior on {1, ..., omega} (where omega > L).
    Returns:
      bf_10: Bayes factor in favor of H1 over H0
      log_bf_10: log(BF_10)
      p_h1_post: Posterior probability P(N > L | S) under equal prior odds
    """
    m = int(m)
    k = int(k)
    L = int(L)
    omega = int(omega)

    if m > L:
        # If observed maximum already exceeds L, P(S | H0) is strictly 0
        return float("inf"), float("inf"), 1.0

    if omega <= L:
        raise ValueError(f"Prior bound omega={omega} must be greater than cap L={L}")

    # States under H0: m <= n <= L
    n_h0 = np.arange(m, L + 1, dtype=np.float64)
    # States under H1: L < n <= omega
    n_h1 = np.arange(L + 1, omega + 1, dtype=np.float64)

    if len(n_h0) == 0:
        return float("inf"), float("inf"), 1.0

    # Log likelihoods: - log_comb(n, k)
    ll_h0 = -log_comb(n_h0, k)
    ll_h1 = -log_comb(n_h1, k)

    # Priors
    if prior_type == "uniform":
        # Prior mass per state = 1/omega
        # P(H0) = L / omega, P(H1) = (omega - L) / omega
        # Marginal likelihood P(S | H0) = sum_{m..L} exp(ll_h0) / L
        # Marginal likelihood P(S | H1) = sum_{L+1..omega} exp(ll_h1) / (omega - L)
        max_h0 = np.max(ll_h0)
        log_marg_h0 = max_h0 + np.log(np.sum(np.exp(ll_h0 - max_h0))) - np.log(L)

        max_h1 = np.max(ll_h1)
        log_marg_h1 = (
            max_h1
            + np.log(np.sum(np.exp(ll_h1 - max_h1)))
            - np.log(float(omega - L))
        )

        log_prior_odds = np.log(float(omega - L)) - np.log(float(L))
    else:
        raise NotImplementedError(f"Prior {prior_type} not implemented for BF")

    log_bf_10 = log_marg_h1 - log_marg_h0
    bf_10 = np.exp(np.clip(log_bf_10, -100, 100))

    # Posterior odds and P(H1 | S)
    log_post_odds = log_bf_10 + log_prior_odds
    p_h1_post = 1.0 / (1.0 + np.exp(-np.clip(log_post_odds, -100, 100)))

    return float(bf_10), float(log_bf_10), float(p_h1_post)


def lindley_jeffreys_sweep(m, k, L, omega_values):
    """
    Evaluates BF_10 across varying prior width omega to demonstrate the Lindley-Jeffreys paradox:
    As prior support omega -> inf, BF_10 -> 0 even when data strongly favors H1.
    """
    results = []
    for omega in omega_values:
        bf_10, log_bf_10, p_h1 = legal_cap_bayes_factor(
            m, k, L, omega=omega, prior_type="uniform"
        )
        results.append(
            {"omega": omega, "bf_10": bf_10, "log_bf_10": log_bf_10, "p_h1_post": p_h1}
        )
    return results
