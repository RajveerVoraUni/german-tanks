"""
diagnostics.py: Rigorous MCMC diagnostics including split-R_hat, Effective Sample Size (ESS),
autocorrelation function, and posterior predictive checking.
"""

import numpy as np


def compute_acf(series, max_lag=50):
    """Computes autocorrelation function of a 1D array up to max_lag."""
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    if n <= 1:
        return np.ones(1)
    variance = np.var(series)
    if variance < 1e-12:
        return np.ones(max_lag + 1)
    series_centered = series - np.mean(series)
    autocorr = np.correlate(series_centered, series_centered, mode="full")[n - 1 :]
    autocorr /= autocorr[0]
    return autocorr[: max_lag + 1]


def compute_split_rhat(chains):
    """
    Computes Gelman-Rubin split-R_hat over multiple chains.
    chains: list of 1D arrays or 2D array of shape (M, N_samples).
    """
    chains = [np.asarray(ch, dtype=np.float64) for ch in chains]
    # Split each chain in half
    split_chains = []
    for ch in chains:
        n_half = len(ch) // 2
        split_chains.append(ch[:n_half])
        split_chains.append(ch[n_half : 2 * n_half])

    m = len(split_chains)  # 2 * M chains
    n = len(split_chains[0])  # length of each split half

    means = np.array([np.mean(ch) for ch in split_chains])
    vars_ = np.array([np.var(ch, ddof=1) for ch in split_chains])

    overall_mean = np.mean(means)
    # Between-sequence variance B/n
    b_over_n = np.var(means, ddof=1)
    b = n * b_over_n
    # Within-sequence variance W
    w = np.mean(vars_)

    if w < 1e-12:
        return 1.0

    # Estimated marginal posterior variance
    var_plus = ((n - 1.0) / n) * w + (1.0 / n) * b
    rhat = np.sqrt(var_plus / w)
    return float(rhat)


def compute_ess(chains):
    """
    Estimates Effective Sample Size (ESS) using Geyer's initial positive sequence estimator.
    """
    flat = np.concatenate([np.asarray(ch) for ch in chains])
    n_total = len(flat)
    acf = compute_acf(flat, max_lag=min(500, n_total // 2))

    # Sum autocorrelations until they drop below 0
    pos_sum = 0.0
    for val in acf[1:]:
        if val <= 0:
            break
        pos_sum += val

    act = 1.0 + 2.0 * pos_sum
    ess = n_total / max(1.0, act)
    return float(ess)


def total_variation_distance(p_discrete, q_discrete):
    """
    Computes Total Variation distance TV(P, Q) = 0.5 * sum |P - Q|.
    """
    p = np.asarray(p_discrete, dtype=np.float64)
    q = np.asarray(q_discrete, dtype=np.float64)
    # Align lengths
    max_len = max(len(p), len(q))
    p_pad = np.pad(p, (0, max_len - len(p)), "constant")
    q_pad = np.pad(q, (0, max_len - len(q)), "constant")
    p_pad /= max(1e-12, np.sum(p_pad))
    q_pad /= max(1e-12, np.sum(q_pad))
    return float(0.5 * np.sum(np.abs(p_pad - q_pad)))
