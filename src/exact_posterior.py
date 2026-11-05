"""
exact_posterior.py: Numerically stable discrete Bayesian posterior inference for the German Tank Problem.
Optimized with master array precomputation for instantaneous slicing across Monte Carlo replicates.
"""

import numpy as np
from scipy.special import gammaln


def log_comb(n, k):
    """
    Numerically stable log binomial coefficient: ln(n! / (k! * (n-k)!))
    Works with scalar or numpy arrays.
    """
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    res = np.full_like(n, -np.inf)
    valid = (n >= k) & (k >= 0) & (n >= 0)
    res[valid] = gammaln(n[valid] + 1.0) - gammaln(k + 1.0) - gammaln(n[valid] - k + 1.0)
    return res


class FastPrecomputedModel:
    """
    Precomputes log-likelihood and log-prior over a master array n in [1, N_max] for a fixed k.
    Allows O(1) posterior evaluation for any observed m by slicing.
    """

    def __init__(self, k, prior_type="flat", prior_params=None, n_max=150000):
        self.k = int(k)
        self.prior_type = prior_type
        self.prior_params = prior_params or {}
        self.n_max = int(n_max)

        # Master support: n = 1 .. n_max
        self.master_n = np.arange(1, self.n_max + 1, dtype=np.float64)

        # Precompute log likelihood: -log_comb(n, k)
        self.log_lik = np.full(self.n_max, -np.inf, dtype=np.float64)
        valid_k = self.master_n >= self.k
        self.log_lik[valid_k] = -(
            gammaln(self.master_n[valid_k] + 1.0)
            - gammaln(float(self.k) + 1.0)
            - gammaln(self.master_n[valid_k] - float(self.k) + 1.0)
        )

        # Precompute log prior
        if self.prior_type == "flat":
            self.log_prior = np.zeros(self.n_max, dtype=np.float64)
        elif self.prior_type in ("reciprocal", "log_uniform", "1/n"):
            self.log_prior = -np.log(self.master_n)
        elif self.prior_type == "truncated_uniform":
            omega = int(self.prior_params.get("omega", 100000))
            self.log_prior = np.zeros(self.n_max, dtype=np.float64)
            if omega < self.n_max:
                self.log_prior[omega:] = -np.inf
        elif self.prior_type == "poisson":
            mu = float(self.prior_params.get("mu", 20000.0))
            self.log_prior = (
                self.master_n * np.log(mu) - mu - gammaln(self.master_n + 1.0)
            )
        elif self.prior_type == "negbin":
            r = float(self.prior_params.get("r", 10.0))
            p = float(self.prior_params.get("p", 0.001))
            self.log_prior = (
                log_comb(self.master_n + r - 1.0, self.master_n)
                + r * np.log(p)
                + self.master_n * np.log(1.0 - p)
            )
        else:
            raise ValueError(f"Unknown prior: {self.prior_type}")

        # Master unnormalized log-posterior
        self.log_unnorm = self.log_lik + self.log_prior

    def evaluate_sample(self, m, n_true=None):
        """
        Evaluates posterior summaries for an observed sample max m in milliseconds.
        Returns:
          mean, median, map, (hpd_low, hpd_high), (eq_low, eq_high), log_score, crps, pit_rank
        """
        m = int(m)
        k = self.k

        # Analytical shortcuts for Flat prior
        if self.prior_type == "flat" and k >= 3:
            mean_est = (k - 1.0) * (m - 1.0) / (k - 2.0)
        else:
            mean_est = None

        # Slicing from m to n_max
        if m > self.n_max:
            # Fallback if m exceeds precomputed grid
            m_idx = self.n_max - 1
        else:
            m_idx = m - 1  # 0-indexed corresponding to n = m

        log_slice = self.log_unnorm[m_idx:]
        support_slice = self.master_n[m_idx:]

        # Filter finite
        valid = np.isfinite(log_slice)
        if not np.any(valid):
            # Degenerate case
            return m, m, m, (m, m), (m, m), 0.0, 0.0, 0.5

        max_val = np.max(log_slice[valid])
        probs = np.zeros_like(log_slice)
        probs[valid] = np.exp(log_slice[valid] - max_val)
        sum_p = np.sum(probs)
        probs /= sum_p
        cdf = np.cumsum(probs)

        # Estimates
        if mean_est is None:
            mean_est = float(np.sum(support_slice * probs))

        map_idx = np.argmax(probs)
        map_est = int(support_slice[map_idx])

        med_idx = np.searchsorted(cdf, 0.5)
        med_est = int(support_slice[min(med_idx, len(support_slice) - 1)])

        # Equal-Tailed 95% interval
        eq_low_idx = np.searchsorted(cdf, 0.025)
        eq_high_idx = np.searchsorted(cdf, 0.975)
        eq_interval = (
            int(support_slice[min(eq_low_idx, len(support_slice) - 1)]),
            int(support_slice[min(eq_high_idx, len(support_slice) - 1)]),
        )

        # HPD 95% interval:
        if self.prior_type == "flat":
            # Exact decreasing property -> starts at m
            hpd_high_idx = np.searchsorted(cdf, 0.95)
            hpd_interval = (m, int(support_slice[min(hpd_high_idx, len(support_slice) - 1)]))
        else:
            sorted_idx = np.argsort(-probs)
            cum_sorted = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cum_sorted, 0.95)
            hpd_set = support_slice[sorted_idx[: cutoff + 1]]
            hpd_interval = (int(np.min(hpd_set)), int(np.max(hpd_set)))

        # Scoring rules
        log_score = float("inf")
        crps = 0.0
        pit_rank = 0.5

        if n_true is not None:
            n_true = int(n_true)
            if n_true >= m:
                idx_true = n_true - m
                if idx_true < len(probs) and probs[idx_true] > 0:
                    log_score = float(-np.log(probs[idx_true]))
                heaviside = (support_slice >= n_true).astype(np.float64)
                crps = float(np.sum((cdf - heaviside) ** 2))
                pit_rank = float(cdf[min(idx_true, len(cdf) - 1)])
            else:
                log_score = float("inf")
                heaviside = (support_slice >= n_true).astype(np.float64)
                crps = float(np.sum((cdf - heaviside) ** 2))
                pit_rank = 0.0

        return (
            mean_est,
            med_est,
            map_est,
            hpd_interval,
            eq_interval,
            log_score,
            crps,
            pit_rank,
        )

    def posterior_pmf(self, m):
        """Returns support array and normalized posterior probabilities."""
        m = int(m)
        m_idx = min(m - 1, self.n_max - 1)
        support = self.master_n[m_idx:].astype(int)
        log_slice = self.log_unnorm[m_idx:]
        valid = np.isfinite(log_slice)
        probs = np.zeros_like(log_slice)
        if np.any(valid):
            probs[valid] = np.exp(log_slice[valid] - np.max(log_slice[valid]))
            probs /= np.sum(probs)
        return support, probs


class GermanTankExactBayesian:
    """Wrapper class for single evaluations."""

    def __init__(self, m, k, prior_type="flat", prior_params=None, grid_max=150000):
        self.model = FastPrecomputedModel(
            k, prior_type=prior_type, prior_params=prior_params, n_max=grid_max
        )
        self.m = m
        self.k = k

    def evaluate(self, n_true=None):
        return self.model.evaluate_sample(self.m, n_true=n_true)

    def posterior_mean(self):
        res = self.evaluate()
        return res[0]

    def posterior_median(self):
        res = self.evaluate()
        return res[1]

    def credible_interval_hpd(self, coverage=0.95):
        res = self.evaluate()
        return res[3]

    def posterior_pmf(self):
        return self.model.posterior_pmf(self.m)
