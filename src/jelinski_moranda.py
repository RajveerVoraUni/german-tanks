"""
jelinski_moranda.py: Exact Bayesian software reliability inference under the Jelinski-Moranda model (WP4).
Integrates out hazard rate phi in closed form, providing exact discrete posteriors for N and remaining defects N-k.
"""

import numpy as np
from scipy.special import gammaln


class JelinskiMorandaBayesian:
    """
    Exact discrete Bayesian inference for Jelinski-Moranda software reliability:
      t_i ~ Exp(phi * (N - i + 1)),  i = 1..k
      phi ~ Gamma(a, b) [shape a, rate b]
      Integrated marginal likelihood:
        p(t_{1:k} | N) = [b^a * Gamma(a+k) / Gamma(a)] * [N! / (N-k)!] / (b + T_N)^(a+k)
        where T_N = sum_{i=1}^k (N - i + 1) * t_i
    """

    def __init__(self, t_failures, a=1.0, b=100.0, prior_type="flat", prior_params=None, n_max=1000):
        self.t = np.asarray(t_failures, dtype=np.float64)
        self.k = len(self.t)
        self.a = float(a)
        self.b = float(b)
        self.prior_type = prior_type
        self.prior_params = prior_params or {}
        self.n_max = max(int(n_max), self.k + 200)

        # Precompute sum(t) and sum((i-1)*t_i)
        i_vec = np.arange(1, self.k + 1, dtype=np.float64)
        self.sum_t = np.sum(self.t)
        self.sum_i_minus_1_t = np.sum((i_vec - 1.0) * self.t)

        self.support = np.arange(self.k, self.n_max + 1, dtype=np.int64)
        self._compute_posterior()

    def _compute_posterior(self):
        n = self.support.astype(np.float64)
        k = float(self.k)

        # T_n = n * sum(t) - sum((i-1)*t_i)
        T_n = n * self.sum_t - self.sum_i_minus_1_t

        # Log marginal likelihood
        # ln p(t | n) = a*ln(b) + gammaln(a+k) - gammaln(a) + gammaln(n+1) - gammaln(n-k+1) - (a+k)*ln(b + T_n)
        log_lik = (
            self.a * np.log(self.b)
            + gammaln(self.a + k)
            - gammaln(self.a)
            + gammaln(n + 1.0)
            - gammaln(n - k + 1.0)
            - (self.a + k) * np.log(self.b + T_n)
        )

        # Prior
        if self.prior_type == "flat":
            log_prior = np.zeros_like(n)
        elif self.prior_type in ("1/n", "reciprocal"):
            log_prior = -np.log(n)
        elif self.prior_type == "poisson":
            mu = float(self.prior_params.get("mu", self.k * 1.5))
            log_prior = n * np.log(mu) - mu - gammaln(n + 1.0)
        else:
            log_prior = np.zeros_like(n)

        log_unnorm = log_lik + log_prior
        valid = np.isfinite(log_unnorm)

        max_val = np.max(log_unnorm[valid])
        probs = np.zeros_like(n)
        probs[valid] = np.exp(log_unnorm[valid] - max_val)
        probs /= np.sum(probs)

        self.probs = probs
        self.cdf = np.cumsum(probs)

    def posterior_mean_N(self):
        return float(np.sum(self.support * self.probs))

    def posterior_mean_remaining(self):
        return self.posterior_mean_N() - self.k

    def posterior_median_N(self):
        idx = np.searchsorted(self.cdf, 0.5)
        return int(self.support[min(idx, len(self.support) - 1)])

    def credible_interval_N(self, coverage=0.95):
        alpha = 1.0 - coverage
        low_idx = np.searchsorted(self.cdf, alpha / 2.0)
        high_idx = np.searchsorted(self.cdf, 1.0 - alpha / 2.0)
        return (
            int(self.support[min(low_idx, len(self.support) - 1)]),
            int(self.support[min(high_idx, len(self.support) - 1)]),
        )

    def release_rule_probability(self, max_remaining=5):
        """P(N - k <= max_remaining | data)."""
        cutoff_N = self.k + int(max_remaining)
        idx = np.searchsorted(self.support, cutoff_N, side="right") - 1
        if idx < 0:
            return 0.0
        return float(self.cdf[min(idx, len(self.cdf) - 1)])

    def asymmetric_loss_quantile(self, cu_delay_ratio=4.0):
        """Quantile tau = cu / (cu + co) for residual bug risk planning."""
        tau = cu_delay_ratio / (cu_delay_ratio + 1.0)
        idx = np.searchsorted(self.cdf, tau)
        return int(self.support[min(idx, len(self.support) - 1)]) - self.k

    def predict_next_failure_log_score(self, t_next):
        """
        Predictive log score for the next held-out inter-failure time.
        Conditioned on n, t_next follows a Lomax / Pareto Type II distribution:
          p(t_next | n) = (a+k)(n-k)(b + T_n)^(a+k) / (b + T_n + (n-k)*t_next)^(a+k+1)
        """
        t_next = float(t_next)
        n = self.support.astype(np.float64)
        k = float(self.k)
        active_remaining = n - k
        valid = active_remaining > 0

        T_n = n * self.sum_t - self.sum_i_minus_1_t
        pred_densities = np.zeros_like(n)

        # Lomax density
        denom = self.b + T_n[valid] + active_remaining[valid] * t_next
        log_pred = (
            np.log(self.a + k)
            + np.log(active_remaining[valid])
            + (self.a + k) * np.log(self.b + T_n[valid])
            - (self.a + k + 1.0) * np.log(denom)
        )
        pred_densities[valid] = np.exp(log_pred)
        marginal_pred_density = float(np.sum(pred_densities * self.probs))
        return (
            -np.log(marginal_pred_density)
            if marginal_pred_density > 0
            else float("inf")
        )
