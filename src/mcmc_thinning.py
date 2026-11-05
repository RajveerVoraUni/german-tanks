"""
mcmc_thinning.py: Gibbs sampler for the German Tank random sample size thinning model (Proposition 1).
"""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import poisson as poisson_dist


def sample_truncated_poisson(lam, m_min, size=1):
    """
    Samples from Poisson(lam) conditioned on X >= m_min.
    Uses inverse CDF transform when lam >= m_min / 3, with robust discrete tail fallback.
    """
    lam = max(1e-6, float(lam))
    m_min = int(m_min)

    # Evaluate CDF at m_min - 1
    if m_min <= 0:
        return poisson_dist.rvs(lam, size=size)

    f_low = poisson_dist.cdf(m_min - 1, lam)

    # If tail is very far out (lam << m_min), sample from normalized discrete local support
    if f_low >= 1.0 - 1e-12:
        support_len = 100
        n_vals = np.arange(m_min, m_min + support_len, dtype=np.float64)
        from scipy.special import gammaln

        log_p = n_vals * np.log(lam) - gammaln(n_vals + 1.0)
        p = np.exp(log_p - np.max(log_p))
        p /= np.sum(p)
        drawn = np.random.choice(n_vals.astype(int), size=size, p=p)
        return int(drawn[0]) if size == 1 else drawn

    u = np.random.uniform(f_low, 1.0, size=size)
    samples = poisson_dist.ppf(u, lam).astype(int)
    samples = np.maximum(samples, m_min)
    return int(samples[0]) if size == 1 else samples


class ThinningGibbsSampler:
    """
    Gibbs Sampler for Proposition 1 Thinning Model:
      N ~ Poisson(mu)
      Captured set S of size k: each unit captured with prob p
      Priors: p ~ Beta(a, b), mu ~ Gamma(alpha, beta) [rate beta]
    """

    def __init__(self, m, k, a=1.0, b=1.0, alpha=1.0, beta=1e-4):
        self.m = int(m)
        self.k = int(k)
        self.a = float(a)
        self.b = float(b)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def sample(
        self,
        n_iter=10000,
        burn_in=2000,
        init_n=None,
        init_p=0.5,
        init_mu=None,
        seed=None,
    ):
        """
        Executes Gibbs sampling chain.
        Returns dict with posterior traces of N, p, and mu.
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize
        n_curr = int(init_n) if init_n is not None else max(self.m, int(self.m * 1.2))
        p_curr = float(init_p)
        mu_curr = float(init_mu) if init_mu is not None else float(n_curr)

        trace_n = np.empty(n_iter, dtype=np.int64)
        trace_p = np.empty(n_iter, dtype=np.float64)
        trace_mu = np.empty(n_iter, dtype=np.float64)

        for it in range(n_iter):
            # 1. Update N | S, mu, p ~ Poisson(mu * (1 - p)) truncated to n >= m
            lam_unobserved = max(1e-6, mu_curr * (1.0 - p_curr))
            n_curr = sample_truncated_poisson(lam_unobserved, self.m)

            # 2. Update p | N, S ~ Beta(a + k, b + N - k)
            a_post = self.a + self.k
            b_post = max(1e-6, self.b + (n_curr - self.k))
            p_curr = np.random.beta(a_post, b_post)

            # 3. Update mu | N ~ Gamma(alpha + N, beta + 1) [shape alpha+N, rate beta+1]
            shape_post = self.alpha + n_curr
            rate_post = self.beta + 1.0
            scale_post = 1.0 / rate_post
            mu_curr = np.random.gamma(shape_post, scale_post)

            trace_n[it] = n_curr
            trace_p[it] = p_curr
            trace_mu[it] = mu_curr

        # Discard burn-in
        return {
            "N": trace_n[burn_in:],
            "p": trace_p[burn_in:],
            "mu": trace_mu[burn_in:],
            "full_N": trace_n,
            "full_p": trace_p,
            "full_mu": trace_mu,
        }
