"""
mcmc_selection.py: Metropolis-within-Gibbs sampler for polynomial selection bias w(s; theta) = s^(-theta)
Includes exact discrete draw for N, Savage-Dickey density ratio test for theta=0, and 2D grid validation.
"""

import numpy as np
from scipy.stats import gaussian_kde, norm


def compute_zn_array(n_max, theta):
    """
    Computes Z_n(theta) = sum_{j=1}^n j^(-theta) for all n in 1..n_max.
    Vectorized cumulative sum.
    """
    j = np.arange(1, n_max + 1, dtype=np.float64)
    weights = np.power(j, -theta)
    return np.cumsum(weights)


class SelectionBiasMCMC:
    """
    Metropolis-within-Gibbs for Selection Bias:
      P(S | N, theta) = prod_{i=1}^k s_i^(-theta) / (Z_N(theta))^k * 1{N >= m}
      Priors: theta ~ N(0, sigma_theta^2), N ~ Truncated Uniform or 1/N
    """

    def __init__(
        self,
        data_sample,
        prior_n="flat",
        omega=100000,
        sigma_theta=1.0,
        proposal_sd=0.08,
    ):
        self.data_sample = np.asarray(data_sample, dtype=np.float64)
        self.k = len(data_sample)
        self.m = int(np.max(data_sample))
        self.sum_log_s = float(np.sum(np.log(self.data_sample)))
        self.prior_n = prior_n
        self.omega = int(omega)
        self.sigma_theta = float(sigma_theta)
        self.proposal_sd = float(proposal_sd)

    def sample(
        self,
        n_iter=10000,
        burn_in=2000,
        init_n=None,
        init_theta=0.0,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        n_curr = int(init_n) if init_n is not None else max(self.m, int(self.m * 1.1))
        theta_curr = float(init_theta)

        trace_n = np.empty(n_iter, dtype=np.int64)
        trace_theta = np.empty(n_iter, dtype=np.float64)
        accepted_theta = 0

        # Master candidate support for N: m .. omega
        n_support = np.arange(self.m, self.omega + 1, dtype=np.int64)
        n_float = n_support.astype(np.float64)

        if self.prior_n == "flat":
            log_prior_n = np.zeros_like(n_float)
        elif self.prior_n in ("1/n", "reciprocal"):
            log_prior_n = -np.log(n_float)
        else:
            log_prior_n = np.zeros_like(n_float)

        current_sd = self.proposal_sd
        burn_accepted = 0

        for it in range(n_iter):
            # 1. Update theta | N, S via Random-Walk Metropolis-Hastings
            theta_prop = theta_curr + np.random.normal(0.0, current_sd)

            # Precompute Z_n for current n under both thetas
            j_vec = np.arange(1, n_curr + 1, dtype=np.float64)
            z_curr = np.sum(np.power(j_vec, -theta_curr))
            z_prop = np.sum(np.power(j_vec, -theta_prop))

            # Log prior ratio
            log_prior_ratio = (
                -0.5 * (theta_prop / self.sigma_theta) ** 2
                + 0.5 * (theta_curr / self.sigma_theta) ** 2
            )

            # Log likelihood ratio
            log_lik_ratio = (
                -self.k * (np.log(z_prop) - np.log(z_curr))
                - (theta_prop - theta_curr) * self.sum_log_s
            )

            log_alpha = log_prior_ratio + log_lik_ratio

            is_accepted = np.log(np.random.uniform(0.0, 1.0)) < log_alpha
            if is_accepted:
                theta_curr = theta_prop
                if it >= burn_in:
                    accepted_theta += 1
                else:
                    burn_accepted += 1

            # Adaptive tuning of proposal_sd during burn-in (every 100 iterations)
            if it < burn_in and it > 0 and it % 100 == 0:
                acc_rate = burn_accepted / 100.0
                if acc_rate > 0.44:
                    current_sd *= 1.15
                elif acc_rate < 0.25:
                    current_sd *= 0.85
                current_sd = float(np.clip(current_sd, 0.02, 0.5))
                burn_accepted = 0

            # 2. Update N | theta, S via exact discrete draw
            # Compute Z_n(theta_curr) for n in 1..omega
            zn_all = compute_zn_array(self.omega, theta_curr)
            zn_slice = zn_all[self.m - 1 :]  # corresponding to m .. omega

            # Log unnormalized mass: log_prior(n) - k * log(Z_n)
            log_mass = log_prior_n - self.k * np.log(zn_slice)
            max_log = np.max(log_mass)
            mass = np.exp(log_mass - max_log)
            mass /= np.sum(mass)

            # Draw discrete N
            n_curr = int(np.random.choice(n_support, p=mass))

            trace_n[it] = n_curr
            trace_theta[it] = theta_curr

        acceptance_rate = accepted_theta / float(n_iter)

        return {
            "N": trace_n[burn_in:],
            "theta": trace_theta[burn_in:],
            "full_N": trace_n,
            "full_theta": trace_theta,
            "acceptance_rate": acceptance_rate,
        }

    def compute_savage_dickey_bf(self, theta_samples):
        """
        Computes Savage-Dickey density ratio for H0: theta = 0 (no selection bias):
          BF_01 = pi(theta = 0 | S) / pi(theta = 0)
        """
        theta_samples = np.asarray(theta_samples)
        kde = gaussian_kde(theta_samples)
        post_density_0 = float(kde(0.0)[0])
        prior_density_0 = float(norm.pdf(0.0, loc=0.0, scale=self.sigma_theta))
        bf_01 = (
            post_density_0 / prior_density_0 if prior_density_0 > 0 else float("inf")
        )
        return bf_01, post_density_0, prior_density_0

    def compute_exact_2d_grid(self, theta_grid, n_grid):
        """
        Evaluates exact joint posterior pi(N, theta | S) on a 2D grid for validation.
        """
        n_grid = np.asarray(n_grid, dtype=np.int64)
        theta_grid = np.asarray(theta_grid, dtype=np.float64)

        log_prior_theta = norm.logpdf(theta_grid, loc=0.0, scale=self.sigma_theta)
        max_n = np.max(n_grid)

        joint_log_post = np.empty((len(theta_grid), len(n_grid)), dtype=np.float64)

        for i, th in enumerate(theta_grid):
            zn_all = compute_zn_array(max_n, th)
            zn_vals = zn_all[n_grid - 1]
            log_lik = -self.k * np.log(zn_vals) - th * self.sum_log_s
            joint_log_post[i, :] = log_prior_theta[i] + log_lik

        max_val = np.max(joint_log_post)
        joint_post = np.exp(joint_log_post - max_val)
        joint_post /= np.sum(joint_post)
        return joint_post
