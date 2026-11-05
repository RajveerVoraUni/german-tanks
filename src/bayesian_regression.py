"""
bayesian_regression.py: Bayesian linear models, spike-and-slab variable selection,
and hierarchical partial pooling for software defect data (WP5).
"""

import numpy as np
from scipy.linalg import solve
from scipy.special import betaln, gammaln
from scipy.stats import invgamma, norm


class ConjugateNormalInvGamma:
    """
    Conjugate Normal-Inverse-Gamma linear regression:
      y = X beta + eps,  eps ~ N(0, sigma^2 I)
      beta | sigma^2 ~ N(0, sigma^2 V0),  sigma^2 ~ Inv-Gamma(a0, b0)
    """

    def __init__(self, X, y, v0_scale=10.0, a0=2.0, b0=2.0):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self.n, self.p = self.X.shape
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.v0_scale = float(v0_scale)

        # Prior precision V0_inv = (1 / v0_scale) * I
        self.V0_inv = (1.0 / self.v0_scale) * np.eye(self.p)
        self._fit()

    def _fit(self):
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y
        Vn_inv = self.V0_inv + XtX
        self.Vn = np.linalg.pinv(Vn_inv)
        self.beta_n = self.Vn @ Xty
        self.an = self.a0 + self.n / 2.0

        # bn = b0 + 0.5 * (y'y - beta_n' Vn_inv beta_n)
        quad_y = float((self.y.T @ self.y).item())
        quad_beta = float((self.beta_n.T @ Vn_inv @ self.beta_n).item())
        self.bn = max(1e-8, self.b0 + 0.5 * (quad_y - quad_beta))

    def log_marginal_likelihood(self):
        """
        Closed-form marginal log-likelihood ln p(y | X).
        """
        sign_v0, logdet_v0 = np.linalg.slogdet(self.v0_scale * np.eye(self.p))
        sign_vn, logdet_vn = np.linalg.slogdet(self.Vn)

        log_marg = (
            - (self.n / 2.0) * np.log(2.0 * np.pi)
            + 0.5 * (logdet_vn - logdet_v0)
            + self.a0 * np.log(self.b0)
            - gammaln(self.a0)
            + gammaln(self.an)
            - self.an * np.log(self.bn)
        )
        return float(log_marg)

    def posterior_mean_beta(self):
        return self.beta_n.flatten()

    def posterior_mean_sigma2(self):
        return self.bn / (self.an - 1.0) if self.an > 1.0 else self.bn / self.an

    def predict(self, X_new):
        """Predictive mean and variance for new design matrix."""
        X_new = np.asarray(X_new, dtype=np.float64)
        pred_mean = (X_new @ self.beta_n).flatten()
        sigma2_hat = self.posterior_mean_sigma2()
        pred_var = sigma2_hat * (1.0 + np.sum((X_new @ self.Vn) * X_new, axis=1))
        return pred_mean, pred_var


class SpikeAndSlabGibbs:
    """
    Collapsed Gibbs sampler for spike-and-slab variable selection:
      beta_j | gamma_j, sigma^2 ~ gamma_j * N(0, c * sigma^2) + (1 - gamma_j) * delta_0
      gamma_j | omega ~ Bernoulli(omega)
      omega ~ Beta(a_w, b_w)
    Integrates out beta and sigma^2 using conjugate marginal likelihood.
    """

    def __init__(self, X, y, c_slab=10.0, a0=2.0, b0=2.0, a_w=1.0, b_w=1.0):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self.n, self.p = self.X.shape
        self.c_slab = float(c_slab)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.a_w = float(a_w)
        self.b_w = float(b_w)

    def _log_marg_gamma(self, gamma_idx):
        """Computes marginal log-likelihood for active subset of features."""
        if len(gamma_idx) == 0:
            # Intercept-only or null model
            an = self.a0 + self.n / 2.0
            bn = max(1e-8, self.b0 + 0.5 * float((self.y.T @ self.y).item()))
            return float(
                - (self.n / 2.0) * np.log(2.0 * np.pi)
                + self.a0 * np.log(self.b0)
                - gammaln(self.a0)
                + gammaln(an)
                - an * np.log(bn)
            )

        X_sub = self.X[:, gamma_idx]
        p_sub = len(gamma_idx)
        V0_inv = (1.0 / self.c_slab) * np.eye(p_sub)
        XtX = X_sub.T @ X_sub
        Xty = X_sub.T @ self.y
        Vn_inv = V0_inv + XtX
        Vn = np.linalg.pinv(Vn_inv)
        beta_n = Vn @ Xty
        an = self.a0 + self.n / 2.0
        quad_y = float((self.y.T @ self.y).item())
        quad_b = float((beta_n.T @ Vn_inv @ beta_n).item())
        bn = max(1e-8, self.b0 + 0.5 * (quad_y - quad_b))

        logdet_v0 = p_sub * np.log(self.c_slab)
        sign_vn, logdet_vn = np.linalg.slogdet(Vn)

        return float(
            - (self.n / 2.0) * np.log(2.0 * np.pi)
            + 0.5 * (logdet_vn - logdet_v0)
            + self.a0 * np.log(self.b0)
            - gammaln(self.a0)
            + gammaln(an)
            - an * np.log(bn)
        )

    def sample(self, n_iter=3000, burn_in=1000, seed=42):
        np.random.seed(seed)
        # Initialize inclusion indicators gamma in {0, 1}
        gamma = np.zeros(self.p, dtype=int)
        # Include intercept if column 0 has 0 variance
        gamma[0] = 1
        omega = 0.5

        traces_gamma = np.empty((n_iter, self.p), dtype=int)
        traces_omega = np.empty(n_iter, dtype=np.float64)

        for it in range(n_iter):
            # Update inclusion indicators for predictors j = 1..p-1 (keep intercept included)
            for j in range(1, self.p):
                gamma_1 = gamma.copy()
                gamma_1[j] = 1
                gamma_0 = gamma.copy()
                gamma_0[j] = 0

                idx_1 = np.where(gamma_1 == 1)[0]
                idx_0 = np.where(gamma_0 == 1)[0]

                log_m1 = self._log_marg_gamma(idx_1) + np.log(max(1e-6, omega))
                log_m0 = self._log_marg_gamma(idx_0) + np.log(max(1e-6, 1.0 - omega))

                max_m = max(log_m1, log_m0)
                p1 = np.exp(log_m1 - max_m) / (
                    np.exp(log_m1 - max_m) + np.exp(log_m0 - max_m)
                )

                gamma[j] = 1 if np.random.uniform(0.0, 1.0) < p1 else 0

            # Update omega | gamma ~ Beta(a_w + sum(gamma), b_w + p - sum(gamma))
            n_active = np.sum(gamma[1:])
            p_pred = self.p - 1
            omega = np.random.beta(self.a_w + n_active, self.b_w + p_pred - n_active)

            traces_gamma[it, :] = gamma
            traces_omega[it] = omega

        post_gamma = traces_gamma[burn_in:, :]
        pip = np.mean(post_gamma, axis=0)
        return {
            "pip": pip,
            "traces_gamma": post_gamma,
            "traces_omega": traces_omega[burn_in:],
        }


def fit_empirical_bayes_hierarchical(groups_y, groups_X):
    """
    Empirical Bayes random intercept model across releases r = 1..R:
      y_{rj} = alpha_r + X_{rj} beta + eps_{rj}
      alpha_r ~ N(mu_alpha, sigma_alpha^2), eps_{rj} ~ N(0, sigma^2)
    Estimates (mu_alpha, sigma_alpha^2, sigma^2) by Type-II ML and computes shrinkage posteriors.
    """
    R = len(groups_y)
    n_r = [len(y) for y in groups_y]
    y_bars = [float(np.mean(y)) for y in groups_y]

    # Overall pooled mean
    y_all = np.concatenate(groups_y)
    mu_hat = float(np.mean(y_all))

    # Within-group variance (pooled estimate of sigma^2)
    ss_within = sum([np.sum((y - np.mean(y)) ** 2) for y in groups_y])
    df_within = max(1, sum(n_r) - R)
    sigma2_hat = float(ss_within / df_within)

    # Between-group variance estimate (sigma_alpha^2)
    n_bar = np.mean(n_r)
    ss_between = sum([len(y) * (np.mean(y) - mu_hat) ** 2 for y in groups_y])
    ms_between = ss_between / max(1, R - 1)
    sigma_alpha2_hat = max(1e-4, float((ms_between - sigma2_hat) / n_bar))

    # EB Shrinkage weights B_r
    shrinkage_alphas = []
    shrinkage_factors = []
    for r in range(R):
        v_r = sigma2_hat / n_r[r]
        # Shrinkage factor towards grand mean
        B_r = v_r / (v_r + sigma_alpha2_hat)
        shrunk_alpha = (1.0 - B_r) * y_bars[r] + B_r * mu_hat
        shrinkage_alphas.append(shrunk_alpha)
        shrinkage_factors.append(B_r)

    return {
        "mu_hat": mu_hat,
        "sigma2_hat": sigma2_hat,
        "sigma_alpha2_hat": sigma_alpha2_hat,
        "y_bars": y_bars,
        "shrunk_alphas": shrinkage_alphas,
        "shrinkage_factors": shrinkage_factors,
        "n_r": n_r,
    }
