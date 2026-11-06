import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats

def run_simulation(N_true=20000, k_vals=[5, 50], R=4000, seed=42):
    np.random.seed(seed)
    results = []

    for k in k_vals:
        print(f"\n--- Running k = {k}, N = {N_true}, R = {R} ---")
        # Draw R samples of size k from 1..N without replacement
        # Fast sampling of maximum m:
        # P(M <= m) = prod_{j=0}^{k-1} (m - j) / (N - j) ~ (m/N)^k
        # Exact discrete sampling:
        m_samples = np.zeros(R, dtype=int)
        for r in range(R):
            # Sample k from 1..N
            sample = np.random.choice(N_true, size=k, replace=False) + 1
            m_samples[r] = np.max(sample)
        
        # 1. UMVUE
        umvue_est = m_samples * (k + 1) / k - 1
        umvue_bias = np.mean(umvue_est - N_true)
        umvue_rmse = np.sqrt(np.mean((umvue_est - N_true)**2))
        umvue_se = np.std(umvue_est - N_true) / np.sqrt(R)
        
        # UMVUE Wald CI: est +- 1.96 * se_hat
        # se_hat = sqrt((est - k)*(est + 1) / (k*(k+2)))
        var_hat = np.maximum(1.0, (umvue_est - k) * (umvue_est + 1) / (k * (k + 2)))
        se_hat = np.sqrt(var_hat)
        wald_lower = umvue_est - 1.96 * se_hat
        wald_upper = umvue_est + 1.96 * se_hat
        wald_cov = np.mean((wald_lower <= N_true) & (N_true <= wald_upper))
        wald_cov_se = np.sqrt(wald_cov * (1 - wald_cov) / R)
        wald_width = np.mean(wald_upper - wald_lower)
        wald_boundary_violations = np.mean(wald_lower < m_samples)
        
        # Exact frequentist CI: [m, m * (alpha)^(-1/k)]
        alpha = 0.05
        exact_freq_lower = m_samples.astype(float)
        exact_freq_upper = m_samples * (alpha ** (-1.0 / k))
        exact_freq_cov = np.mean((exact_freq_lower <= N_true) & (N_true <= exact_freq_upper))
        exact_freq_cov_se = np.sqrt(exact_freq_cov * (1 - exact_freq_cov) / R)
        exact_freq_width = np.mean(exact_freq_upper - exact_freq_lower)

        # 2. Bayes Flat Prior pi(N) \propto 1
        # Posterior: P(N=n | m, k) = ((k-1)/k) * binom(m-1, k-1) / binom(n, k) for n >= m
        # Posterior Mean (for k >= 3): (k-1)*(m-1)/(k-2)
        flat_mean_est = (k - 1) * (m_samples - 1) / (k - 2) if k >= 3 else np.full(R, np.nan)
        flat_mean_bias = np.mean(flat_mean_est - N_true) if k >= 3 else np.nan
        flat_mean_rmse = np.sqrt(np.mean((flat_mean_est - N_true)**2)) if k >= 3 else np.nan
        
        # Posterior Median: ~ m * 2^(1/(k-1))
        flat_med_est = m_samples * (2.0 ** (1.0 / (k - 1)))
        flat_med_bias = np.mean(flat_med_est - N_true)
        flat_med_rmse = np.sqrt(np.mean((flat_med_est - N_true)**2))

        # Posterior Mode (MAP): m
        flat_mode_est = m_samples.astype(float)
        flat_mode_bias = np.mean(flat_mode_est - N_true)
        flat_mode_rmse = np.sqrt(np.mean((flat_mode_est - N_true)**2))

        # Exact 95% Credible Interval for Flat Prior:
        # HPD / Equal-tailed: CDF F(n | m, k) = 1 - binom(m-1, k-1) / binom(n, k-1)
        # Lower bound: m (since mode is m, HPD starts at m)
        # Upper bound: n such that 1 - binom(m-1, k-1)/binom(n, k-1) >= 0.95
        # binom(m-1, k-1) / binom(n, k-1) <= 0.05 => n ~ m * (0.05)^(-1/(k-1))
        flat_hpd_lower = m_samples.astype(float)
        flat_hpd_upper = m_samples * (0.05 ** (-1.0 / (k - 1)))
        flat_hpd_cov = np.mean((flat_hpd_lower <= N_true) & (N_true <= flat_hpd_upper))
        flat_hpd_cov_se = np.sqrt(flat_hpd_cov * (1 - flat_hpd_cov) / R)
        flat_hpd_width = np.mean(flat_hpd_upper - flat_hpd_lower)

        # 3. Bayes Reciprocal Prior pi(N) \propto 1/N
        # Pareto approximation: p(N | m, k) ~ k * m^k / N^(k+1) for N >= m
        # Posterior Mean (k >= 2): k*m / (k-1)
        recip_mean_est = k * m_samples / (k - 1)
        recip_mean_bias = np.mean(recip_mean_est - N_true)
        recip_mean_rmse = np.sqrt(np.mean((recip_mean_est - N_true)**2))

        # Posterior Median: m * 2^(1/k)
        recip_med_est = m_samples * (2.0 ** (1.0 / k))
        recip_med_bias = np.mean(recip_med_est - N_true)
        recip_med_rmse = np.sqrt(np.mean((recip_med_est - N_true)**2))

        # Reciprocal HPD: [m, m * (0.05)^(-1/k)]
        recip_hpd_lower = m_samples.astype(float)
        recip_hpd_upper = m_samples * (0.05 ** (-1.0 / k))
        recip_hpd_cov = np.mean((recip_hpd_lower <= N_true) & (N_true <= recip_hpd_upper))
        recip_hpd_cov_se = np.sqrt(recip_hpd_cov * (1 - recip_hpd_cov) / R)
        recip_hpd_width = np.mean(recip_hpd_upper - recip_hpd_lower)

        # 4. Hierarchical Hyperprior (Log-normal / Gamma hyperprior on scale)
        # N ~ Pareto(m, k), but with scale hyperprior => smoothly contracts without point-mass distortion
        # E.g. Gamma-Poisson or Hierarchical Bayes scale model
        # Let's test a broad lognormal hyperprior mu ~ LogNormal(ln(20000), 1.0)
        # Evaluated via numerical integration for a subset or Pareto conjugate
        
        # 5. Fixed Informative Prior Misspecification (testing the reviewer's point)
        # If someone uses a tight Gamma/Poisson with mu = 10,000, variance = mu:
        # Prior is essentially concentrated at 10,000 +- 200
        # If someone uses a moderately informative prior (e.g. Gamma with CV = 0.5):
        # alpha = 4, beta = 4/mu => SD = mu / 2 = 5,000 (wide, realistic)
        # Let's compute exact numbers:
        
        print(f"UMVUE: Bias={umvue_bias:.1f} +- {umvue_se:.1f}, RMSE={umvue_rmse:.1f}, WaldCov={wald_cov*100:.1f}% (+-{wald_cov_se*100:.2f}%), ExactCov={exact_freq_cov*100:.1f}% (+-{exact_freq_cov_se*100:.2f}%)")
        print(f"Flat Mean: Bias={flat_mean_bias:.1f}, RMSE={flat_mean_rmse:.1f}, HPD Cov={flat_hpd_cov*100:.1f}%, Width={flat_hpd_width:.1f}")
        print(f"Flat Median: Bias={flat_med_bias:.1f}, RMSE={flat_med_rmse:.1f}")
        print(f"Recip Mean: Bias={recip_mean_bias:.1f}, RMSE={recip_mean_rmse:.1f}, HPD Cov={recip_hpd_cov*100:.1f}%, Width={recip_hpd_width:.1f}")
        print(f"Recip Median: Bias={recip_med_bias:.1f}, RMSE={recip_med_rmse:.1f}")
        print(f"Wald Lower Bound Violation Rate: {wald_boundary_violations*100:.1f}%")

        results.append({
            "k": k,
            "N_true": N_true,
            "R": R,
            "umvue_bias": umvue_bias,
            "umvue_rmse": umvue_rmse,
            "wald_cov": wald_cov,
            "wald_cov_se": wald_cov_se,
            "wald_width": wald_width,
            "wald_violations": wald_boundary_violations,
            "exact_freq_cov": exact_freq_cov,
            "exact_freq_cov_se": exact_freq_cov_se,
            "exact_freq_width": exact_freq_width,
            "flat_mean_bias": flat_mean_bias,
            "flat_mean_rmse": flat_mean_rmse,
            "flat_med_bias": flat_med_bias,
            "flat_med_rmse": flat_med_rmse,
            "flat_hpd_cov": flat_hpd_cov,
            "flat_hpd_cov_se": flat_hpd_cov_se,
            "flat_hpd_width": flat_hpd_width,
            "recip_mean_bias": recip_mean_bias,
            "recip_mean_rmse": recip_mean_rmse,
            "recip_med_bias": recip_med_bias,
            "recip_med_rmse": recip_med_rmse,
            "recip_hpd_cov": recip_hpd_cov,
            "recip_hpd_cov_se": recip_hpd_cov_se,
            "recip_hpd_width": recip_hpd_width,
        })

    df = pd.DataFrame(results)
    df.to_csv("results/reproduced_benchmark_table.csv", index=False)
    print("\nSaved results/reproduced_benchmark_table.csv")
    return df

if __name__ == "__main__":
    run_simulation()
