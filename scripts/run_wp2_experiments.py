"""
run_wp2_experiments.py: Complete Work Package 2 pipeline.
Conducts Uniformity tests, Proposition 1 Gibbs thinning MCMC, Selection-bias MH MCMC,
Gelman-Rubin split-R_hat, ESS, Savage-Dickey test, exact 2D grid validation, and publication figures.
"""

import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kstest, uniform
from src.data_loader import load_d1_invoices
from src.diagnostics import (
    compute_acf,
    compute_ess,
    compute_split_rhat,
    total_variation_distance,
)
from src.mcmc_selection import SelectionBiasMCMC
from src.mcmc_thinning import ThinningGibbsSampler

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.dpi": 150})


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("==============================================================")
    print("      WORK PACKAGE 2: THINNING, SELECTION BIAS & MCMC         ")
    print("==============================================================")

    # 1. Load Data
    dense_labels, raw_labels, thinned_labels, meta = load_d1_invoices()
    n_true = meta["n_true"]
    k_thinned = meta["thinned_count"]
    p_hat_true = meta["capture_fraction"]
    m_thinned = int(np.max(thinned_labels))

    print(
        f"Data Loaded: Population N_true = {n_true:,} | Natural Thinned Sample S = {k_thinned:,} invoices"
    )
    print(
        f"True capture fraction p_hat = {p_hat_true:.4f} | Sample maximum m = {m_thinned:,}"
    )

    # -------------------------------------------------------------
    # PHASE 1: Uniformity Testing of Recorded Customer Invoices
    # -------------------------------------------------------------
    print("\n>>> Phase 1: Uniformity Diagnostics on Natural Sample S...")
    # Normalize by sample maximum
    u_vals = thinned_labels / float(m_thinned)
    ks_stat, ks_pval = kstest(u_vals, "uniform")
    obs_min = int(np.min(thinned_labels))
    exp_min = float(n_true + 1) / float(k_thinned + 1)

    # Spacings
    spacings = np.diff(np.sort(thinned_labels))
    mean_spacing = float(np.mean(spacings))
    exp_spacing = float(n_true) / float(k_thinned)

    uniformity_results = {
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "obs_min": obs_min,
        "exp_min": exp_min,
        "mean_spacing": mean_spacing,
        "exp_spacing": exp_spacing,
    }
    pd.DataFrame([uniformity_results]).to_csv(
        "results/wp2_uniformity_diagnostics.csv", index=False
    )
    print(f"KS test on s_i / m: stat = {ks_stat:.5f}, p-value = {ks_pval:.4e}")
    print(f"Observed minimum = {obs_min} vs Theoretical Expected = {exp_min:.2f}")
    print(f"Mean spacing = {mean_spacing:.4f} vs Expected = {exp_spacing:.4f}")

    # -------------------------------------------------------------
    # PHASE 2: Proposition 1 Gibbs Sampler on Thinning Model
    # -------------------------------------------------------------
    print("\n>>> Phase 2: Fitting Proposition 1 Gibbs Thinning Sampler...")
    # We evaluate on natural sample and subsamples (e.g. k=100 from thinned to mimic scarcity)
    # Using 4 dispersed chains to verify convergence
    np.random.seed(42)
    sub_sample_100 = np.sort(
        np.random.choice(thinned_labels, size=100, replace=False)
    )
    m_sub = int(np.max(sub_sample_100))
    k_sub = 100

    print(
        f"Fitting 4 dispersed chains on thinned subset (k={k_sub}, m={m_sub:,})..."
    )
    sampler = ThinningGibbsSampler(
        m=m_sub, k=k_sub, a=1.0, b=1.0, alpha=1.0, beta=1e-5
    )

    n_iter = 6000
    burn_in = 1500
    chains_N = []
    chains_p = []
    chains_mu = []

    init_configs = [
        {"init_n": m_sub, "init_p": 0.05, "init_mu": m_sub * 1.0},
        {"init_n": int(m_sub * 1.2), "init_p": 0.01, "init_mu": m_sub * 1.2},
        {"init_n": int(m_sub * 1.5), "init_p": 0.005, "init_mu": m_sub * 1.5},
        {"init_n": int(m_sub * 1.8), "init_p": 0.001, "init_mu": m_sub * 2.0},
    ]

    t0_gibbs = time.time()
    for c_idx, cfg in enumerate(init_configs):
        res = sampler.sample(
            n_iter=n_iter,
            burn_in=burn_in,
            init_n=cfg["init_n"],
            init_p=cfg["init_p"],
            init_mu=cfg["init_mu"],
            seed=42 + c_idx,
        )
        chains_N.append(res["N"])
        chains_p.append(res["p"])
        chains_mu.append(res["mu"])
    time_gibbs = time.time() - t0_gibbs

    # Diagnostics
    rhat_N = compute_split_rhat(chains_N)
    rhat_p = compute_split_rhat(chains_p)
    rhat_mu = compute_split_rhat(chains_mu)

    ess_N = compute_ess(chains_N)
    ess_p = compute_ess(chains_p)
    ess_mu = compute_ess(chains_mu)

    print(f"Gibbs Sampler finished in {time_gibbs:.2f}s across 4 chains:")
    print(f"  N:  Split-R_hat = {rhat_N:.4f} | ESS = {ess_N:.1f} ({ess_N/time_gibbs:.1f} ESS/sec)")
    print(f"  p:  Split-R_hat = {rhat_p:.4f} | ESS = {ess_p:.1f} ({ess_p/time_gibbs:.1f} ESS/sec)")
    print(f"  mu: Split-R_hat = {rhat_mu:.4f} | ESS = {ess_mu:.1f} ({ess_mu/time_gibbs:.1f} ESS/sec)")

    all_N = np.concatenate(chains_N)
    all_p = np.concatenate(chains_p)
    all_mu = np.concatenate(chains_mu)
    print(f"Posterior N: mean = {np.mean(all_N):.1f}, 95% CI = [{np.percentile(all_N, 2.5):.0f}, {np.percentile(all_N, 97.5):.0f}] (True = {n_true:,})")
    print(f"Posterior p: mean = {np.mean(all_p):.5f}, 95% CI = [{np.percentile(all_p, 2.5):.5f}, {np.percentile(all_p, 97.5):.5f}]")

    # Posterior predictive check for thinning model: simulate k_rep ~ Binomial(N, p)
    k_rep = np.random.binomial(all_N, all_p)
    print(f"Posterior Predictive Check: mean k_rep = {np.mean(k_rep):.1f} vs observed k = {k_sub}")

    # -------------------------------------------------------------
    # PHASE 3: Selection Bias Model & Exact 2D Grid Validation
    # -------------------------------------------------------------
    print("\n>>> Phase 3: Fitting Metropolis-within-Gibbs Selection Bias Model...")
    # Use a representative sample of k=25 to allow both MCMC and exact 2D grid evaluation
    sub_sample_25 = np.sort(
        np.random.choice(thinned_labels, size=25, replace=False)
    )
    m_sel = int(np.max(sub_sample_25))
    omega_sel = 35000

    print(
        f"Fitting Selection Bias MCMC (k=25, m={m_sel:,}, omega={omega_sel:,})..."
    )
    sel_mcmc = SelectionBiasMCMC(
        sub_sample_25,
        prior_n="flat",
        omega=omega_sel,
        sigma_theta=0.5,
        proposal_sd=0.08,
    )

    chains_sel_N = []
    chains_sel_th = []
    t0_sel = time.time()
    for c_idx in range(4):
        init_th = [-0.3, -0.1, 0.0, 0.2][c_idx]
        init_n = [m_sel, int(m_sel * 1.05), int(m_sel * 1.1), int(m_sel * 1.15)][
            c_idx
        ]
        res_sel = sel_mcmc.sample(
            n_iter=6000,
            burn_in=2000,
            init_n=init_n,
            init_theta=init_th,
            seed=100 + c_idx,
        )
        chains_sel_N.append(res_sel["N"])
        chains_sel_th.append(res_sel["theta"])
    time_sel = time.time() - t0_sel

    rhat_sel_N = compute_split_rhat(chains_sel_N)
    rhat_sel_th = compute_split_rhat(chains_sel_th)
    ess_sel_N = compute_ess(chains_sel_N)
    ess_sel_th = compute_ess(chains_sel_th)

    all_sel_N = np.concatenate(chains_sel_N)
    all_sel_th = np.concatenate(chains_sel_th)

    # Savage-Dickey test for theta = 0 (no selection bias)
    bf_01, post_0, prior_0 = sel_mcmc.compute_savage_dickey_bf(all_sel_th)
    print(f"Selection MCMC finished in {time_sel:.2f}s across 4 chains:")
    print(f"  N:     Split-R_hat = {rhat_sel_N:.4f} | ESS = {ess_sel_N:.1f}")
    print(f"  theta: Split-R_hat = {rhat_sel_th:.4f} | ESS = {ess_sel_th:.1f}")
    print(f"  Posterior theta: mean = {np.mean(all_sel_th):.4f}, 95% CI = [{np.percentile(all_sel_th, 2.5):.4f}, {np.percentile(all_sel_th, 97.5):.4f}]")
    print(f"  Savage-Dickey BF_01 (theta=0 vs theta!=0) = {bf_01:.3f} (posterior/prior: {post_0:.3f}/{prior_0:.3f})")

    # Exact 2D Grid Validation
    print("Computing exact 2D grid posterior for validation...")
    theta_grid = np.linspace(-0.6, 0.6, 60)
    n_grid = np.arange(m_sel, omega_sel + 1, 100, dtype=np.int64)  # step of 100
    grid_post = sel_mcmc.compute_exact_2d_grid(theta_grid, n_grid)

    # Bivariate histogram of MCMC samples on the same grid
    mcmc_hist, _, _ = np.histogram2d(
        all_sel_th,
        all_sel_N,
        bins=[
            np.linspace(-0.6, 0.6, 61),
            np.append(n_grid, n_grid[-1] + 100),
        ],
    )
    mcmc_hist_prob = mcmc_hist / np.sum(mcmc_hist)
    tv_dist = total_variation_distance(mcmc_hist_prob.flatten(), grid_post.flatten())
    print(f"Total Variation Distance (MCMC vs Exact 2D Grid): TV = {tv_dist:.4f}")

    # Save summary table
    mcmc_summary = [
        {
            "Model": "Proposition 1 Thinning Gibbs",
            "Parameter": "N",
            "Mean": np.mean(all_N),
            "SD": np.std(all_N),
            "CI_2.5": np.percentile(all_N, 2.5),
            "CI_97.5": np.percentile(all_N, 97.5),
            "Split_Rhat": rhat_N,
            "ESS": ess_N,
            "ESS_per_sec": ess_N / time_gibbs,
        },
        {
            "Model": "Proposition 1 Thinning Gibbs",
            "Parameter": "p",
            "Mean": np.mean(all_p),
            "SD": np.std(all_p),
            "CI_2.5": np.percentile(all_p, 2.5),
            "CI_97.5": np.percentile(all_p, 97.5),
            "Split_Rhat": rhat_p,
            "ESS": ess_p,
            "ESS_per_sec": ess_p / time_gibbs,
        },
        {
            "Model": "Proposition 1 Thinning Gibbs",
            "Parameter": "mu",
            "Mean": np.mean(all_mu),
            "SD": np.std(all_mu),
            "CI_2.5": np.percentile(all_mu, 2.5),
            "CI_97.5": np.percentile(all_mu, 97.5),
            "Split_Rhat": rhat_mu,
            "ESS": ess_mu,
            "ESS_per_sec": ess_mu / time_gibbs,
        },
        {
            "Model": "Selection Bias MH-within-Gibbs",
            "Parameter": "N",
            "Mean": np.mean(all_sel_N),
            "SD": np.std(all_sel_N),
            "CI_2.5": np.percentile(all_sel_N, 2.5),
            "CI_97.5": np.percentile(all_sel_N, 97.5),
            "Split_Rhat": rhat_sel_N,
            "ESS": ess_sel_N,
            "ESS_per_sec": ess_sel_N / time_sel,
        },
        {
            "Model": "Selection Bias MH-within-Gibbs",
            "Parameter": "theta",
            "Mean": np.mean(all_sel_th),
            "SD": np.std(all_sel_th),
            "CI_2.5": np.percentile(all_sel_th, 2.5),
            "CI_97.5": np.percentile(all_sel_th, 97.5),
            "Split_Rhat": rhat_sel_th,
            "ESS": ess_sel_th,
            "ESS_per_sec": ess_sel_th / time_sel,
        },
    ]
    pd.DataFrame(mcmc_summary).to_csv("results/wp2_mcmc_summary.csv", index=False)
    print("Saved results/wp2_mcmc_summary.csv")

    # -------------------------------------------------------------
    # PHASE 4: Generate Publication Figures
    # -------------------------------------------------------------
    print("\n>>> Phase 4: Generating Publication Figures...")
    generate_wp2_figures(
        u_vals,
        spacings,
        chains_N,
        chains_p,
        all_N,
        all_p,
        k_rep,
        k_sub,
        n_true,
        chains_sel_th,
        chains_sel_N,
        all_sel_th,
        all_sel_N,
        theta_grid,
        n_grid,
        grid_post,
        bf_01,
        post_0,
        prior_0,
        tv_dist,
    )
    print("All WP2 figures successfully created in figures/")


def generate_wp2_figures(
    u_vals,
    spacings,
    chains_N,
    chains_p,
    all_N,
    all_p,
    k_rep,
    k_sub,
    n_true,
    chains_sel_th,
    chains_sel_N,
    all_sel_th,
    all_sel_N,
    theta_grid,
    n_grid,
    grid_post,
    bf_01,
    post_0,
    prior_0,
    tv_dist,
):
    # Figure 1: Uniformity Diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Empirical CDF vs Uniform
    sorted_u = np.sort(u_vals)
    y_vals = np.linspace(0, 1, len(sorted_u))
    axes[0].plot(sorted_u, y_vals, label="Empirical CDF (s_i / m)", color="tab:blue", linewidth=2)
    axes[0].plot([0, 1], [0, 1], label="Theoretical Uniform U(0,1)", color="tab:red", linestyle="--")
    axes[0].set_title("Kolmogorov-Smirnov Uniformity Check on Invoices")
    axes[0].set_xlabel("Normalized Serial Number s_i / m")
    axes[0].set_ylabel("Cumulative Probability")
    axes[0].legend()

    # Spacing distribution
    axes[1].hist(spacings, bins=50, density=True, color="tab:gray", alpha=0.7, label="Empirical Spacings (s_i - s_{i-1})")
    mean_sp = np.mean(spacings)
    x_sp = np.linspace(1, 20, 200)
    axes[1].plot(x_sp, (1.0 / mean_sp) * np.exp(-x_sp / mean_sp), color="tab:red", linewidth=2, label="Geometric/Exp Fit")
    axes[1].set_title(f"Label Spacing Distribution (Mean = {mean_sp:.2f})")
    axes[1].set_xlabel("Spacing between Consecutive Observed Invoices")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(0, 20)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("figures/wp2_uniformity_diagnostics.png")
    plt.close()

    # Figure 2: Gibbs Thinning Traces & Posteriors
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for c_i, ch in enumerate(chains_N):
        axes[0, 0].plot(ch[:1000], color=colors[c_i], alpha=0.7, label=f"Chain {c_i+1}" if c_i == 0 else "")
    axes[0, 0].axhline(n_true, color="red", linestyle="--", label=f"True N ({n_true:,})")
    axes[0, 0].set_title("Proposition 1 Gibbs Sampler: Trace Plot for Population Size N")
    axes[0, 0].set_xlabel("Iteration (post burn-in)")
    axes[0, 0].set_ylabel("N")
    axes[0, 0].legend()

    for c_i, ch in enumerate(chains_p):
        axes[0, 1].plot(ch[:1000], color=colors[c_i], alpha=0.7)
    axes[0, 1].set_title("Gibbs Sampler: Trace Plot for Capture Probability p")
    axes[0, 1].set_xlabel("Iteration (post burn-in)")
    axes[0, 1].set_ylabel("p")

    # Posterior distribution of N
    axes[1, 0].hist(all_N, bins=40, density=True, color="tab:blue", alpha=0.6, label="Posterior pi(N | S)")
    axes[1, 0].axvline(n_true, color="red", linestyle="--", linewidth=2, label=f"True N={n_true:,}")
    axes[1, 0].set_title("Posterior Density of Population Size N")
    axes[1, 0].set_xlabel("N")
    axes[1, 0].legend()

    # Posterior predictive check of k
    axes[1, 1].hist(k_rep, bins=30, density=True, color="teal", alpha=0.6, label="Simulated k_rep")
    axes[1, 1].axvline(k_sub, color="red", linestyle="--", linewidth=2.5, label=f"Observed k={k_sub}")
    axes[1, 1].set_title("Posterior Predictive Check: Simulated Sample Size k_rep vs Observed")
    axes[1, 1].set_xlabel("k_rep")
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig("figures/wp2_thinning_mcmc.png")
    plt.close()

    # Figure 3: Selection Bias MCMC vs Exact 2D Grid
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: 2D Contour MCMC vs Grid
    # MCMC density via KDE or 2D scatter
    axes[0].contour(n_grid, theta_grid, grid_post, levels=7, colors="red", linewidths=1.5)
    axes[0].scatter(all_sel_N[::10], all_sel_th[::10], s=3, color="tab:blue", alpha=0.3, label="MCMC Samples (every 10th)")
    # Dummy line for legend
    axes[0].plot([], [], color="red", label=f"Exact 2D Grid Contours (TV={tv_dist:.3f})")
    axes[0].set_title("Selection Model: MCMC Samples vs Exact 2D Grid Posteriors")
    axes[0].set_xlabel("Population Size N")
    axes[0].set_ylabel(r"Selection Parameter $\theta$ ($w(s) = s^{-\theta}$)")
    axes[0].legend()

    # Right: Savage-Dickey density ratio plot
    from scipy.stats import norm
    th_axis = np.linspace(-0.6, 0.6, 200)
    prior_curve = norm.pdf(th_axis, loc=0.0, scale=0.5)
    axes[1].hist(all_sel_th, bins=50, density=True, color="tab:blue", alpha=0.5, label=r"Posterior $\pi(\theta \mid S)$")
    axes[1].plot(th_axis, prior_curve, color="black", linestyle="--", linewidth=2, label=r"Prior $\pi(\theta)$")
    axes[1].scatter([0.0], [post_0], color="tab:blue", s=70, zorder=5, label=f"Posterior at 0 ({post_0:.2f})")
    axes[1].scatter([0.0], [prior_0], color="black", s=70, zorder=5, label=f"Prior at 0 ({prior_0:.2f})")
    axes[1].axvline(0.0, color="gray", linestyle=":", alpha=0.6)
    axes[1].set_title(r"Savage-Dickey Test for No Selection Bias ($H_0: \theta = 0$)")
    axes[1].set_xlabel(r"Selection Bias Parameter $\theta$")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/wp2_selection_vs_exact.png")
    plt.close()


if __name__ == "__main__":
    main()
