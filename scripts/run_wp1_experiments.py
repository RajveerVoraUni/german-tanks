"""
run_wp1_experiments.py: Full pipeline runner for WP1 experiments, data persistence, and plotting.
"""

import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.bayes_factors import legal_cap_bayes_factor, lindley_jeffreys_sweep
from src.data_loader import load_d1_invoices
from src.evaluator import run_wp1_simulation

# Set style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.dpi": 150})


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("==============================================================")
    print("           WORK PACKAGE 1: BAYESIAN GERMAN TANK PIPELINE      ")
    print("==============================================================")

    # 1. Load Data
    dense_labels, raw_labels, thinned_labels, meta = load_d1_invoices()
    n_true = meta["n_true"]
    print(f"Data Loaded: N_true = {n_true:,} dense invoices.")
    print(
        f"Raw invoice range: {meta['min_inv']} to {meta['max_inv']} (Gaps: {meta['gap_count']:,}, {meta['gap_fraction']*100:.2f}%)"
    )

    k_list = [5, 10, 20, 50, 100]
    r_reps = 2000

    # 2. Run Dense Labels Simulation Study
    print("\n>>> Phase 1: Evaluating Dense Labels (Ideal Textbook Mechanism)...")
    df_dense = run_wp1_simulation(
        dense_labels,
        n_true,
        k_list=k_list,
        r_replicates=r_reps,
        seed=42,
        label_type="dense",
    )
    df_dense.to_csv("results/wp1_dense_summary.csv", index=False)
    print("Saved results/wp1_dense_summary.csv")

    # 3. Run Raw Labels Simulation Study (Measuring Gap Degradation)
    print("\n>>> Phase 2: Evaluating Raw Labels (Measuring Gap Degradation)...")
    # For raw labels, the true population of items is still N_true=22,061, but the labels have gaps up to 45,223!
    df_raw = run_wp1_simulation(
        raw_labels,
        n_true,
        k_list=k_list,
        r_replicates=r_reps,
        seed=42,
        label_type="raw_gapped",
    )
    df_raw.to_csv("results/wp1_raw_summary.csv", index=False)
    print("Saved results/wp1_raw_summary.csv")

    # 4. Bayes Factors Evaluation for Legal Cap
    print("\n>>> Phase 3: Evaluating Bayes Factors for H0: N <= L vs H1: N > L...")
    l_multipliers = [0.5, 0.9, 1.0, 1.1]
    bf_records = []

    for k in k_list:
        for mult in l_multipliers:
            L_val = int(mult * n_true)
            # Sample R=500 replicates to estimate Power / False Alarm rate
            h1_detections = 0
            bf_list = []
            for _ in range(500):
                sample = np.random.choice(dense_labels, size=k, replace=False)
                m = int(np.max(sample))
                bf, log_bf, p_h1 = legal_cap_bayes_factor(
                    m, k, L_val, omega=100000, prior_type="uniform"
                )
                bf_list.append(bf)
                # Decision threshold: choose H1 if BF_10 > 1 (or log_BF > 0)
                if log_bf > 0:
                    h1_detections += 1

            p_detect = h1_detections / 500.0
            # Classification
            if mult >= 1.0:
                metric_name = "False Alarm Rate (Type I error)"
            else:
                metric_name = "Power (1 - Type II error)"

            bf_records.append(
                {
                    "k": k,
                    "L_mult": mult,
                    "L_val": L_val,
                    "detection_prob": p_detect,
                    "metric_type": metric_name,
                    "median_log_bf": float(
                        np.median([b for b in bf_list if np.isfinite(b)])
                    )
                    if any(np.isfinite(bf_list))
                    else 100.0,
                }
            )

    df_bf = pd.DataFrame(bf_records)
    df_bf.to_csv("results/wp1_bayes_factors.csv", index=False)
    print("Saved results/wp1_bayes_factors.csv")

    # 5. Lindley-Jeffreys Prior Width Sweep
    print("\n>>> Phase 4: Demonstrating Lindley-Jeffreys Paradox...")
    # Fix sample at k=20, m = 21,500, L = 20,000 (true N is definitely > L)
    m_fixed, k_fixed, L_fixed = 21500, 20, 20000
    omega_range = np.logspace(np.log10(25000), np.log10(2000000), num=30)
    lj_results = lindley_jeffreys_sweep(m_fixed, k_fixed, L_fixed, omega_range)
    df_lj = pd.DataFrame(lj_results)
    df_lj.to_csv("results/wp1_lindley_jeffreys.csv", index=False)

    # 6. Generate Publication Figures
    print("\n>>> Phase 5: Generating Publication Figures...")
    generate_figures(df_dense, df_raw, df_bf, df_lj)
    print("All figures successfully created in figures/")


def generate_figures(df_dense, df_raw, df_bf, df_lj):
    # Figure 1: Bias and RMSE comparison (Dense vs UMVUE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter to main representative methods
    methods_show = [
        ("Frequentist UMVUE", "UMVUE", "black", "--"),
        ("Bayes (Flat)", "Posterior Mean (SE loss)", "tab:blue", "-"),
        ("Bayes (Flat)", "Posterior Median (Abs loss)", "tab:cyan", ":"),
        ("Bayes (1/n (Scale-Inv))", "Posterior Mean (SE loss)", "tab:green", "-"),
        ("Bayes (Poisson (mu=N_true))", "Posterior Mean (SE loss)", "tab:purple", "-"),
        (
            "Bayes (Poisson (mu=0.5*N_true))",
            "Posterior Mean (SE loss)",
            "tab:red",
            "-.",
        ),
    ]

    for meth, est, color, ls in methods_show:
        sub = df_dense[(df_dense["method"] == meth) & (df_dense["estimator"] == est)]
        if not sub.empty:
            label_str = f"{meth} ({est.split()[0]})"
            axes[0].plot(
                sub["k"],
                sub["bias"],
                marker="o",
                color=color,
                linestyle=ls,
                label=label_str,
                linewidth=2,
            )
            axes[1].plot(
                sub["k"],
                sub["rmse"],
                marker="s",
                color=color,
                linestyle=ls,
                label=label_str,
                linewidth=2,
            )

    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_title("Estimator Bias vs Sample Size k (N=22,061)")
    axes[0].set_xlabel("Sample size k")
    axes[0].set_ylabel("Bias (Estimated - True)")
    axes[0].legend(fontsize=9)

    axes[1].set_title("Estimator RMSE vs Sample Size k")
    axes[1].set_xlabel("Sample size k")
    axes[1].set_ylabel("Root Mean Squared Error")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/wp1_bias_rmse.png")
    plt.close()

    # Figure 2: Coverage and Width Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    coverage_methods = [
        ("Frequentist UMVUE", "UMVUE", "Frequentist 95% CI", "black", "--"),
        ("Bayes (Flat)", "Posterior Mean (SE loss)", "Bayes Flat 95% HPD", "tab:blue", "-"),
        ("Bayes (Flat)", "Posterior Median (Abs loss)", "Bayes Flat 95% Equal-Tailed", "tab:cyan", ":"),
        ("Bayes (1/n (Scale-Inv))", "Posterior Mean (SE loss)", "Bayes 1/n 95% HPD", "tab:green", "-"),
        ("Bayes (Poisson (mu=0.5*N_true))", "Posterior Mean (SE loss)", "Bayes Misspecified Poisson HPD", "tab:red", "-."),
    ]

    for meth, est, label_str, color, ls in coverage_methods:
        sub = df_dense[(df_dense["method"] == meth) & (df_dense["estimator"] == est)]
        if not sub.empty:
            axes[0].plot(
                sub["k"],
                sub["coverage_95"],
                marker="o",
                color=color,
                linestyle=ls,
                label=label_str,
                linewidth=2,
            )
            axes[1].plot(
                sub["k"],
                sub["mean_interval_width"],
                marker="s",
                color=color,
                linestyle=ls,
                label=label_str,
                linewidth=2,
            )

    axes[0].axhline(0.95, color="red", linestyle="--", label="Nominal 95%")
    axes[0].set_title("Empirical 95% Interval Coverage across R=2,000 Replicates")
    axes[0].set_xlabel("Sample size k")
    axes[0].set_ylabel("Empirical Coverage")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Mean 95% Interval Width vs Sample Size k")
    axes[1].set_xlabel("Sample size k")
    axes[1].set_ylabel("Interval Width (Upper - Lower)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/wp1_coverage_width.png")
    plt.close()

    # Figure 3: Gaps Impact (Dense vs Raw Gapped)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dense_flat = df_dense[
        (df_dense["method"] == "Bayes (Flat)")
        & (df_dense["estimator"] == "Posterior Mean (SE loss)")
    ]
    raw_flat = df_raw[
        (df_raw["method"] == "Bayes (Flat)")
        & (df_raw["estimator"] == "Posterior Mean (SE loss)")
    ]

    axes[0].plot(
        dense_flat["k"],
        dense_flat["rmse"],
        marker="o",
        color="tab:blue",
        label="Dense Labels (No Gaps)",
        linewidth=2,
    )
    axes[0].plot(
        raw_flat["k"],
        raw_flat["rmse"],
        marker="s",
        color="tab:red",
        label="Raw Gapped Labels (51.2% gaps)",
        linewidth=2,
    )
    axes[0].set_title("Impact of Missing Gaps on RMSE")
    axes[0].set_xlabel("Sample size k")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()

    axes[1].plot(
        dense_flat["k"],
        dense_flat["coverage_95"],
        marker="o",
        color="tab:blue",
        label="Dense Labels (No Gaps)",
        linewidth=2,
    )
    axes[1].plot(
        raw_flat["k"],
        raw_flat["coverage_95"],
        marker="s",
        color="tab:red",
        label="Raw Gapped Labels (Coverage Crash!)",
        linewidth=2,
    )
    axes[1].axhline(0.95, color="gray", linestyle="--", label="Nominal 95%")
    axes[1].set_title("Coverage Breakdown Under Missing Gaps")
    axes[1].set_xlabel("Sample size k")
    axes[1].set_ylabel("95% HPD Coverage")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("figures/wp1_gaps_impact.png")
    plt.close()

    # Figure 4: Bayes Factors & Lindley-Jeffreys Paradox
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Detection Probability across L
    for mult in df_bf["L_mult"].unique():
        sub_bf = df_bf[df_bf["L_mult"] == mult]
        lbl = (
            f"L = {mult}*N_true ({'Power' if mult < 1.0 else 'False Alarm'})"
        )
        axes[0].plot(
            sub_bf["k"],
            sub_bf["detection_prob"],
            marker="o",
            label=lbl,
            linewidth=2,
        )

    axes[0].set_title("Bayes Factor Decision Power & False Alarm (H1: N > L)")
    axes[0].set_xlabel("Sample size k")
    axes[0].set_ylabel("P(Choose H1 | Data)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()

    # Right: Lindley-Jeffreys Paradox plot
    axes[1].plot(
        df_lj["omega"],
        df_lj["p_h1_post"],
        color="tab:red",
        linewidth=2.5,
        label="P(N > 20,000 | m=21,500, k=20)",
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Lindley-Jeffreys Paradox: P(H1|Data) vs Prior Width Ω")
    axes[1].set_xlabel("Prior Upper Bound Ω (Log scale)")
    axes[1].set_ylabel("Posterior Probability P(H1 | S)")
    axes[1].axhline(0.5, color="gray", linestyle=":")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/wp1_bayes_factors.png")
    plt.close()


if __name__ == "__main__":
    main()
