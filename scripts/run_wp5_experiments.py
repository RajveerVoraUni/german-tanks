"""
run_wp5_experiments.py: Full pipeline for Work Package 5.
Normal-Inverse-Gamma regression, Spike-and-Slab variable selection (PIPs),
Ridge & LASSO comparison, Hierarchical Bayes vs Empirical Bayes partial pooling,
and temporal validation across software releases.
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
from scipy.stats import norm
from sklearn.linear_model import LassoCV, RidgeCV
from src.bayesian_regression import (
    ConjugateNormalInvGamma,
    SpikeAndSlabGibbs,
    fit_empirical_bayes_hierarchical,
)

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.dpi": 150})


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("==============================================================")
    print("      WORK PACKAGE 5: BAYESIAN REGRESSION & SPIKE-AND-SLAB    ")
    print("==============================================================")

    # 1. Load Primary Dataset: Ant 1.7 with 20 CK metrics
    csv_path = "data/d4_promise/ck_metrics/ant-1.7.csv"
    df_ant17 = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}: {df_ant17.shape[0]} classes, {df_ant17.shape[1]} columns")

    feature_cols = [
        "wmc",
        "dit",
        "noc",
        "cbo",
        "rfc",
        "lcom",
        "ca",
        "ce",
        "npm",
        "lcom3",
        "loc",
        "dam",
        "moa",
        "mfa",
        "cam",
        "ic",
        "cbm",
        "amc",
        "max_cc",
        "avg_cc",
    ]

    # Preprocess: y = log(1 + bug), X = log(1 + X) standardized
    y = np.log1p(df_ant17["bug"].values)
    X_raw = df_ant17[feature_cols].values
    X_log = np.log1p(np.maximum(0, X_raw))
    X_mean = np.mean(X_log, axis=0)
    X_std = np.std(X_log, axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X_log - X_mean) / X_std

    # Add intercept column
    X = np.hstack([np.ones((len(y), 1)), X_norm])
    feature_names = ["Intercept"] + feature_cols

    # -------------------------------------------------------------
    # PHASE 1: Conjugate Normal-Inverse-Gamma Regression & Bayes Factors
    # -------------------------------------------------------------
    print("\n>>> Phase 1: Fitting Conjugate Normal-Inverse-Gamma Regression...")
    model_full = ConjugateNormalInvGamma(X, y, v0_scale=10.0, a0=2.0, b0=2.0)
    log_marg_full = model_full.log_marginal_likelihood()
    beta_mean = model_full.posterior_mean_beta()
    sigma2_mean = model_full.posterior_mean_sigma2()

    print(f"Full Model Log Marginal Likelihood: {log_marg_full:.2f}")
    print(f"Posterior mean sigma^2: {sigma2_mean:.4f}")

    # Subsets for Bayes Factors: Size metrics vs Complexity vs Full
    # Size metrics: loc, amc (indices in X: 11, 18)
    idx_size = [0, 11, 18]
    model_size = ConjugateNormalInvGamma(
        X[:, idx_size], y, v0_scale=10.0, a0=2.0, b0=2.0
    )
    log_marg_size = model_size.log_marginal_likelihood()

    # Complexity metrics: wmc, rfc, cbo, max_cc (indices 1, 4, 5, 19)
    idx_complex = [0, 1, 4, 5, 19]
    model_complex = ConjugateNormalInvGamma(
        X[:, idx_complex], y, v0_scale=10.0, a0=2.0, b0=2.0
    )
    log_marg_complex = model_complex.log_marginal_likelihood()

    bf_full_vs_size = np.exp(log_marg_full - log_marg_size)
    bf_complex_vs_size = np.exp(log_marg_complex - log_marg_size)
    print(
        f"Bayes Factor (Full vs Size-Only): BF = {bf_full_vs_size:.2e} (ln BF = {log_marg_full - log_marg_size:.2f})"
    )
    print(
        f"Bayes Factor (Complexity vs Size-Only): BF = {bf_complex_vs_size:.2e} (ln BF = {log_marg_complex - log_marg_size:.2f})"
    )

    # -------------------------------------------------------------
    # PHASE 2: Collapsed Gibbs Spike-and-Slab Variable Selection
    # -------------------------------------------------------------
    print(
        "\n>>> Phase 2: Running Collapsed Gibbs Spike-and-Slab Variable Selection..."
    )
    t0_ss = time.time()
    ss_sampler = SpikeAndSlabGibbs(
        X, y, c_slab=10.0, a0=2.0, b0=2.0, a_w=1.0, b_w=1.0
    )
    res_ss = ss_sampler.sample(n_iter=4000, burn_in=1000, seed=42)
    time_ss = time.time() - t0_ss
    pips = res_ss["pip"]
    print(f"Spike-and-Slab completed in {time_ss:.2f}s.")

    # Top predictors by PIP
    pip_df = pd.DataFrame(
        {
            "Feature": feature_names[1:],
            "PIP": pips[1:],
            "Beta_Conjugate": beta_mean[1:],
        }
    ).sort_values("PIP", ascending=False)
    print("Top 8 Selected Features by Posterior Inclusion Probability (PIP):")
    print(pip_df.head(8).to_string(index=False))
    pip_df.to_csv("results/wp5_spike_and_slab_pips.csv", index=False)

    # -------------------------------------------------------------
    # PHASE 3: Comparison with LASSO and Ridge (MAP Estimates)
    # -------------------------------------------------------------
    print("\n>>> Phase 3: Benchmarking against LASSO and Ridge MAP Estimates...")
    # Standard 10-fold CV for LASSO and Ridge
    lasso_cv = LassoCV(cv=10, random_state=42).fit(X[:, 1:], y)
    ridge_cv = RidgeCV(cv=10).fit(X[:, 1:], y)

    comp_df = pip_df.copy()
    comp_df["LASSO_Coef"] = lasso_cv.coef_[
        [feature_cols.index(f) for f in comp_df["Feature"]]
    ]
    comp_df["Ridge_Coef"] = ridge_cv.coef_[
        [feature_cols.index(f) for f in comp_df["Feature"]]
    ]
    comp_df.to_csv("results/wp5_model_comparison.csv", index=False)
    print("Saved results/wp5_model_comparison.csv")

    # -------------------------------------------------------------
    # PHASE 4: Hierarchical Bayes & Empirical Bayes across Releases
    # -------------------------------------------------------------
    print(
        "\n>>> Phase 4: Fitting Empirical Bayes Hierarchical Partial Pooling..."
    )
    ant_releases = [
        "ant-1.3.csv",
        "ant-1.4.csv",
        "ant-1.5.csv",
        "ant-1.6.csv",
        "ant-1.7.csv",
    ]
    groups_y = []
    groups_X = []
    rel_names = []

    for rel in ant_releases:
        fpath = f"data/d4_promise/ck_metrics/{rel}"
        df_r = pd.read_csv(fpath)
        groups_y.append(np.log1p(df_r["bug"].values))
        groups_X.append(df_r[feature_cols].values)
        rel_names.append(rel.replace(".csv", ""))

    eb_res = fit_empirical_bayes_hierarchical(groups_y, groups_X)
    print(f"Hierarchical Grand Mean mu_hat = {eb_res['mu_hat']:.4f}")
    print(f"Within-Release Variance sigma^2 = {eb_res['sigma2_hat']:.4f}")
    print(
        f"Between-Release Variance sigma_alpha^2 = {eb_res['sigma_alpha2_hat']:.4f}"
    )

    eb_df = pd.DataFrame(
        {
            "Release": rel_names,
            "Sample_Size_N": eb_res["n_r"],
            "Observed_Mean_y": eb_res["y_bars"],
            "EB_Shrunk_Mean": eb_res["shrunk_alphas"],
            "Shrinkage_Weight_B": eb_res["shrinkage_factors"],
        }
    )
    print(eb_df.to_string(index=False))
    eb_df.to_csv("results/wp5_hierarchical_eb.csv", index=False)

    # -------------------------------------------------------------
    # PHASE 5: Out-of-Sample Temporal Validation (Ant 1.6 -> Ant 1.7)
    # -------------------------------------------------------------
    print(
        "\n>>> Phase 5: Temporal Validation (Train on Ant 1.6, Test on Ant 1.7)..."
    )
    df_16 = pd.read_csv("data/d4_promise/ck_metrics/ant-1.6.csv")
    y_train = np.log1p(df_16["bug"].values)
    X_train_raw = df_16[feature_cols].values
    X_train_norm = (np.log1p(np.maximum(0, X_train_raw)) - X_mean) / X_std
    X_train = np.hstack([np.ones((len(y_train), 1)), X_train_norm])

    y_test = y
    X_test = X

    # 1. Full Bayesian Conjugate Model on Ant 1.6
    model_train = ConjugateNormalInvGamma(X_train, y_train, v0_scale=10.0)
    pred_mean, pred_var = model_train.predict(X_test)

    # 2. Spike-and-Slab Restricted Model (using features with PIP > 0.5)
    active_idx = [0] + [i + 1 for i, p in enumerate(pips[1:]) if p >= 0.5]
    model_sparse = ConjugateNormalInvGamma(
        X_train[:, active_idx], y_train, v0_scale=10.0
    )
    pred_sparse_mean, pred_sparse_var = model_sparse.predict(
        X_test[:, active_idx]
    )

    # Metrics on test set
    rmse_full = float(np.sqrt(np.mean((y_test - pred_mean) ** 2)))
    log_score_full = float(
        -np.mean(
            norm.logpdf(y_test, loc=pred_mean, scale=np.sqrt(pred_var))
        )
    )

    rmse_sparse = float(np.sqrt(np.mean((y_test - pred_sparse_mean) ** 2)))
    log_score_sparse = float(
        -np.mean(
            norm.logpdf(
                y_test, loc=pred_sparse_mean, scale=np.sqrt(pred_sparse_var)
            )
        )
    )

    val_records = [
        {
            "Model": "Full 20-Metric Conjugate Bayes",
            "Test_RMSE": rmse_full,
            "Test_Log_Score": log_score_full,
        },
        {
            "Model": "Spike-and-Slab Sparse Bayes (PIP > 0.5)",
            "Test_RMSE": rmse_sparse,
            "Test_Log_Score": log_score_sparse,
        },
    ]
    df_val = pd.DataFrame(val_records)
    print(df_val.to_string(index=False))
    df_val.to_csv("results/wp5_temporal_validation.csv", index=False)

    # -------------------------------------------------------------
    # PHASE 6: Publication Figures
    # -------------------------------------------------------------
    print("\n>>> Phase 6: Generating Publication Figures...")
    generate_wp5_figures(comp_df, eb_df, y_test, pred_sparse_mean, feature_names)
    print("All WP5 figures successfully created in figures/")


def generate_wp5_figures(comp_df, eb_df, y_test, pred_test, feature_names):
    # Figure 1: Posterior Inclusion Probabilities (PIPs)
    plt.figure(figsize=(10, 5))
    colors = [
        "tab:red" if p > 0.5 else "tab:blue"
        for p in comp_df["PIP"]
    ]
    bars = plt.bar(comp_df["Feature"], comp_df["PIP"], color=colors, alpha=0.85)
    plt.axhline(0.5, color="red", linestyle="--", label="Median Probability Model (PIP = 0.5)")
    plt.title("Spike-and-Slab Posterior Inclusion Probabilities (Apache Ant 1.7)")
    plt.xlabel("Software Code Complexity Metric")
    plt.ylabel("Inclusion Probability P(γ_j = 1 | Data)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/wp5_spike_and_slab_pips.png")
    plt.close()

    # Figure 2: PIPs vs LASSO vs Ridge Coefficients
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(comp_df["PIP"], np.abs(comp_df["LASSO_Coef"]), color="tab:red", s=50)
    for _, row in comp_df.iterrows():
        if row["PIP"] > 0.4 or abs(row["LASSO_Coef"]) > 0.05:
            axes[0].annotate(row["Feature"], (row["PIP"] + 0.01, abs(row["LASSO_Coef"]) + 0.005), fontsize=9)
    axes[0].set_title("Bayesian PIP vs Absolute LASSO Coefficient")
    axes[0].set_xlabel("Posterior Inclusion Probability (PIP)")
    axes[0].set_ylabel("|LASSO Coefficient|")

    axes[1].scatter(comp_df["PIP"], np.abs(comp_df["Ridge_Coef"]), color="tab:green", s=50)
    for _, row in comp_df.iterrows():
        if row["PIP"] > 0.4 or abs(row["Ridge_Coef"]) > 0.05:
            axes[1].annotate(row["Feature"], (row["PIP"] + 0.01, abs(row["Ridge_Coef"]) + 0.005), fontsize=9)
    axes[1].set_title("Bayesian PIP vs Absolute Ridge Coefficient")
    axes[1].set_xlabel("Posterior Inclusion Probability (PIP)")
    axes[1].set_ylabel("|Ridge Coefficient|")
    plt.tight_layout()
    plt.savefig("figures/wp5_lasso_vs_pips.png")
    plt.close()

    # Figure 3: Empirical Bayes Shrinkage across Releases
    plt.figure(figsize=(8, 5))
    x_pos = np.arange(len(eb_df))
    plt.plot(x_pos, eb_df["Observed_Mean_y"], marker="o", color="tab:gray", linestyle=":", label="Observed Mean log(1+bugs)")
    plt.plot(x_pos, eb_df["EB_Shrunk_Mean"], marker="s", color="tab:blue", linewidth=2.5, label="Empirical Bayes Shrunk Mean")
    plt.axhline(eb_df["Observed_Mean_y"].mean(), color="tab:red", linestyle="--", label="Grand Mean")
    plt.xticks(x_pos, eb_df["Release"])
    plt.title("Empirical Bayes Partial Pooling across Software Releases")
    plt.xlabel("Release")
    plt.ylabel("Mean Defect Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/wp5_hierarchical_shrinkage.png")
    plt.close()

    # Figure 4: Temporal Validation Scatter
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, pred_test, alpha=0.4, color="tab:purple", s=25)
    diag = np.linspace(0, max(y_test), 100)
    plt.plot(diag, diag, color="red", linestyle="--", label="Perfect Calibration")
    plt.title("Out-of-Sample Temporal Prediction: Ant 1.6 Model -> Ant 1.7 Data")
    plt.xlabel("Actual log(1 + bugs)")
    plt.ylabel("Predicted log(1 + bugs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/wp5_temporal_validation.png")
    plt.close()


if __name__ == "__main__":
    main()
