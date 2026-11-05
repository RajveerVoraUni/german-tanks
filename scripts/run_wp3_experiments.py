"""
run_wp3_experiments.py: Full pipeline for Work Package 3.
Dynamic production over time, constrained cumulative volume modeling (N_t >= N_{t-1}),
Normal-Gamma second stage on log-rates, and Student-t predictive forecasting for Online Retail II.
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
from scipy.special import gammaln
from scipy.stats import t as student_t
from src.exact_posterior import GermanTankExactBayesian

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.dpi": 150})


def fit_normal_gamma(y, eta0=6.0, kappa0=1.0, a0=2.0, b0=2.0):
    """
    Fits conjugate Normal-Gamma model to 1D log-volumes y:
      y_t ~ N(eta, 1/tau)
      eta | tau ~ N(eta0, 1/(kappa0 * tau)),  tau ~ Gamma(a0, b0)
    Returns posterior hyperparameters and Student-t predictive distribution parameters.
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    y_bar = np.mean(y)
    ss_y = np.sum((y - y_bar) ** 2)

    kappa_T = kappa0 + T
    eta_T = (kappa0 * eta0 + T * y_bar) / kappa_T
    a_T = a0 + T / 2.0
    b_T = b0 + 0.5 * ss_y + (kappa0 * T * (y_bar - eta0) ** 2) / (2.0 * kappa_T)

    # Student-t predictive distribution for next observation y_{T+1}:
    # df = 2 * a_T, loc = eta_T, scale = sqrt(b_T * (kappa_T + 1) / (a_T * kappa_T))
    pred_df = 2.0 * a_T
    pred_loc = eta_T
    pred_scale = np.sqrt(b_T * (kappa_T + 1.0) / (a_T * kappa_T))

    return {
        "eta_T": eta_T,
        "kappa_T": kappa_T,
        "a_T": a_T,
        "b_T": b_T,
        "pred_df": pred_df,
        "pred_loc": pred_loc,
        "pred_scale": pred_scale,
    }


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("==============================================================")
    print("      WORK PACKAGE 3: TIME-VARYING PRODUCTION & NORMAL-GAMMA  ")
    print("==============================================================")

    # 1. Load 104 Weeks from Online Retail II
    df_weekly = pd.read_csv("data/d2_online_retail_ii/weekly_invoices.csv")
    print(f"Loaded {len(df_weekly)} weeks of retail invoice flows.")
    print(
        f"Total 2-Year Cumulative Invoices: {df_weekly['cumulative_invoices'].iloc[-1]:,}"
    )

    # 2. Simulate Weekly Snapshot Tank Sampling
    # At each week t, we draw a sample of k=20 from invoices issued so far
    np.random.seed(42)
    weeks = df_weekly["week_idx"].values
    true_cumulative = df_weekly["cumulative_invoices"].values
    true_weekly = df_weekly["weekly_invoices"].values

    est_cumulative_unconstrained = []
    est_cumulative_constrained = []
    prev_constrained = 0

    for t, n_cum in zip(weeks, true_cumulative):
        # Sample k=20 items uniformly from 1..n_cum
        k_t = 20
        sub = np.random.choice(n_cum, size=k_t, replace=False) + 1
        m_t = int(np.max(sub))

        # Unconstrained Bayes estimate
        est_unconst = (k_t - 1.0) * (m_t - 1.0) / (k_t - 2.0)
        est_cumulative_unconstrained.append(est_unconst)

        # Monotonically constrained estimate N_t >= max(m_t, N_{t-1})
        constrained_val = max(est_unconst, prev_constrained, m_t)
        est_cumulative_constrained.append(constrained_val)
        prev_constrained = constrained_val

    df_weekly["est_N_unconstrained"] = est_cumulative_unconstrained
    df_weekly["est_N_constrained"] = est_cumulative_constrained

    # Weekly rate differences
    df_weekly["est_weekly_unconstrained"] = np.diff(
        np.insert(est_cumulative_unconstrained, 0, 0)
    )
    df_weekly["est_weekly_constrained"] = np.diff(
        np.insert(est_cumulative_constrained, 0, 0)
    )

    # Compare error in weekly estimates
    rmse_unconst = np.sqrt(
        np.mean(
            (df_weekly["weekly_invoices"] - df_weekly["est_weekly_unconstrained"])
            ** 2
        )
    )
    rmse_const = np.sqrt(
        np.mean(
            (df_weekly["weekly_invoices"] - df_weekly["est_weekly_constrained"])
            ** 2
        )
    )
    print(f"Weekly Rate Estimation RMSE (Unconstrained Differences): {rmse_unconst:.1f}")
    print(f"Weekly Rate Estimation RMSE (Constrained Differences):   {rmse_const:.1f}")

    # 3. Second Stage: Normal-Gamma on Log Weekly Volume
    log_weekly = np.log(df_weekly["weekly_invoices"].values)
    # Train on first 80 weeks, forecast remaining 24 weeks
    T_train = 80
    train_log = log_weekly[:T_train]
    test_log = log_weekly[T_train:]
    actual_test_vol = df_weekly["weekly_invoices"].values[T_train:]

    ng_res = fit_normal_gamma(train_log, eta0=6.0, kappa0=1.0, a0=2.0, b0=2.0)
    print(f"Fitted Normal-Gamma on {T_train} weeks:")
    print(f"  Posterior mean log-intensity eta: {ng_res['eta_T']:.4f} (Volume ~ {np.exp(ng_res['eta_T']):.0f} invoices/wk)")
    print(f"  Student-t Predictive df: {ng_res['pred_df']:.1f}, scale: {ng_res['pred_scale']:.4f}")

    # Forecast interval (95%)
    pred_low_log = student_t.ppf(
        0.025,
        df=ng_res["pred_df"],
        loc=ng_res["pred_loc"],
        scale=ng_res["pred_scale"],
    )
    pred_high_log = student_t.ppf(
        0.975,
        df=ng_res["pred_df"],
        loc=ng_res["pred_loc"],
        scale=ng_res["pred_scale"],
    )
    pred_low_vol = np.exp(pred_low_log)
    pred_high_vol = np.exp(pred_high_log)
    pred_median_vol = np.exp(ng_res["pred_loc"])

    # Score against test period
    coverage_test = np.mean(
        (actual_test_vol >= pred_low_vol) & (actual_test_vol <= pred_high_vol)
    )
    print(
        f"Forecasting 95% Predictive Interval: [{pred_low_vol:.0f}, {pred_high_vol:.0f}] invoices/wk (Median = {pred_median_vol:.0f})"
    )
    print(f"Empirical Coverage on Held-out 24 Weeks: {coverage_test*100:.1f}%")

    df_weekly.to_csv("results/wp3_dynamic_production.csv", index=False)
    print("Saved results/wp3_dynamic_production.csv")

    # 4. Generate Publication Figures
    print("\n>>> Generating Publication Figures for WP3...")
    generate_wp3_figures(
        df_weekly,
        T_train,
        pred_low_vol,
        pred_high_vol,
        pred_median_vol,
        ng_res,
        test_log,
    )
    print("All WP3 figures successfully created in figures/")


def generate_wp3_figures(
    df, T_train, pred_low_vol, pred_high_vol, pred_median_vol, ng_res, test_log
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Cumulative Production Tracking
    axes[0].plot(
        df["week_idx"],
        df["cumulative_invoices"],
        color="black",
        linewidth=2,
        label="Actual Cumulative N_t",
    )
    axes[0].plot(
        df["week_idx"],
        df["est_N_constrained"],
        color="tab:blue",
        linestyle="--",
        linewidth=2,
        label="Constrained Tank Posterior Mean",
    )
    axes[0].plot(
        df["week_idx"],
        df["est_N_unconstrained"],
        color="tab:gray",
        linestyle=":",
        alpha=0.6,
        label="Unconstrained Naive Estimate",
    )
    axes[0].set_title(
        "Cumulative Production Trajectory N_t over 104 Weeks"
    )
    axes[0].set_xlabel("Week Index (2009 - 2011)")
    axes[0].set_ylabel("Total Cumulative Invoices")
    axes[0].legend()

    # Right: Weekly Production Flow & Normal-Gamma Predictive Interval
    axes[1].plot(
        df["week_idx"][:T_train],
        df["weekly_invoices"][:T_train],
        color="tab:blue",
        label="Training Weekly Volume",
    )
    axes[1].plot(
        df["week_idx"][T_train:],
        df["weekly_invoices"][T_train:],
        color="tab:green",
        label="Held-out Test Volume",
    )
    axes[1].axhline(
        pred_median_vol,
        color="tab:red",
        linestyle="--",
        label=f"Normal-Gamma Predictive Median ({pred_median_vol:.0f})",
    )
    axes[1].axhspan(
        pred_low_vol,
        pred_high_vol,
        color="tab:red",
        alpha=0.2,
        label="95% Student-t Predictive Band",
    )
    axes[1].axvline(T_train, color="gray", linestyle=":", label="Forecast Horizon")
    axes[1].set_title("Weekly Production Flow & Second-Stage Normal-Gamma Forecast")
    axes[1].set_xlabel("Week Index")
    axes[1].set_ylabel("Invoices per Week")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("figures/wp3_dynamic_production.png")
    plt.close()


if __name__ == "__main__":
    main()
