"""
run_wp4_experiments.py: Full pipeline for Work Package 4 (Jelinski-Moranda Software Reliability).
Executes exact Bayesian posterior calculations on Musa/DACS systems, hold-out predictive scoring,
and software release decision rule evaluation.
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
from src.jelinski_moranda import JelinskiMorandaBayesian

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.dpi": 150})


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("==============================================================")
    print("      WORK PACKAGE 4: JELINSKI-MORANDA SOFTWARE RELIABILITY   ")
    print("==============================================================")

    # 1. Load Musa sys1 data (valid failure intervals x >= 0, row 136 is negative censoring flag)
    df_sys1 = pd.read_csv("data/d3_musa/sys1.csv")
    raw_t = df_sys1.iloc[:, 0].values.astype(float)
    t_all = raw_t[raw_t >= 0]
    total_failures = len(t_all)
    print(
        f"Loaded Musa System 1: {total_failures} valid failures, execution times range {t_all.min():.0f}s to {t_all.max():.0f}s"
    )

    k_train_list = [30, 50, 80, 100]
    results = []
    models_by_k = {}

    # 2. Sequential Training and Hold-out Evaluation
    for k_train in k_train_list:
        t_train = t_all[:k_train]
        t_test = t_all[k_train:]

        model = JelinskiMorandaBayesian(
            t_train, a=1.0, b=100.0, prior_type="1/n", n_max=1000
        )
        models_by_k[k_train] = model

        mean_N = model.posterior_mean_N()
        mean_rem = model.posterior_mean_remaining()
        ci_N = model.credible_interval_N(0.95)
        p_release_5 = model.release_rule_probability(max_remaining=5)
        p_release_15 = model.release_rule_probability(max_remaining=15)
        plan_bugs = model.asymmetric_loss_quantile(cu_delay_ratio=4.0)

        # Hold-out log score on first 20 test points
        heldout_scores = [
            model.predict_next_failure_log_score(t_val) for t_val in t_test[:20]
        ]
        mean_test_log_score = (
            float(np.mean(heldout_scores)) if heldout_scores else np.nan
        )

        results.append(
            {
                "k_train": k_train,
                "Heldout_Window": len(t_test),
                "Posterior_Mean_N": mean_N,
                "Posterior_Mean_Remaining": mean_rem,
                "CI_95_Low": ci_N[0],
                "CI_95_High": ci_N[1],
                "P_Release_Under_5_Bugs": p_release_5,
                "P_Release_Under_15_Bugs": p_release_15,
                "Planned_Bugs_Asymm_Loss": plan_bugs,
                "Test_Log_Score": mean_test_log_score,
            }
        )

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    df_res.to_csv("results/wp4_jm_results.csv", index=False)
    print("Saved results/wp4_jm_results.csv")

    # 3. Generate Publication Figures
    print("\n>>> Generating Publication Figures for WP4...")
    generate_wp4_figures(models_by_k, df_res, total_failures)
    print("All WP4 figures successfully created in figures/")


def generate_wp4_figures(models_by_k, df_res, total_failures):
    # Figure 1: Posterior contraction of N across training sample sizes
    plt.figure(figsize=(9, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for idx, (k_train, model) in enumerate(models_by_k.items()):
        plt.plot(
            model.support,
            model.probs,
            label=f"k = {k_train} observed (Mean = {model.posterior_mean_N():.0f})",
            color=colors[idx],
            linewidth=2,
        )

    plt.axvline(total_failures, color="red", linestyle="--", linewidth=2, label=f"Total Documented Failures ({total_failures})")
    plt.xlim(20, 250)
    plt.title("Jelinski-Moranda Bayesian Posterior Contraction for Total Defects N")
    plt.xlabel("Total Initial Defects N")
    plt.ylabel("Posterior Probability π(N | Data)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/wp4_jm_posteriors_growth.png")
    plt.close()

    # Figure 2: Software Release Rule & Asymmetric Loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Release probability vs training failures
    axes[0].plot(df_res["k_train"], df_res["P_Release_Under_15_Bugs"], marker="o", color="tab:green", linewidth=2, label="P(Remaining ≤ 15 | Data)")
    axes[0].plot(df_res["k_train"], df_res["P_Release_Under_5_Bugs"], marker="s", color="tab:blue", linewidth=2, label="P(Remaining ≤ 5 | Data)")
    axes[0].axhline(0.95, color="red", linestyle="--", label="95% Release Threshold")
    axes[0].set_title("Software Release Certification Probability vs Testing Horizon")
    axes[0].set_xlabel("Number of Observed Failures k")
    axes[0].set_ylabel("Release Certification Probability")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()

    # Right: Credible interval contraction for remaining defects N - k
    axes[1].fill_between(
        df_res["k_train"],
        df_res["CI_95_Low"] - df_res["k_train"],
        df_res["CI_95_High"] - df_res["k_train"],
        color="tab:blue",
        alpha=0.25,
        label="95% Credible Band for Remaining Defects",
    )
    axes[1].plot(df_res["k_train"], df_res["Posterior_Mean_Remaining"], marker="o", color="tab:blue", linewidth=2, label="Posterior Mean Remaining")
    axes[1].plot(df_res["k_train"], df_res["Planned_Bugs_Asymm_Loss"], marker="^", color="tab:red", linestyle="--", label="Asymmetric Loss Plan (cu:co = 4:1)")
    axes[1].set_title("Uncertainty in Remaining Defects (N - k)")
    axes[1].set_xlabel("Observed Failures k")
    axes[1].set_ylabel("Remaining Defects (N - k)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/wp4_jm_release_decisions.png")
    plt.close()


if __name__ == "__main__":
    main()
