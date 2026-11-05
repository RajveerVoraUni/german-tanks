"""
run_wp8_experiments.py: Full pipeline for Work Package 8.
Two-stage wildlife abundance model:
  Stage 1: German-tank posterior for tags deployed M from recovered tag serials.
  Stage 2: Beta-Binomial mark-recapture abundance model N_pop = M / p.
Validated on the published Snowshoe Hare dataset (CRAN Rcapture).
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
from src.exact_posterior import GermanTankExactBayesian

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.dpi": 150})


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("==============================================================")
    print("      WORK PACKAGE 8: TWO-STAGE WILDLIFE ABUNDANCE MODEL      ")
    print("==============================================================")

    # 1. Load Snowshoe Hare Capture-Recapture Data
    df_hare = pd.read_csv("data/d9_rcapture/hare.csv")
    total_individuals = len(df_hare)
    print(f"Loaded Snowshoe Hare dataset: {total_individuals} unique hares captured over 6 occasions.")

    # Occasion 1 & 2 capture counts
    M_occ1 = int(df_hare["c1"].sum())
    n_occ2 = int(df_hare["c2"].sum())
    r_marked = int(((df_hare["c1"] == 1) & (df_hare["c2"] == 1)).sum())

    print(f"Occasion 1 (Marked Initial Set): M = {M_occ1}")
    print(f"Occasion 2 (Recapture Sample):   n = {n_occ2}, Marked Recaptures = {r_marked}")

    # Frequentist Chapman benchmark
    chapman_est = float(((M_occ1 + 1) * (n_occ2 + 1) / (r_marked + 1)) - 1)
    chapman_var = float(
        ((M_occ1 + 1) * (n_occ2 + 1) * (M_occ1 - r_marked) * (n_occ2 - r_marked))
        / ((r_marked + 1) ** 2 * (r_marked + 2))
    )
    chapman_se = np.sqrt(chapman_var)
    print(f"Frequentist Chapman Abundance Estimate: N_hat = {chapman_est:.1f} (SE = {chapman_se:.1f})")

    # 2. Stage 1: German Tank Tag Inference (Incomplete Tagging Records)
    # Suppose tags were sequentially deployed 1..M, but only k=10 tag numbers are recorded, max = 16
    k_tags = 10
    m_tag_max = M_occ1
    post_M = GermanTankExactBayesian(
        m_tag_max, k_tags, prior_type="1/n", grid_max=100
    )
    mean_M = post_M.posterior_mean()
    hpd_M = post_M.credible_interval_hpd(0.95)
    print(f"Stage 1 Tag Posterior: Mean M = {mean_M:.1f}, 95% HPD = {hpd_M} (Actual M = {M_occ1})")

    # 3. Stage 2: Mark-Recapture Beta-Binomial Updating
    # p ~ Beta(a, b)
    a_p, b_p = 1.0, 1.0
    a_post = a_p + r_marked
    b_post = b_p + (n_occ2 - r_marked)

    # 4. Joint Two-Stage Posterior Composition
    # Draw M^{(s)} from Stage 1 posterior, draw p^{(s)} from Stage 2 posterior, N_pop = M / p
    np.random.seed(42)
    n_sims = 10000

    # Draw M from discrete posterior
    supp_M, probs_M = post_M.posterior_pmf()
    m_samples = np.random.choice(supp_M, size=n_sims, p=probs_M)

    # Draw capture probability p
    p_samples = np.random.beta(a_post, b_post, size=n_sims)

    # Composed population abundance
    n_pop_samples = m_samples / p_samples

    mean_n_pop = float(np.mean(n_pop_samples))
    median_n_pop = float(np.median(n_pop_samples))
    ci_n_pop = (
        float(np.percentile(n_pop_samples, 2.5)),
        float(np.percentile(n_pop_samples, 97.5)),
    )

    print("\nTwo-Stage Composed Bayesian Abundance Posterior:")
    print(f"  Posterior Mean N_pop:   {mean_n_pop:.1f}")
    print(f"  Posterior Median N_pop: {median_n_pop:.1f}")
    print(f"  95% Credible Interval:  [{ci_n_pop[0]:.1f}, {ci_n_pop[1]:.1f}]")
    print("  Published Log-Linear Abundance Estimates: ~77 to 81 hares")

    wildlife_df = pd.DataFrame(
        [
            {
                "Method": "Frequentist Chapman",
                "Estimate": chapman_est,
                "SE": chapman_se,
                "CI_Low": max(0, chapman_est - 1.96 * chapman_se),
                "CI_High": chapman_est + 1.96 * chapman_se,
            },
            {
                "Method": "Two-Stage Composed Bayes (Tank + Recapture)",
                "Estimate": median_n_pop,
                "SE": float(np.std(n_pop_samples)),
                "CI_Low": ci_n_pop[0],
                "CI_High": ci_n_pop[1],
            },
        ]
    )
    wildlife_df.to_csv("results/wp8_wildlife_summary.csv", index=False)
    print("Saved results/wp8_wildlife_summary.csv")

    # 5. Generate Publication Figure
    print("\n>>> Generating Publication Figure for WP8...")
    generate_wp8_figure(n_pop_samples, chapman_est, ci_n_pop)
    print("All WP8 figures successfully created in figures/")


def generate_wp8_figure(n_pop_samples, chapman_est, ci_n_pop):
    plt.figure(figsize=(8, 5))
    # Filter extreme tail for display
    display_samples = n_pop_samples[n_pop_samples <= 350]
    plt.hist(
        display_samples,
        bins=50,
        density=True,
        color="tab:olive",
        alpha=0.6,
        label="Two-Stage Posterior π(N_pop | Data)",
    )
    plt.axvline(
        chapman_est,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Chapman Estimator ({chapman_est:.0f})",
    )
    plt.axvspan(
        77,
        81,
        color="tab:green",
        alpha=0.3,
        label="Published Abundance Benchmark (77 - 81)",
    )
    plt.axvline(
        np.median(n_pop_samples),
        color="black",
        linewidth=2,
        label=f"Bayes Posterior Median ({np.median(n_pop_samples):.0f})",
    )
    plt.title(
        "Two-Stage Wildlife Population Abundance (Snowshoe Hares)"
    )
    plt.xlabel("Snowshoe Hare Population Size N_pop")
    plt.ylabel("Posterior Density")
    plt.xlim(30, 300)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/wp8_wildlife_abundance.png")
    plt.close()


if __name__ == "__main__":
    main()
