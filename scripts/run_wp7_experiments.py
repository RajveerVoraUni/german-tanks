"""
run_wp7_experiments.py: Full pipeline for Work Package 7.
Dirichlet-Multinomial shares across aircraft models (FAA Registry),
and semi-synthetic counterfeit detection experiment (cloned serials & out-of-range serials)
fitted with a Beta-Bernoulli contamination Gibbs sampler.
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
from scipy.special import betaln, gammaln
from sklearn.metrics import auc, roc_curve

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.dpi": 150})


def fit_dirichlet_multinomial(counts, alpha_prior=1.0):
    """
    Computes Dirichlet-Multinomial posterior distribution of model shares pi:
      counts ~ Mult(k, pi),  pi ~ Dir(alpha)
      pi | counts ~ Dir(alpha + counts)
    """
    counts = np.asarray(counts, dtype=np.float64)
    F = len(counts)
    alpha = np.full(F, alpha_prior, dtype=np.float64)
    alpha_post = alpha + counts
    mean_shares = alpha_post / np.sum(alpha_post)
    # Variance of Dirichlet component: alpha_f * (sum - alpha_f) / (sum^2 * (sum + 1))
    sum_a = np.sum(alpha_post)
    var_shares = (alpha_post * (sum_a - alpha_post)) / (sum_a**2 * (sum_a + 1.0))
    sd_shares = np.sqrt(var_shares)
    return mean_shares, sd_shares, alpha_post


class CounterfeitGibbsSampler:
    """
    Beta-Bernoulli Contamination Gibbs Sampler:
      z_i ~ Bernoulli(eps), eps ~ Beta(a, b)
      If z_i = 0 (genuine): s_i ~ Uniform({1, ..., L})
      If z_i = 1 (counterfeit): s_i ~ Contamination distribution
        (either duplicate clone or out-of-range s > L)
    """

    def __init__(self, observed_serials, legal_cap_L, a_eps=1.0, b_eps=10.0):
        self.serials = np.asarray(observed_serials, dtype=np.int64)
        self.k = len(self.serials)
        self.L = int(legal_cap_L)
        self.a_eps = float(a_eps)
        self.b_eps = float(b_eps)

    def sample(self, n_iter=3000, burn_in=1000, seed=42):
        np.random.seed(seed)
        eps = 0.05
        # z_i in {0, 1}
        z = np.zeros(self.k, dtype=int)
        # Any serial > L is definitely counterfeit (z_i = 1)
        z[self.serials > self.L] = 1

        trace_eps = np.empty(n_iter, dtype=np.float64)
        trace_n_fake = np.empty(n_iter, dtype=np.int64)

        for it in range(n_iter):
            # 1. Update z_i for each serial
            for i in range(self.k):
                s = self.serials[i]
                if s > self.L:
                    z[i] = 1
                else:
                    # Likelihood if genuine: 1/L
                    lik_0 = 1.0 / self.L
                    # Likelihood if counterfeit: uniform on broader illegal domain [1, 2*L]
                    lik_1 = 1.0 / (2.0 * self.L)

                    p0 = (1.0 - eps) * lik_0
                    p1 = eps * lik_1
                    prob_fake = p1 / (p0 + p1)
                    z[i] = 1 if np.random.uniform(0.0, 1.0) < prob_fake else 0

            # 2. Update eps | z ~ Beta(a + sum(z), b + k - sum(z))
            n_fake = np.sum(z)
            eps = np.random.beta(
                self.a_eps + n_fake, self.b_eps + self.k - n_fake
            )

            trace_eps[it] = eps
            trace_n_fake[it] = n_fake

        return {
            "eps": trace_eps[burn_in:],
            "n_fake": trace_n_fake[burn_in:],
            "z_final": z,
        }


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("==============================================================")
    print("      WORK PACKAGE 7: MULTI-SERIES & COUNTERFEIT AIRCRAFT     ")
    print("==============================================================")

    # 1. Load FAA Aircraft Database
    print("Loading FAA Aircraft records...")
    df_master = pd.read_csv(
        "data/d7_faa_aircraft/MASTER.txt",
        usecols=["SERIAL NUMBER", "MFR MDL CODE"],
        low_memory=False,
    )
    df_master.columns = [c.strip() for c in df_master.columns]

    # Clean serials for Cessna 172M (MFR MDL CODE 2072418)
    c172_mask = df_master["MFR MDL CODE"] == "2072418"
    raw_serials = df_master[c172_mask]["SERIAL NUMBER"].str.strip()
    valid_numeric = raw_serials[raw_serials.str.isdigit()].astype(np.int64).values

    min_s = valid_numeric.min()
    # Shift to start at 1:
    genuine_serials = valid_numeric - min_s + 1
    print(
        f"Loaded {len(genuine_serials):,} Cessna 172M serial numbers (Range: 1 to {genuine_serials.max():,})"
    )

    # 2. Multi-Series Dirichlet-Multinomial Analysis across Cessna Fleet
    # Top 4 Cessna light aircraft models: 172M (2072418), 172N (2072434), 150M (2075816), 182P (2072714)
    cessna_models = {
        "172M": "2072418",
        "172N": "2072434",
        "150M": "2075816",
        "182P": "2072714",
    }
    counts = [
        int(np.sum(df_master["MFR MDL CODE"] == code))
        for code in cessna_models.values()
    ]
    mean_shares, sd_shares, alpha_post = fit_dirichlet_multinomial(counts)

    df_shares = pd.DataFrame(
        {
            "Model": list(cessna_models.keys()),
            "FAA_Registrations": counts,
            "Dirichlet_Mean_Share": mean_shares,
            "Share_Posterior_SD": sd_shares,
            "Dirichlet_95_Low": np.maximum(0, mean_shares - 1.96 * sd_shares),
            "Dirichlet_95_High": mean_shares + 1.96 * sd_shares,
        }
    )
    print("\nDirichlet-Multinomial Shares across Cessna Fleet:")
    print(df_shares.to_string(index=False))
    df_shares.to_csv("results/wp7_dirichlet_shares.csv", index=False)

    # 3. Semi-Synthetic Counterfeit Injection Experiment
    print("\n>>> Phase 2: Semi-Synthetic Counterfeit Detection Experiment...")
    legal_cap_L = 8000
    sub_sample_size = 100
    rates_epsilon = [0.0, 0.01, 0.05, 0.10]
    counterfeit_results = []

    np.random.seed(42)
    for eps_true in rates_epsilon:
        n_fake = int(round(sub_sample_size * eps_true))
        n_real = sub_sample_size - n_fake

        # Draw genuine units <= L
        pool_real = genuine_serials[genuine_serials <= legal_cap_L]
        real_draws = np.random.choice(pool_real, size=n_real, replace=False)

        # Generate counterfeits: 50% clones, 50% out-of-range (s > L)
        fake_draws = []
        for _ in range(n_fake):
            if np.random.uniform(0, 1) < 0.5:
                # Clone: duplicate existing real draw
                fake_draws.append(np.random.choice(real_draws))
            else:
                # Out of range serials: uniformly drawn from [L+1, L+3000]
                fake_draws.append(
                    np.random.randint(legal_cap_L + 1, legal_cap_L + 3000)
                )

        combined_sample = np.concatenate([real_draws, fake_draws])
        sampler = CounterfeitGibbsSampler(
            combined_sample, legal_cap_L=legal_cap_L
        )
        res_cf = sampler.sample(n_iter=3000, burn_in=1000, seed=42)

        eps_post = res_cf["eps"]
        mean_eps = float(np.mean(eps_post))
        ci_eps = (
            float(np.percentile(eps_post, 2.5)),
            float(np.percentile(eps_post, 97.5)),
        )

        counterfeit_results.append(
            {
                "True_Epsilon": eps_true,
                "Injected_Fakes": n_fake,
                "Observed_Max": int(np.max(combined_sample)),
                "Posterior_Mean_Eps": mean_eps,
                "CI_95_Low": ci_eps[0],
                "CI_95_High": ci_eps[1],
                "Naive_Rule_Detection": 1
                if np.max(combined_sample) > legal_cap_L
                else 0,
            }
        )

    df_cf = pd.DataFrame(counterfeit_results)
    print(df_cf.to_string(index=False))
    df_cf.to_csv("results/wp7_counterfeit_results.csv", index=False)

    # 4. ROC Evaluation: Bayesian Contamination vs Naive Max Rule
    print("\n>>> Phase 3: ROC Curve Analysis (Bayesian vs Naive Rule)...")
    y_true_labels = []
    bayes_scores = []
    naive_scores = []

    for b_idx in range(200):
        # 50% chance of clean batch (eps=0), 50% chance of contaminated batch (eps=0.05)
        is_contaminated = (b_idx % 2 == 1)
        eps_cur = 0.05 if is_contaminated else 0.0
        n_fk = int(round(sub_sample_size * eps_cur))
        n_rl = sub_sample_size - n_fk

        real_d = np.random.choice(pool_real, size=n_rl, replace=False)
        fake_d = [
            np.random.choice(real_d)
            if np.random.uniform(0, 1) < 0.5
            else np.random.randint(legal_cap_L + 1, legal_cap_L + 2000)
            for _ in range(n_fk)
        ]
        comb = np.concatenate([real_d, fake_d])

        smplr = CounterfeitGibbsSampler(comb, legal_cap_L=legal_cap_L)
        res = smplr.sample(n_iter=1000, burn_in=300, seed=1000 + b_idx)

        y_true_labels.append(1 if is_contaminated else 0)
        bayes_scores.append(float(np.mean(res["eps"])))
        naive_scores.append(1.0 if np.max(comb) > legal_cap_L else 0.0)

    fpr_b, tpr_b, _ = roc_curve(y_true_labels, bayes_scores)
    auc_b = auc(fpr_b, tpr_b)

    fpr_n, tpr_n, _ = roc_curve(y_true_labels, naive_scores)
    auc_n = auc(fpr_n, tpr_n)

    print(
        f"ROC AUC - Bayesian Contamination Detector: {auc_b:.3f} vs Naive Max Rule: {auc_n:.3f}"
    )

    # 5. Generate Figures
    print("\n>>> Generating Publication Figures for WP7...")
    generate_wp7_figures(df_shares, df_cf, fpr_b, tpr_b, auc_b, fpr_n, tpr_n, auc_n)
    print("All WP7 figures successfully created in figures/")


def generate_wp7_figures(
    df_shares, df_cf, fpr_b, tpr_b, auc_b, fpr_n, tpr_n, auc_n
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Dirichlet Shares
    x_pos = np.arange(len(df_shares))
    axes[0].bar(
        x_pos,
        df_shares["Dirichlet_Mean_Share"],
        yerr=1.96 * df_shares["Share_Posterior_SD"],
        capsize=5,
        color="tab:blue",
        alpha=0.8,
        label="Posterior Share (95% Credible Band)",
    )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(df_shares["Model"])
    axes[0].set_title(
        "Dirichlet-Multinomial Fleet Shares across Cessna Aircraft Families"
    )
    axes[0].set_xlabel("Aircraft Model")
    axes[0].set_ylabel("Production Share π_f")
    axes[0].legend()

    # Right: ROC Curve (Bayesian Contamination vs Naive Rule)
    axes[1].plot(
        fpr_b,
        tpr_b,
        color="tab:red",
        linewidth=2.5,
        label=f"Bayesian Gibbs Contamination Detector (AUC = {auc_b:.3f})",
    )
    axes[1].plot(
        fpr_n,
        tpr_n,
        color="tab:gray",
        linestyle="--",
        linewidth=2,
        label=f"Naive Max Rule (s_max > L) (AUC = {auc_n:.3f})",
    )
    axes[1].plot([0, 1], [0, 1], color="black", linestyle=":", alpha=0.5)
    axes[1].set_title(
        "Counterfeit Detection ROC: Bayesian Gibbs vs Naive Max Rule"
    )
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("figures/wp7_counterfeit_roc.png")
    plt.close()


if __name__ == "__main__":
    main()
