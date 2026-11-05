"""
evaluator.py: Simulation and evaluation engine for German Tank Bayesian inference (WP1).
Uses FastPrecomputedModel for sub-second Monte Carlo evaluation across thousands of replicates.
"""

import time
import numpy as np
import pandas as pd
from src.exact_posterior import FastPrecomputedModel
from src.frequentist import frequentist_ci, mle, umvue


def run_wp1_simulation(
    labels,
    n_true,
    k_list=(5, 10, 20, 50, 100),
    r_replicates=2000,
    seed=42,
    priors_config=None,
    label_type="dense",
):
    """
    Executes WP1 replication study:
    - Draws R random subsets of size k without replacement.
    - Evaluates Bayesian posteriors (Flat, 1/n, Truncated, Poisson, NegBin).
    - Evaluates Frequentist UMVUE and MLE.
    - Tracks point error, interval coverage, scoring rules.
    """
    np.random.seed(seed)

    if priors_config is None:
        priors_config = {
            "Flat": ("flat", {}),
            "1/n (Scale-Inv)": ("reciprocal", {}),
            "Truncated (Omega=100k)": ("truncated_uniform", {"omega": 100000}),
            "Poisson (mu=N_true)": ("poisson", {"mu": float(n_true)}),
            "Poisson (mu=0.5*N_true)": ("poisson", {"mu": float(0.5 * n_true)}),
            "Poisson (mu=2.0*N_true)": ("poisson", {"mu": float(2.0 * n_true)}),
            "NegBin (mu=N_true)": ("negbin", {"r": 10.0, "p": 10.0 / (10.0 + n_true)}),
        }

    results_summary = []
    print(
        f"--- Starting WP1 Evaluation ({label_type.upper()}) | N_true={n_true:,} | R={r_replicates:,} ---"
    )

    # Determine master n_max based on label bounds
    max_label = int(np.max(labels))
    n_max_eval = max(150000, max_label + 50000)

    for k in k_list:
        t0 = time.time()
        print(f"Running k = {k}...")

        # 1. Instantiate fast precomputed models for each prior
        models = {}
        for prior_name, (ptype, pparams) in priors_config.items():
            models[prior_name] = FastPrecomputedModel(
                k, prior_type=ptype, prior_params=pparams, n_max=n_max_eval
            )

        # 2. Draw all R samples in batch
        samples_m = np.empty(r_replicates, dtype=np.int64)
        for rep in range(r_replicates):
            sub = np.random.choice(labels, size=k, replace=False)
            samples_m[rep] = np.max(sub)

        # 3. Frequentist evaluations
        u_vals = umvue(samples_m, k)
        m_vals = mle(samples_m)
        u_err = u_vals - n_true
        m_err = m_vals - n_true

        # Frequentist CI coverage
        freq_covered = 0
        freq_widths = []
        for m_val in samples_m:
            f_ci = frequentist_ci(m_val, k, confidence=0.95)
            if f_ci[0] <= n_true <= f_ci[1]:
                freq_covered += 1
            freq_widths.append(f_ci[1] - f_ci[0])

        results_summary.append(
            {
                "k": k,
                "label_type": label_type,
                "method": "Frequentist UMVUE",
                "estimator": "UMVUE",
                "bias": float(np.mean(u_err)),
                "rmse": float(np.sqrt(np.mean(u_err**2))),
                "coverage_95": float(freq_covered / r_replicates),
                "mean_interval_width": float(np.mean(freq_widths)),
                "log_score": np.nan,
                "crps": np.nan,
            }
        )
        results_summary.append(
            {
                "k": k,
                "label_type": label_type,
                "method": "Frequentist MLE",
                "estimator": "MLE (m)",
                "bias": float(np.mean(m_err)),
                "rmse": float(np.sqrt(np.mean(m_err**2))),
                "coverage_95": np.nan,
                "mean_interval_width": np.nan,
                "log_score": np.nan,
                "crps": np.nan,
            }
        )

        # 4. Bayesian evaluations per prior
        for prior_name, model in models.items():
            mean_errors = []
            med_errors = []
            map_errors = []
            hpd_covered = 0
            hpd_widths = []
            eq_covered = 0
            eq_widths = []
            log_scores = []
            crps_scores = []

            # Unique m caching per replicate batch
            m_cache = {}
            for m_val in samples_m:
                if m_val not in m_cache:
                    m_cache[m_val] = model.evaluate_sample(m_val, n_true=n_true)

                (
                    mean_e,
                    med_e,
                    map_e,
                    hpd_ci,
                    eq_ci,
                    l_score,
                    crps_val,
                    pit_val,
                ) = m_cache[m_val]
                mean_errors.append(mean_e - n_true)
                med_errors.append(med_e - n_true)
                map_errors.append(map_e - n_true)

                if hpd_ci[0] <= n_true <= hpd_ci[1]:
                    hpd_covered += 1
                hpd_widths.append(hpd_ci[1] - hpd_ci[0])

                if eq_ci[0] <= n_true <= eq_ci[1]:
                    eq_covered += 1
                eq_widths.append(eq_ci[1] - eq_ci[0])

                if np.isfinite(l_score):
                    log_scores.append(l_score)
                crps_scores.append(crps_val)

            # Summaries
            results_summary.append(
                {
                    "k": k,
                    "label_type": label_type,
                    "method": f"Bayes ({prior_name})",
                    "estimator": "Posterior Mean (SE loss)",
                    "bias": float(np.mean(mean_errors)),
                    "rmse": float(np.sqrt(np.mean(np.array(mean_errors) ** 2))),
                    "coverage_95": float(hpd_covered / r_replicates),
                    "mean_interval_width": float(np.mean(hpd_widths)),
                    "log_score": float(np.mean(log_scores)) if log_scores else np.nan,
                    "crps": float(np.mean(crps_scores)),
                }
            )
            results_summary.append(
                {
                    "k": k,
                    "label_type": label_type,
                    "method": f"Bayes ({prior_name})",
                    "estimator": "Posterior Median (Abs loss)",
                    "bias": float(np.mean(med_errors)),
                    "rmse": float(np.sqrt(np.mean(np.array(med_errors) ** 2))),
                    "coverage_95": float(eq_covered / r_replicates),
                    "mean_interval_width": float(np.mean(eq_widths)),
                    "log_score": float(np.mean(log_scores)) if log_scores else np.nan,
                    "crps": float(np.mean(crps_scores)),
                }
            )

        elapsed = time.time() - t0
        print(f"Finished k={k} across {len(priors_config)} priors in {elapsed:.2f}s")

    return pd.DataFrame(results_summary)
