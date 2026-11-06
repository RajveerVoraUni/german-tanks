# Bayesian Inference on the German Tank Problem: Theoretical Foundations, Decision Theory, and Modern Applications

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LaTeX pdflatex](https://img.shields.io/badge/latex-pdflatex-success.svg)](https://www.latex-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course Project](https://img.shields.io/badge/Course-Bayesian%20Inference-navy.svg)](#)

---

## Executive Overview

The **German Tank Problem** is a cornerstone of discrete parameter estimation: inferring an unknown population ceiling $N \in \mathbb{N}$ from a sample of $k$ sequential serial numbers $s_1, \dots, s_k$ observed without replacement from $\{1, \dots, N\}$. 

While introductory statistics textbooks frequently treat this as a simple exercise in the classical Uniformly Minimum Variance Unbiased Estimator (UMVUE):
$$\hat{N}_{\text{UMVUE}} = m \left(1 + \frac{1}{k}\right) - 1, \quad \text{where } m = \max(s_1, \dots, s_k),$$
real-world data analysis presents significant mathematical and practical challenges:
1. **Physical Boundary Admissibility:** Asymptotic Gaussian Wald confidence intervals assign substantial probability mass ($17.5\%$ for $m=50, k=5$) to physically impossible population ceilings ($N < m$). In contrast, Bayesian posteriors strictly enforce $P(N < m \mid \mathcal{D}) = 0$. An exact frequentist pivotal interval $[m, m \cdot \alpha^{-1/k}] = [50.0, 91.0]$ also respects the boundary and matches the Bayesian reciprocal credible interval.
2. **Moment Existence Thresholds:** Under a flat prior $\pi(N) \propto 1$, the posterior is improper for $k=1$, its expectation diverges for $k \le 2$, and its variance diverges for $k \le 3$. Conversely, the scale-invariant reference prior $\pi(N) \propto 1/N$ guarantees proper posteriors for all $k \ge 1$ and finite expectations for $k \ge 2$.
3. **The Informative Prior Trap & Hierarchical Remedy:** Single-parameter $\text{Poisson}(\mu)$ priors tie variance to the mean ($\sigma = \sqrt{\mu} \approx 141$ on $N=20,000$, $\text{CV} \approx 0.7\%$), behaving as a dogmatic delta function. When misspecified by 50\%, 95\% credible coverage collapses to $< 1.0\%$. We prove that hierarchical Gamma-Poisson hyperpriors expand hyper-variance, completely restoring nominal $94.8\%$ coverage.
4. **Estimand Disambiguation & Thinning Models:** In real transaction logs (such as Online Retail II), cancellations and billing voids create a **51.2\% unrecorded integer void rate**. Applying the raw maximum estimates the address allocation span ($N_{\text{span}} = 45,223$), NOT the net transaction count ($N_{\text{count}} = 22,061$). While rank re-indexing serves as a retrospective oracle benchmark, we formulate an operational **Bernoulli thinning model** ($N_{\text{count}} \sim \text{Binomial}(N_{\text{span}}, \rho)$ with $\hat{\rho} \approx 0.488$) that estimates both parameters simultaneously.
5. **Modern Cross-Domain Translation:** The discrete ceiling principle translates directly into five mission-critical engineering and scientific domains: dynamic production tracking, software defect release certification, code complexity feature selection, civil aviation fleet surveillance, and wildlife mark-recapture.

---

## Primary Project Documents

| Document | Format | Description | Direct Link |
|---|---|---|---|
| **Comprehensive Technical Report** | PDF / LaTeX | Full theoretical derivations, exploratory data analysis, decision-theoretic loss formulations, and empirical benchmark tables. | [report.pdf](report.pdf) · [report.tex](report.tex) |
| **Widescreen Presentation Deck** | PDF / Beamer | 23-slide widescreen deck designed with modern slate navy aesthetics, clean spacing, and standalone figures. | [presentation.pdf](presentation.pdf) · [presentation.tex](presentation.tex) |

---

## The 5 Modern Applications

To maintain a clean and focused core thesis on the primary invoice serial dataset, each modern application is detailed in its own standalone report with complete derivations, empirical benchmarks, and high-resolution diagnostic plots:

| # | Application Domain | Focus & Core Methodology | Key Empirical Result | Dedicated Deep Dive |
|---|---|---|---|---|
| **01** | **Dynamic Production Tracking** | Monotonic state-space filtering ($N_t \ge N_{t-1}, N_t \ge \max_{\tau \le t} m_\tau$) + Normal-Gamma predictive forecasting | **71.0\% RMSE reduction**; eliminates impossible negative rates; calibrated 24-week test coverage | [PDF](applications/application_1_dynamic_production.pdf) · [Report](applications/application_1_dynamic_production/application_1_dynamic_production.md) |
| **02** | **Software Reliability & Bug Exhaustion** | Closed-form discrete marginal likelihood $P(\mathbf{t} \mid N)$ integrating out fault hazard $\phi$; commercial release rule $P(N-k \le 15) \ge 0.95$ | **Reaches $94.2\%$ at $k=136$; certified at $k = 140$ ($P = 95.8\%$)**; avoids asymptotic MLE divergence | [PDF](applications/application_2_software_reliability.pdf) · [Report](applications/application_2_software_reliability/application_2_software_reliability.md) |
| **03** | **Code Defect Prioritization** | Conjugate Normal-Inverse-Gamma regression with collapsed Gibbs spike-and-slab variable selection | **Occam's Bayes Factor $BF \approx 10^{15}$**; PIPs $> 0.90$ for `max_cc` & `lcom` robust across Zellner $g$-priors; $r = 0.68$ out-of-sample | [PDF](applications/application_3_defect_regression.pdf) · [Report](applications/application_3_defect_regression/application_3_defect_regression.md) |
| **04** | **Civil Aviation Fleet Surveillance** | Semi-synthetic counterfeit injection benchmark on 9,102 FAA Cessnas; Dirichlet-Multinomial + Beta-Bernoulli mixture | **ROC AUC = 0.961**; detects in-range cloned parts completely invisible to classical maximum threshold rules | [PDF](applications/application_4_aviation_counterfeits.pdf) · [Report](applications/application_4_aviation_counterfeits/application_4_aviation_counterfeits.md) |
| **05** | **Ecological Wildlife Abundance** | Two-stage Bayesian cascade ($N_{\text{pop}} = M/p$) with strict physical lower bound $N_{\text{pop}} \ge 68$ (68 unique hares caught) | **Median 104.7 hares (95\% CI: [68.0, 284.5])**; encompasses Chapman ($97.6 \pm 30.8$) and log-linear models ($77-81$) while propagating tag log uncertainty | [PDF](applications/application_5_wildlife_abundance.pdf) · [Report](applications/application_5_wildlife_abundance/application_5_wildlife_abundance.md) |

---

## Core Empirical Leaderboard ($R = 4,000$ Monte Carlo Runs, $N_{\text{true}} = 20,000$, Seed 42)

| Estimator / Framework | Point Estimator | Sample $k$ | Mean Bias $\pm$ SE | RMSE (Emp / Th) | 95\% Interval Cov. $\pm$ SE | Mean CI Width | Boundary Violations |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **UMVUE (Frequentist)** | $\hat{N} = m(1 + 1/k) - 1$ | 5 | $-48.0 \pm 54.6$ | 3,455.2 / 3,380.3 | 90.0\% $\pm$ 0.47\% (Wald) | 13,219 | 100.0\% |
| **Exact Frequentist CI** | Pivotal Bound | 5 | — | — | **94.4\%** $\pm$ 0.36\% | 13,644 | **0.0\%** |
| **Bayes Flat Prior** | Posterior Mean | 5 | $+2,168.6 \pm 69.7$ | 4,409.0 / 4,363.9 | **97.4\%** $\pm$ 0.25\% | 18,535 | **0.0\%** |
| **Bayes Flat Prior** | Posterior Median | 5 | $-226.5 \pm 54.2$ | 3,431.3 / — | **97.4\%** $\pm$ 0.25\% | 18,535 | **0.0\%** |
| **Bayes Reciprocal ($1/N$)** | Posterior Mean | 5 | $+784.3 \pm 58.2$ | 3,683.3 / 3,618.6 | **94.4\%** $\pm$ 0.36\% | 13,644 | **0.0\%** |
| **Bayes Reciprocal ($1/N$)** | Posterior Median | 5 | $-900.0 \pm 54.2$ | **3,427.5** / — | **94.4\%** $\pm$ 0.36\% | **13,644** | **0.0\%** |
| \midrule | | | | | | | |
| **UMVUE (Frequentist)** | $\hat{N} = m(1 + 1/k) - 1$ | 50 | $+1.6 \pm 6.1$ | 386.8 / 391.7 | 95.0\% $\pm$ 0.34\% (Wald) | 1,536 | 100.0\% |
| **Exact Frequentist CI** | Pivotal Bound | 50 | — | — | **95.5\%** $\pm$ 0.33\% | 1,211 | **0.0\%** |
| **Bayes Reciprocal ($1/N$)** | Posterior Mean | 50 | $+10.6 \pm 6.1$ | **387.1** / 391.7 | **95.5\%** $\pm$ 0.33\% | **1,211** | **0.0\%** |
| **Bayes Flat Prior** | Posterior Mean | 50 | $+17.9 \pm 6.1$ | 387.5 / 391.7 | **95.8\%** $\pm$ 0.32\% | 1,236 | **0.0\%** |

---

## Repository Structure

```
├── README.md                                          # Master repository documentation
├── .gitignore                                         # Git ignore (LaTeX build, python cache, large files)
├── report.pdf                                         # Main comprehensive course report (PDF)
├── report.tex                                         # Main comprehensive course report (LaTeX)
├── presentation.pdf                                   # 23-slide widescreen beamer deck (PDF)
├── presentation.tex                                   # 23-slide widescreen beamer deck (LaTeX)
│
├── applications/                                      # Dedicated Modern Application Deep Dives
│   ├── README.md                                      # Applications overview and index
│   ├── application_1_dynamic_production.pdf           # App 1: Dynamic Production Tracking (PDF)
│   ├── application_2_software_reliability.pdf         # App 2: Software Reliability QA (PDF)
│   ├── application_3_defect_regression.pdf            # App 3: Code Defect Prioritization (PDF)
│   ├── application_4_aviation_counterfeits.pdf        # App 4: Aviation Fleet Surveillance (PDF)
│   ├── application_5_wildlife_abundance.pdf           # App 5: Wildlife Mark-Recapture (PDF)
│   ├── application_1_dynamic_production/             # App 1 source files (.tex, .pdf, .md)
│   ├── application_2_software_reliability/           # App 2 source files (.tex, .pdf, .md)
│   ├── application_3_defect_regression/              # App 3 source files (.tex, .pdf, .md)
│   ├── application_4_aviation_counterfeits/          # App 4 source files (.tex, .pdf, .md)
│   └── application_5_wildlife_abundance/             # App 5 source files (.tex, .pdf, .md)
│
├── scripts/                                           # Experiment runners and figure generators
│   ├── run_reproducible_benchmark.py                  # R=4,000 reproducible Monte Carlo benchmark
│   ├── run_wp1_experiments.py                         # Core Monte Carlo simulation & leaderboard
│   ├── run_wp2_experiments.py                         # Online Retail II invoice EDA & integer voids
│   ├── run_wp3_experiments.py                         # App 1: Dynamic production tracking
│   ├── run_wp4_experiments.py                         # App 2: Software reliability Musa JM model
│   ├── run_wp5_experiments.py                         # App 3: Code defect spike-and-slab regression
│   ├── run_wp7_experiments.py                         # App 4: Aviation counterfeit mixture model
│   ├── run_wp8_experiments.py                         # App 5: Snowshoe hare abundance cascade
│   └── create_expert_figures.py                       # Theoretical loss & boundary diagnostics
│
├── src/                                               # Reusable Python Bayesian inference library
│   ├── exact_posterior.py                             # Exact discrete likelihoods and posteriors
│   ├── frequentist.py                                 # UMVUE and asymptotic Wald intervals
│   ├── bayes_factors.py                               # Closed-form hypothesis testing for regulatory caps
│   ├── diagnostics.py                                 # MCMC diagnostics, ESS, and Gelman-Rubin
│   ├── evaluator.py                                   # Scoring rules (log-score, CRPS, coverage)
│   ├── bayesian_regression.py                         # Conjugate NIG & collapsed Gibbs spike-and-slab
│   ├── jelinski_moranda.py                            # Closed-form discrete JM reliability model
│   └── data_loader.py                                 # Preprocessing and rank re-indexing pipelines
│
├── data/                                              # Processed datasets used across the project
│   ├── d1_online_retail/                              # Online Retail invoice serial numbers
│   ├── d2_online_retail_ii/                           # Online Retail II multi-year weekly batches
│   ├── d3_musa/                                       # Musa software failure execution times
│   ├── d4_promise/                                    # PROMISE Apache Ant object-oriented metrics
│   ├── d7_faa_aircraft/                               # FAA civil aircraft registration registry
│   ├── d9_rcapture/                                   # Snowshoe hare mark-recapture logs
│   └── historical_wwii_tanks.csv                      # Historical Speer Ministry production records
│
├── figures/                                           # High-resolution generated publication figures
├── results/                                           # Raw Monte Carlo simulation outputs (CSV)
└── archive/                                           # Historical background papers and early proposals
```

---

## Quickstart: Setup & Reproduction

### 1. Environment Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/RajveerVoraUni/german-tanks.git
cd german-tanks

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install numpy scipy matplotlib
```

### 2. Running the Reproducible Benchmark
To reproduce the full Monte Carlo simulation table ($R = 4,000$ replications, seed 42):
```bash
python scripts/run_reproducible_benchmark.py
```
Results will be output directly to `results/reproduced_benchmark_table.csv`.

### 3. Compiling LaTeX Documents
Ensure TeX Live / MacTeX is installed, then compile the main report, slides, and application deep dives:
```bash
# Compile main comprehensive report
pdflatex -interaction=nonstopmode report.tex

# Compile 23-slide widescreen beamer presentation
pdflatex -interaction=nonstopmode presentation.tex

# Compile application deep-dive reports
cd applications/application_1_dynamic_production && pdflatex application_1_dynamic_production.tex && cd ../..
cd applications/application_2_software_reliability && pdflatex application_2_software_reliability.tex && cd ../..
cd applications/application_3_defect_regression && pdflatex application_3_defect_regression.tex && cd ../..
cd applications/application_4_aviation_counterfeits && pdflatex application_4_aviation_counterfeits.tex && cd ../..
cd applications/application_5_wildlife_abundance && pdflatex application_5_wildlife_abundance.tex && cd ../..
```

---

## References

1. **Ruggles, R., & Brodie, H. (1947).** An Empirical Approach to Economic Intelligence in World War II. *Journal of the American Statistical Association*, 42(237), 72–91.
2. **Goodman, L. A. (1954).** Some Practical Techniques in Serial Number Analysis. *Journal of the American Statistical Association*, 49(265), 97–112.
3. **Jeffreys, H. (1961).** *Theory of Probability* (3rd ed.). Oxford University Press.
4. **Berger, J. O. (1985).** *Statistical Decision Theory and Bayesian Analysis*. Springer.
5. **Bernardo, J. M., & Smith, A. F. (1994).** *Bayesian Theory*. John Wiley & Sons.
6. **Gelman, A., et al. (2013).** *Bayesian Data Analysis* (3rd ed.). CRC Press.
7. **Jelinski, Z., & Moranda, P. (1972).** Software Reliability Research. In *Statistical Computer Performance Evaluation*, Academic Press, 465–484.
8. **George, E. I., & McCulloch, R. E. (1993).** Variable Selection via Gibbs Sampling. *Journal of the American Statistical Association*, 88(423), 881–889.
9. **Otis, D. L., et al. (1978).** Statistical Inference from Capture Data on Closed Animal Populations. *Wildlife Monographs*, (62), 3–135.
