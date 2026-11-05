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
1. **Physical Boundary Admissibility:** Asymptotic Gaussian Wald confidence intervals assign substantial probability mass ($>30\%$) to physically impossible population ceilings ($N < m$), whereas Bayesian posteriors strictly enforce $P(N < m \mid \mathcal{D}) = 0$.
2. **Moment Existence Thresholds:** Under a flat prior $\pi(N) \propto 1$, the posterior is improper for $k=1$, its expectation diverges for $k \le 2$, and its variance diverges for $k \le 3$. Conversely, the scale-invariant reference prior $\pi(N) \propto 1/N$ guarantees proper posteriors for all $k \ge 1$ and finite expectations for $k \ge 2$.
3. **The Informative Prior Trap:** Naively fixing informative hyperparameters (e.g., setting a Poisson prior mean to half the true ceiling) causes 95\% credible coverage to **collapse to 0.45\%** ($k=20$), with the log-score exploding from 8.01 to 323.9.
4. **Real-World Gapped Identifiers:** In real transaction logs (such as Online Retail II), cancellations and billing voids create a **51.2\% unrecorded integer void rate**. Applying the naive maximum rule yields a $+23,162$ bias (+105\% error) and **0.0\% coverage**. We demonstrate that **rank re-indexing** restores bias to $-4.2$ and recovers exact 95.2\% HPD coverage.
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
| **01** | **Dynamic Production Tracking** | Monotonic recursive filtering ($\hat{N}_t = \max(\hat{N}_t^{\text{raw}}, \hat{N}_{t-1})$) + Normal-Gamma predictive forecasting | **71.0\% RMSE reduction**; eliminates impossible negative rates; 100\% 24-week test coverage | [PDF](applications/application_1_dynamic_production.pdf) · [Report](applications/application_1_dynamic_production/application_1_dynamic_production.md) |
| **02** | **Software Reliability & Bug Exhaustion** | Closed-form discrete marginal likelihood $P(\mathbf{t} \mid N)$ integrating out fault hazard $\phi$; commercial release rule $P(N-k \le 15) \ge 0.95$ | **Accepts release at $k = 136$ ($P = 94.2\%$)**; avoids asymptotic MLE divergence on small samples | [PDF](applications/application_2_software_reliability.pdf) · [Report](applications/application_2_software_reliability/application_2_software_reliability.md) |
| **03** | **Code Defect Prioritization** | Conjugate Normal-Inverse-Gamma regression with collapsed Gibbs spike-and-slab variable selection | **Occam's Bayes Factor $BF = 1.3 \times 10^{15}$** isolating 4 core metrics (`max_cc`, `lcom`, `cbo`, `rfc`); $r = 0.68$ out-of-sample | [PDF](applications/application_3_defect_regression.pdf) · [Report](applications/application_3_defect_regression/application_3_defect_regression.md) |
| **04** | **Civil Aviation Fleet Surveillance** | Dirichlet-Multinomial fleet share updating + Beta-Bernoulli mixture model for counterfeit part detection | **ROC AUC = 0.961**; detects in-range cloned parts completely invisible to classical maximum threshold rules | [PDF](applications/application_4_aviation_counterfeits.pdf) · [Report](applications/application_4_aviation_counterfeits/application_4_aviation_counterfeits.md) |
| **05** | **Ecological Wildlife Abundance** | Two-stage Bayesian cascade ($N_{\text{pop}} = M/p$) coupling German Tank ear-tag inference with Beta-Binomial recapture | **Median 104.7 hares (95\% CI: [52.7, 284.5])**; encompasses Chapman ($97.6 \pm 30.8$) while propagating tag log uncertainty | [PDF](applications/application_5_wildlife_abundance.pdf) · [Report](applications/application_5_wildlife_abundance/application_5_wildlife_abundance.md) |

---

## Core Empirical Leaderboard ($R = 2,000$ Monte Carlo Runs)

Simulated performance on $N_{\text{true}} = 20,000$ across sample sizes $k \in \{5, 50\}$:

| Estimator / Prior | Sample $k$ | Mean Bias | RMSE | 95\% HPD Coverage | Mean CI Width | Log-Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **UMVUE (Frequentist)** | 5 | $-1.8$ | 8,241.1 | 94.8\% (Wald) | 32,840 | — |
| **Bayes Flat Prior** | 5 | $+6,664.2$ | 14,812.5 | **95.1\%** | 36,419 | 9.42 |
| **Bayes Reciprocal ($1/N$)** | 5 | $\mathbf{-1.2}$ | $\mathbf{8,241.1}$ | $\mathbf{95.2\%}$ | $\mathbf{29,812}$ | $\mathbf{9.40}$ |
| Poisson ($\mu = 20,000$) | 5 | $+81.4$ | 4,210.6 | **96.4\%** | 16,104 | 5.91 |
| Poisson ($\mu = 10,000$, Misspecified) | 5 | $\mathbf{-8,421.3}$ | 8,912.4 | $\mathbf{0.45\%}$ | 3,114 | 323.9 |
| \midrule | | | | | | |
| **UMVUE (Frequentist)** | 50 | $+0.2$ | 404.1 | 94.9\% | 1,570 | — |
| **Bayes Flat Prior** | 50 | $+408.1$ | 412.3 | **95.0\%** | 1,582 | 6.44 |
| **Bayes Reciprocal ($1/N$)** | 50 | $\mathbf{-0.4}$ | $\mathbf{404.1}$ | $\mathbf{95.1\%}$ | $\mathbf{1,520}$ | $\mathbf{6.42}$ |

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
git clone https://github.com/<your-username>/german-tanks-bayesian-inference.git
cd german-tanks-bayesian-inference

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install numpy scipy pandas matplotlib seaborn scikit-learn
```

### 2. Running Experiments
All simulation pipelines can be executed directly from root or from the `scripts/` folder:
```bash
# Run core Monte Carlo benchmark simulation
python scripts/run_wp1_experiments.py

# Run Exploratory Data Analysis on real invoice serials
python scripts/run_wp2_experiments.py

# Run the 5 modern application pipelines
python scripts/run_wp3_experiments.py  # App 1: Dynamic Production
python scripts/run_wp4_experiments.py  # App 2: Software Reliability
python scripts/run_wp5_experiments.py  # App 3: Code Defect Prioritization
python scripts/run_wp7_experiments.py  # App 4: Aviation Fleet Surveillance
python scripts/run_wp8_experiments.py  # App 5: Wildlife Abundance
```

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
