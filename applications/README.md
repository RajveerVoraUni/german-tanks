# Modern Applications of Discrete Population Inference

This directory contains standalone, in-depth technical deep dives extending the core Bayesian German Tank discrete estimation framework into five modern applied engineering and scientific domains.

Each application has its own dedicated directory containing:
1. **A compiled PDF report** typeset in LaTeX with exact mathematical formulations, tables, and embedded high-resolution figures.
2. **The LaTeX source file** (`.tex`) for full reproducible compilation.
3. **The technical markdown report** (`.md`) formatted for instant GitHub and online viewing.

---

## Applications Overview

| # | Domain & Application | Focus & Core Methodology | Key Empirical Result | Dedicated Artifacts |
|---|---|---|---|---|
| **01** | **Dynamic Production Tracking** | Monotonic recursive filtering ($\hat{N}_t = \max(\hat{N}_t^{\text{raw}}, \hat{N}_{t-1})$) + Normal-Gamma predictive forecasting | **71.0\% RMSE reduction**; eliminates impossible negative rates; 100\% 24-week test coverage | [PDF](application_1_dynamic_production.pdf) · [TeX](application_1_dynamic_production/application_1_dynamic_production.tex) · [MD](application_1_dynamic_production/application_1_dynamic_production.md) |
| **02** | **Software Reliability & Bug Exhaustion** | Closed-form discrete marginal likelihood $P(\mathbf{t} \mid N)$ integrating out fault hazard $\phi$; commercial release rule $P(N-k \le 15) \ge 0.95$ | **Accepts release at $k = 136$ ($P = 94.2\%$)**; avoids asymptotic MLE divergence on small samples | [PDF](application_2_software_reliability.pdf) · [TeX](application_2_software_reliability/application_2_software_reliability.tex) · [MD](application_2_software_reliability/application_2_software_reliability.md) |
| **03** | **Code Defect Prioritization** | Conjugate Normal-Inverse-Gamma regression with collapsed Gibbs spike-and-slab variable selection | **Occam's Bayes Factor $BF = 1.3 \times 10^{15}$** isolating 4 core metrics (`max_cc`, `lcom`, `cbo`, `rfc`); $r = 0.68$ out-of-sample | [PDF](application_3_defect_regression.pdf) · [TeX](application_3_defect_regression/application_3_defect_regression.tex) · [MD](application_3_defect_regression/application_3_defect_regression.md) |
| **04** | **Civil Aviation Fleet Surveillance** | Dirichlet-Multinomial fleet share updating + Beta-Bernoulli mixture model for counterfeit part detection | **ROC AUC = 0.961**; detects in-range cloned parts completely invisible to classical maximum threshold rules | [PDF](application_4_aviation_counterfeits.pdf) · [TeX](application_4_aviation_counterfeits/application_4_aviation_counterfeits.tex) · [MD](application_4_aviation_counterfeits/application_4_aviation_counterfeits.md) |
| **05** | **Ecological Wildlife Abundance** | Two-stage Bayesian cascade ($N_{\text{pop}} = M/p$) coupling German Tank ear-tag inference with Beta-Binomial recapture | **Median 104.7 hares (95\% CI: [52.7, 284.5])**; encompasses Chapman ($97.6 \pm 30.8$) while propagating tag log uncertainty | [PDF](application_5_wildlife_abundance.pdf) · [TeX](application_5_wildlife_abundance/application_5_wildlife_abundance.tex) · [MD](application_5_wildlife_abundance/application_5_wildlife_abundance.md) |

---

## Directory Organization

```
applications/
├── README.md                                          # This summary index
├── application_1_dynamic_production.pdf               # Application 1 standalone PDF
├── application_2_software_reliability.pdf             # Application 2 standalone PDF
├── application_3_defect_regression.pdf                # Application 3 standalone PDF
├── application_4_aviation_counterfeits.pdf            # Application 4 standalone PDF
├── application_5_wildlife_abundance.pdf               # Application 5 standalone PDF
├── application_1_dynamic_production/
│   ├── application_1_dynamic_production.tex           # LaTeX source
│   ├── application_1_dynamic_production.pdf           # Compiled PDF report
│   └── application_1_dynamic_production.md           # Markdown report
├── application_2_software_reliability/
│   ├── application_2_software_reliability.tex
│   ├── application_2_software_reliability.pdf
│   └── application_2_software_reliability.md
├── application_3_defect_regression/
│   ├── application_3_defect_regression.tex
│   ├── application_3_defect_regression.pdf
│   └── application_3_defect_regression.md
├── application_4_aviation_counterfeits/
│   ├── application_4_aviation_counterfeits.tex
│   ├── application_4_aviation_counterfeits.pdf
│   └── application_4_aviation_counterfeits.md
└── application_5_wildlife_abundance/
    ├── application_5_wildlife_abundance.tex
    ├── application_5_wildlife_abundance.pdf
    └── application_5_wildlife_abundance.md
```
