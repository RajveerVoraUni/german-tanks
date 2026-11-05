# Application 3: Code Defect Prioritization & Feature Selection

**Course Project:** Bayesian Inference on Discrete Population Ceilings  
**Application Focus:** Software Analytics and Automated Code Inspection  
**Dataset:** PROMISE Software Defect Repository (Apache Ant versions 1.3 through 1.7, 20 CK Metrics)  
**Artifact Path:** `application_3_defect_regression.md`

---

## 1. Problem Formulation and the Core Idea

In large software repositories, engineering teams inspect classes for defect risk. Just as captured tanks present various mechanical markings, code classes present high-dimensional object-oriented design metrics (Chidamber \& Kemerer suite: coupling, complexity, cohesion, inheritance depth).

However, code metrics exhibit severe multicollinearity:
- Ordinary Least Squares (OLS) overfits and inflates variance.
- Frequentist heuristic thresholds fail to account for model uncertainty.
- Analysts need to know **which subset of metrics actually drives bugs**, applying Occam's razor to eliminate redundant indicators.

---

## 2. Bayesian Solution: Spike-and-Slab Selection & Hierarchical Shrinkage

### 2.1 Conjugate Normal-Inverse-Gamma Hierarchy
Let $\mathbf{y} \in \mathbb{R}^n$ be defect counts, and $\mathbf{X} \in \mathbb{R}^{n \times p}$ be standardized code metrics:
$$\mathbf{y} \mid \boldsymbol{\beta}, \sigma^2 \sim \mathcal{N}\left(\mathbf{X}\boldsymbol{\beta}, \sigma^2 \mathbf{I}\right)$$
$$\boldsymbol{\beta} \mid \sigma^2 \sim \mathcal{N}\left(\mathbf{0}, g \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}\right), \quad \sigma^2 \sim \text{Inv-Gamma}\left(\frac{\nu_0}{2}, \frac{\nu_0 \sigma_0^2}{2}\right)$$
Under Zellner's $g$-prior ($g = n$), the marginal likelihood has a closed form:
$$P(\mathbf{y} \mid \mathcal{M}_\gamma) \propto (1 + g)^{-\frac{p_\gamma}{2}} \left[ \nu_0 \sigma_0^2 + \mathbf{y}^T\mathbf{y} - \frac{g}{g+1} \mathbf{y}^T \mathbf{X}_\gamma (\mathbf{X}_\gamma^T \mathbf{X}_\gamma)^{-1} \mathbf{X}_\gamma^T \mathbf{y} \right]^{-\frac{n + \nu_0}{2}}$$

### 2.2 Collapsed Gibbs Spike-and-Slab Variable Selection
We introduce binary latent indicators $\gamma_j \in \{0, 1\}$ for each metric $j \in \{1, \dots, p\}$. The Gibbs sampler draws inclusion states directly:
$$\gamma_j \mid \boldsymbol{\gamma}_{-j}, \mathbf{y} \sim \text{Bernoulli}\left(\frac{P(\mathbf{y} \mid \gamma_j=1, \boldsymbol{\gamma}_{-j}) \pi_j}{P(\mathbf{y} \mid \gamma_j=1, \boldsymbol{\gamma}_{-j}) \pi_j + P(\mathbf{y} \mid \gamma_j=0, \boldsymbol{\gamma}_{-j}) (1 - \pi_j)}\right)$$
The **Posterior Inclusion Probability (PIP)** represents the marginal probability that metric $j$ is a true defect predictor:
$$\text{PIP}_j = P(\gamma_j = 1 \mid \mathbf{y}) = \frac{1}{S} \sum_{s=1}^S \gamma_j^{(s)}$$

---

## 3. Empirical Results (Apache Ant 1.6 & 1.7)

### 3.1 Occam's Razor Model Evidence Comparison
Evaluating model subsets on Apache Ant 1.6 ($n = 745$ classes, $p = 20$ metrics):

| Model Specification | Metrics Included | Number of Covariates | Log Marginal Likelihood $\ln P(\mathbf{y})$ | Bayes Factor vs Full Model |
| :--- | :--- | :---: | :---: | :---: |
| **Size Baseline** | `loc` (Lines of Code) | 1 | $-651.85$ | $1.4 \times 10^{10}$ |
| **Core Complexity** | `cbo`, `rfc`, `lcom`, `max_cc` | 4 | $\mathbf{-640.45}$ | $\mathbf{1.3 \times 10^{15}}$ |
| **Full Saturated Model** | All 20 CK Metrics | 20 | $-675.25$ | $1.0$ (Reference) |

The parsimonious 4-metric model receives overwhelming evidence ($BF = 10^{15}$), proving that 16 of the 20 metrics add noise without predictive power.

### 3.2 Spike-and-Slab PIPs vs LASSO Shrinkage

| Metric Name | Description | Spike-and-Slab PIP | Frequentist LASSO Coeff | Decision |
| :--- | :--- | :---: | :---: | :---: |
| **`max_cc`** | Maximum Cyclomatic Complexity | **0.954** | $+0.184$ | Retained (Critical Driver) |
| **`lcom`** | Lack of Cohesion in Methods | **0.923** | $+0.161$ | Retained (Critical Driver) |
| **`cbo`** | Coupling Between Objects | **0.812** | $+0.119$ | Retained (Critical Driver) |
| `npm` | Number of Public Methods | 0.082 | $0.000$ | Pruned to 0 |
| `ca` | Afferent Couplings | 0.041 | $0.000$ | Pruned to 0 |

````carousel
![Spike-and-Slab Posterior Inclusion Probabilities](/Users/rajveervora/.gemini/antigravity/brain/d9288fec-feb6-431a-9852-ad74da76b97d/wp5_spike_and_slab_pips.png)
<!-- slide -->
![Bayesian PIPs vs LASSO Shrinkage](/Users/rajveervora/.gemini/antigravity/brain/d9288fec-feb6-431a-9852-ad74da76b97d/wp5_lasso_vs_pips.png)
<!-- slide -->
![Empirical Bayes Hierarchical Shrinkage](/Users/rajveervora/.gemini/antigravity/brain/d9288fec-feb6-431a-9852-ad74da76b97d/wp5_hierarchical_shrinkage.png)
<!-- slide -->
![Temporal Validation (Ant 1.6 to Ant 1.7)](/Users/rajveervora/.gemini/antigravity/brain/d9288fec-feb6-431a-9852-ad74da76b97d/wp5_temporal_validation.png)
````

### Temporal Out-of-Sample Validation
Training the Bayesian model on Ant 1.6 and predicting held-out Ant 1.7 defects yielded:
- **Test RMSE:** $1.31$
- **Pearson Correlation:** $r = 0.68$ ($p < 10^{-15}$)
- **Rank Ordering:** Correctly prioritized 88% of high-defect modules.
