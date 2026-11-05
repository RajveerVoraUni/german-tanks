# Application 5: Ecological Wildlife Abundance Modeling

**Course Project:** Bayesian Inference on Discrete Population Ceilings  
**Application Focus:** Quantitative Ecology and Conservation Biology  
**Dataset:** Snowshoe Hare Capture-Recapture (*Rcapture* Study, 68 unique hares across 6 trapping occasions)  
**Artifact Path:** `application_5_wildlife_abundance.md`

---

## 1. Problem Formulation and the Core Idea

Estimating wildlife population abundance ($N_{\text{pop}}$) is central to ecological conservation. In standard Lincoln-Petersen and Chapman mark-recapture studies:
1. Animals are captured on Occasion 1, marked with sequential numbered ear tags ($1$ to $M$), and released.
2. On Occasion 2, animals are trapped, and the number of marked recaptures $r$ out of total catch $n$ is recorded.

### The Imperfect Tag Tracking Problem
Textbook mark-recapture assumes that the initial marked total $M$ is known with 100% certainty. In remote field biology, tag distribution logs are frequently lost, partially recorded, or illegible.
- If researchers only observe a sample of $k$ tag numbers with maximum $m_{\text{tag}}$, then **the total marked pool $M$ is itself an unknown discrete parameter**!
- Classical estimators (such as the Chapman index) cannot propagate uncertainty in $M$, leading to artificially narrow confidence intervals and uncalibrated risk assessments.

---

## 2. Bayesian Solution: Two-Stage Composed Cascade

We establish a two-stage Bayesian cascade that models total abundance $N_{\text{pop}} = M / p$:

### 2.1 Stage 1: German Tank Tag Inference
Given $k$ recorded tag serials with maximum $m = \max(s_i)$, the unknown marked total $M \in \mathbb{N}$ follows the exact German Tank posterior:
$$P(M = n \mid m, k) \propto \frac{1}{\binom{n}{k}} \pi(M)$$
Under a scale-invariant reciprocal prior $\pi(M) \propto 1/M$:
$$\mathbb{E}[M \mid m, k] = m - 1$$

### 2.2 Stage 2: Beta-Binomial Recapture Updating
On Occasion 2, $n$ hares are trapped, with $r$ marked individuals:
$$r \mid M, p \sim \text{Binomial}(n, p), \quad p \sim \text{Beta}(\alpha_0, \beta_0)$$
Under conjugate updating:
$$p \mid r, n \sim \text{Beta}(\alpha_0 + r, \beta_0 + n - r)$$

### 2.3 Posterior Composition
To generate samples from the composed joint posterior of total population $N_{\text{pop}}$:
1. Draw $M^{(s)} \sim P(M \mid m, k)$ from the discrete German Tank posterior.
2. Draw $p^{(s)} \sim \text{Beta}(\alpha_0 + r, \beta_0 + n - r)$.
3. Compute $N_{\text{pop}}^{(s)} = M^{(s)} / p^{(s)}$.

---

## 3. Empirical Results (Snowshoe Hare Dataset)

Using the canonical Snowshoe Hare study ($68$ unique animals observed over 6 trapping occasions):
- Occasion 1: $M = 16$ marked animals deployed.
- Occasion 2: $n = 28$ hares trapped, with $r = 4$ marked recaptures.
- Incomplete tag record scenario: $k = 10$ tag numbers observed with maximum $m = 16$.

| Estimation Method | Abundance Estimate | Standard Error | 95% Confidence / Credible Interval |
| :--- | :---: | :---: | :---: |
| **Frequentist Chapman Benchmark** | **97.6** | **30.8** | [37.3, 157.9] |
| **Stage 1 Tag Posterior ($M$)** | 16.8 | 1.1 | [16, 19] |
| **Stage 2 Capture Probability ($p$)** | 0.167 | 0.068 | [0.054, 0.317] |
| **Two-Stage Composed Bayesian ($N_{\text{pop}}$)** | **104.7 (Median)** | **69.1** | **[52.7, 284.5]** |

![Wildlife Abundance Posterior](/Users/rajveervora/.gemini/antigravity/brain/d9288fec-feb6-431a-9852-ad74da76b97d/wp8_wildlife_abundance.png)

### Key Takeaway
The two-stage Bayesian posterior naturally incorporates the discrete variance in total deployed tags, providing a realistic, right-skewed credible interval $[52.7, 284.5]$ that encompasses the classical Chapman estimate while protecting against underestimation when tag deployments are uncertain.
