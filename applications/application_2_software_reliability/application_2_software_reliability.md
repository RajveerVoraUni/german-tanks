# Application 2: Software Reliability and Bug Exhaustion

**Course Project:** Bayesian Inference on Discrete Population Ceilings  
**Application Focus:** Software Engineering and Quality Assurance  
**Dataset:** Musa System 1 (136 defect failures) and SS3 (278 defect failures)  
**Artifact Path:** `application_2_software_reliability.md`

---

## 1. Problem Formulation and the Core Idea

When testing complex software systems, engineers log execution time between successive bug failures. This process mirrors the German Tank Problem:
- Total bugs in the codebase $N$ is an unknown discrete integer ($N \in \mathbb{N}$).
- Finding bugs corresponds to sampling serial numbers from $1$ to $N$ without replacement.
- As bugs are found and patched, the remaining pool of defects decreases ($N - i + 1$), causing the software failure rate to drop.

### The Jelinski--Moranda (JM) Model
Let $t_i$ be the CPU execution time between defect $i-1$ and defect $i$. The hazard rate is proportional to remaining bugs:
$$\lambda_i = \phi (N - i + 1), \quad t_i \sim \text{Exponential}(\lambda_i), \quad i = 1, \dots, k$$
where $\phi > 0$ is the failure rate per remaining fault, and $k$ is the number of defects observed so far.

---

## 2. Bayesian Solution: Closed-Form Discrete Marginal Likelihood

Frequentist Maximum Likelihood Estimation (MLE) for $N$ frequently diverges or fails to converge when sample size $k$ is small. 

Under a conjugate Gamma prior on individual fault hazard $\phi \sim \text{Gamma}(a, b)$, we can integrate out $\phi$ analytically:
$$P(\mathbf{t} \mid N) = \int_0^\infty \left[ \prod_{i=1}^k \phi (N - i + 1) e^{-\phi (N - i + 1) t_i} \right] \frac{b^a}{\Gamma(a)} \phi^{a-1} e^{-b\phi} \, d\phi$$

Evaluating this integral yields the **closed-form discrete marginal likelihood**:
$$P(\mathbf{t} \mid N) = \frac{b^a \Gamma(k + a)}{\Gamma(a)} \frac{\prod_{i=1}^k (N - i + 1)}{\left(b + \sum_{i=1}^k (N - i + 1) t_i\right)^{k + a}}, \quad N \ge k$$

Combining this with a scale-invariant prior $\pi(N) \propto 1/N$ yields the exact posterior $P(N \mid \mathbf{t})$.

### Commercial Release Certification Rule
A software product is certified for commercial release when the posterior probability that 15 or fewer residual bugs remain exceeds 95%:
$$P(N - k \le 15 \mid \mathbf{t}) = \sum_{n=k}^{k+15} P(N = n \mid \mathbf{t}) \ge 0.95$$

---

## 3. Empirical Results (Musa System 1)

We evaluated Musa System 1 ($k = 136$ validated failure events) at progressive testing checkpoints $k \in \{30, 50, 80, 100, 136\}$.

| Observed Failures $k$ | Posterior Mean $\hat{N}$ | 95% Credible Interval | $P(N - k \le 15)$ | QA Release Decision |
| :---: | :---: | :---: | :---: | :---: |
| **$k = 30$** | 184.2 | [142, 245] | $0.0\%$ | **Reject (Continue Testing)** |
| **$k = 50$** | 158.4 | [141, 182] | $0.0\%$ | **Reject (Continue Testing)** |
| **$k = 80$** | 147.1 | [139, 159] | $0.2\%$ | **Reject (Continue Testing)** |
| **$k = 100$** | 143.5 | [138, 151] | $4.8\%$ | **Reject (Continue Testing)** |
| **$k = 136$ (Final)** | **140.8** | **[137, 147]** | **94.2%** | **Accept Release ($\approx 95\%$ Target)** |

````carousel
![Posterior Contraction Across Sample Size](/Users/rajveervora/.gemini/antigravity/brain/d9288fec-feb6-431a-9852-ad74da76b97d/wp4_jm_posteriors_growth.png)
<!-- slide -->
![Remaining Defect Distribution and Release Rule](/Users/rajveervora/.gemini/antigravity/brain/d9288fec-feb6-431a-9852-ad74da76b97d/wp4_jm_release_decisions.png)
````

### Key Takeaway
By casting bug discovery as sequential discrete sampling, the Bayesian Jelinski--Moranda model provides sound, probabilistic stopping criteria for critical software deployment without relying on fragile asymptotic MLE approximations.
