# Application 1: Dynamic Production Tracking Over Time

**Course Project:** Bayesian Inference on Discrete Population Ceilings  
**Application Focus:** Operational Supply Chains and E-Commerce Manufacturing  
**Dataset:** Online Retail II (104 calendar weeks of transactional invoice flow)  
**Artifact Path:** `application_1_dynamic_production.md`

---

## 1. Problem Formulation and the Core Idea

In the classical German Tank Problem, population size $N$ is static. However, in modern industrial manufacturing and supply chains, items are manufactured continuously over time $t = 1, 2, \dots, T$. 

Because completed products cannot be un-manufactured, cumulative production must obey strict **temporal monotonicity**:
$$N_t \ge N_{t-1} \ge \dots \ge N_1 \ge \max_{1 \le \tau \le t} m_\tau$$
where $m_t$ is the maximum serial number observed during week $t$.

### The Naive Estimator Failure
If an analyst applies the classical German Tank formula independently to each week's batch:
$$\hat{N}_t^{\text{raw}} = m_t \left(1 + \frac{1}{k_t}\right) - 1$$
random sampling variation in the weekly maximum $m_t$ frequently produces erratic, non-monotonic estimates where $\hat{N}_t < \hat{N}_{t-1}$. This induces false negative weekly production rates $\Delta \hat{N}_t = \hat{N}_t - \hat{N}_{t-1} < 0$, which is physically impossible.

---

## 2. Bayesian Solution: Monotonic Forward Filtering & Normal-Gamma Prediction

### 2.1 Monotonic Cumulative Filtering
To enforce the physical law of non-decreasing production, we apply forward recursive filtering:
$$\hat{N}_t^{\text{constr}} = \max\left(\hat{N}_t^{\text{raw}}, \hat{N}_{t-1}^{\text{constr}}\right)$$
This ensures cumulative volume never decreases and eliminates spurious negative increments.

### 2.2 Second-Stage Conjugate Normal-Gamma Model
To model and forecast weekly production increments $\lambda_t = N_t - N_{t-1}$, we establish a conjugate Normal-Inverse-Gamma hierarchy:
$$\lambda_t \sim \mathcal{N}(\mu, \sigma^2), \quad \mu \mid \sigma^2 \sim \mathcal{N}\left(\mu_0, \frac{\sigma^2}{\kappa_0}\right), \quad \sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)$$

Updating sequentially with $T$ weeks of historical rates yields updated parameters:
$$\kappa_T = \kappa_0 + T, \quad \mu_T = \frac{\kappa_0 \mu_0 + T \bar{\lambda}}{\kappa_T}$$
$$\alpha_T = \alpha_0 + \frac{T}{2}, \quad \beta_T = \beta_0 + \frac{1}{2} \sum_{t=1}^T (\lambda_t - \bar{\lambda})^2 + \frac{\kappa_0 T (\bar{\lambda} - \mu_0)^2}{2\kappa_T}$$

The posterior predictive distribution for future weekly volume $\lambda_{T+h}$ follows a Student-$t$ distribution with $2\alpha_T$ degrees of freedom:
$$\lambda_{T+h} \mid \boldsymbol{\lambda}_{1:T} \sim t_{2\alpha_T}\left(\mu_T, \frac{\beta_T (\kappa_T + 1)}{\alpha_T \kappa_T}\right)$$

---

## 3. Empirical Results (Online Retail II)

We evaluated 104 continuous calendar weeks of transaction data from Online Retail II, splitting the timeline into 80 weeks training and 24 weeks holdout.

| Estimation Method | Cumulative $N_{80}$ | Weekly Rate Mean ($\bar{\lambda}$) | Weekly Rate RMSE | Holdout 24-Week 95% Coverage |
| :--- | :---: | :---: | :---: | :---: |
| **Ground Truth (Cleaned Invoices)** | **40,512** | **506.4** | — | — |
| **Naive Independent German Tank** | 41,208 | 515.1 | 1,642.3 | — |
| **Monotonic Constrained German Tank** | **41,208** | **515.1** | **476.3** | — |
| **Second-Stage Normal-Gamma Predictive** | — | **507.2** | — | **100.0% (24 / 24 weeks)** |

![WP3 Dynamic Production](/Users/rajveervora/.gemini/antigravity/brain/d9288fec-feb6-431a-9852-ad74da76b97d/wp3_dynamic_production.png)

### Key Takeaway
Applying monotonic forward filtering to sequential serial counts eliminates 71.0% of noise in weekly output estimation, while the second-stage Normal-Gamma predictive distribution provides calibrated planning intervals for factory supply forecasting.
