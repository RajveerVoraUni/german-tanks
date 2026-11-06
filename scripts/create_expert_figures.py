import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc
import seaborn as sns
import math

os.makedirs('figures', exist_ok=True)
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'figure.dpi': 200})

# -------------------------------------------------------------
# 1. Compute Exact Discrete Values for Loss Functions (m=100, k=5)
# -------------------------------------------------------------
k = 5
m = 100
n_grid = np.arange(m, 350)
p = (k - 1) / k * sc.comb(m - 1, k - 1) / sc.comb(n_grid, k)
p /= np.sum(p)

mean_val = (k - 1) * (m - 1) / (k - 2) # 132.0
umvue_val = m * (1 + 1.0 / k) - 1      # 119
map_val = m                            # 100

# Analytical exact discrete median and 90% quantile from CDF F(n) = 1 - binom(m-1, k-1)/binom(n, k-1)
# Median: binom(n, 4) >= 2 * binom(99, 4) => n = 118
# 90% Quantile: binom(n, 4) >= 10 * binom(99, 4) => n = 175
med_val = 118
q90_val = 175

print(f"Panel (a) Values: Mode={map_val}, Med={med_val}, UMVUE={umvue_val}, Mean={mean_val:.1f}, Q90={q90_val}")

# -------------------------------------------------------------
# 2. Panel (b) Wald vs Bayesian vs Exact Frequentist (m=50, k=5)
# -------------------------------------------------------------
m_ex = 50
k_ex = 5
wald_mean = m_ex * (1 + 1.0/k_ex) - 1 # 59.0
wald_se = np.sqrt((wald_mean - k_ex) * (wald_mean + 1.0) / (k_ex * (k_ex + 2.0))) # 9.62
wald_lower = wald_mean - 1.96 * wald_se # 40.14
wald_upper = wald_mean + 1.96 * wald_se # 77.86

# Probability mass below m_ex = 50 in Normal(59, 9.62)
z_bound = (m_ex - wald_mean) / wald_se
mass_below = sc.ndtr(z_bound) # ~0.1747 (17.5%)
print(f"Panel (b) Wald: Mean={wald_mean}, SE={wald_se:.2f}, CI=[{wald_lower:.1f}, {wald_upper:.1f}], Mass below m={mass_below*100:.1f}%")

wald_x = np.linspace(15, 120, 500)
wald_pdf = 1.0 / (np.sqrt(2*np.pi)*wald_se) * np.exp(-0.5 * ((wald_x - wald_mean)/wald_se)**2)

n_b = np.arange(m_ex, 150)
p_b = (k_ex - 1) / k_ex * sc.comb(m_ex - 1, k_ex - 1) / sc.comb(n_b, k_ex)
p_b /= np.sum(p_b)

exact_freq_upper = m_ex * (0.05 ** (-1.0 / k_ex)) # 91.03

# -------------------------------------------------------------
# 3. Create Combined 4-Panel Figure
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a)
ax1 = axes[0, 0]
ax1.plot(n_grid, p, color='#1f77b4', lw=2.5, label=r'Posterior $P(N \mid m=100, k=5)$')
ax1.fill_between(n_grid, 0, p, color='#1f77b4', alpha=0.15)
ax1.axvline(map_val, color='#d62728', linestyle=':', lw=2, label=f'MAP (0-1 Loss) = {map_val}')
ax1.axvline(med_val, color='#2ca02c', linestyle='--', lw=2, label=f'Median ($L_1$ Loss) = {med_val}')
ax1.axvline(umvue_val, color='#9467bd', linestyle='-.', lw=2, label=f'UMVUE = {umvue_val:.0f}')
ax1.axvline(mean_val, color='#ff7f0e', linestyle='-', lw=2, label=f'Mean ($L_2$ Loss) = {mean_val:.1f}')
ax1.axvline(q90_val, color='#8c564b', linestyle='--', lw=1.5, label=f'Pinball ($q=0.90$) = {q90_val}')

ax1.set_title('(a) Bayesian Decision Theory: Point Estimators Across Loss Functions', fontweight='bold')
ax1.set_xlabel('Population Ceiling $N$')
ax1.set_ylabel(r'Posterior Probability Mass $P(N \mid \mathcal{D})$')
ax1.legend(loc='upper right', frameon=True, fontsize=9)
ax1.set_xlim(95, 300)

# Panel (b)
ax2 = axes[0, 1]
ax2.plot(wald_x, wald_pdf, color='#d62728', lw=2, label=f'Frequentist Wald Normal Approx (CI: [{wald_lower:.1f}, {wald_upper:.1f}])')
ax2.fill_between(wald_x[wald_x < m_ex], 0, wald_pdf[wald_x < m_ex], color='#d62728', alpha=0.3, label=f'Inadmissible Mass: $N < m$ ({mass_below*100:.1f}%)')
ax2.plot(n_b, p_b, color='#1f77b4', lw=2.5, label='Exact Bayesian Discrete Posterior')
ax2.axvline(m_ex, color='black', linestyle='--', lw=2, label=f'Physical Boundary $m = {m_ex}$')
ax2.axvline(exact_freq_upper, color='#2ca02c', linestyle=':', lw=2, label=f'Exact Freq 95% Upper = {exact_freq_upper:.1f}')

ax2.set_title('(b) Physical Boundary Violation: Wald vs. Exact Bayesian Posterior', fontweight='bold')
ax2.set_xlabel('Population Ceiling $N$')
ax2.set_ylabel('Probability Density / Mass')
ax2.legend(loc='upper right', frameon=True, fontsize=9)
ax2.set_xlim(15, 120)

# Panel (c) Moment existence table
ax3 = axes[1, 0]
ax3.axis('off')
table_data = [
    ['Sample Size', r'Flat Prior $\pi(N) \propto 1$', r'Reciprocal Prior $\pi(N) \propto 1/N$'],
    ['k = 1', 'Improper (Harmonic Diverges)', 'Proper, Mean Diverges'],
    ['k = 2', 'Proper, Mean Diverges', 'Proper, Mean Exists, Var Diverges'],
    ['k = 3', r'Mean Exists: $(k-1)(m-1)/(k-2)$', 'Mean and Var Exist'],
    ['k >= 4', 'Mean and Variance Exist', 'All Low-Order Moments Exist'],
    ['k >= 5', 'Asymptotically Gaussian', 'Asymptotically Gaussian']
]
tbl = ax3.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.24, 0.38, 0.38])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 2.2)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2b5c8f')
        cell.set_text_props(color='white', weight='bold')
    elif row == 1:
        cell.set_facecolor('#ffebee')
    elif row in (2, 3):
        cell.set_facecolor('#fff8e1')
    else:
        cell.set_facecolor('#e8f5e9')
ax3.set_title('(c) Posterior Propriety & Moment Existence Thresholds', fontweight='bold', pad=20)

# Panel (d)
ax4 = axes[1, 1]
gammas = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
biases_pct = []
N_sim = 20000
k_sim = 20
R_sim = 1000
serials_full = np.arange(1, N_sim + 1)
np.random.seed(42)

for g in gammas:
    pr = serials_full ** g
    pr /= np.sum(pr)
    samp_m = [np.max(np.random.choice(serials_full, size=k_sim, replace=False, p=pr)) for _ in range(R_sim)]
    u_est = np.mean(samp_m) * (1 + 1.0/k_sim) - 1
    biases_pct.append((u_est - N_sim) / N_sim * 100)

bars = ax4.bar(gammas, biases_pct, width=0.18, color=np.where(np.array(biases_pct) >= 0, '#d62728', '#1f77b4'), edgecolor='black', alpha=0.85)
ax4.axhline(0, color='black', lw=1.5)
ax4.set_title(f'(d) Non-Uniform Capture: Deployment Tilt ($N={N_sim}, k={k_sim}, R={R_sim}$)', fontweight='bold')
ax4.set_xlabel(r'Deployment Tilt Parameter $\gamma$ ($\gamma < 0$: Aging Fleet, $\gamma > 0$: Front-Line Concentration)')
ax4.set_ylabel('Estimation Bias of Naive Estimator (%)')
ax4.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/expert_bayesian_diagnostics.png', dpi=200)
plt.close()
print('Generated figures/expert_bayesian_diagnostics.png successfully!')

# -------------------------------------------------------------
# 4. Generate Standalone Figures for Presentation Deck
# -------------------------------------------------------------
# Standalone 1: Loss functions
fig1, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(n_grid, p, color='#1f77b4', lw=2.5, label=r'Posterior $P(N \mid m=100, k=5)$')
ax.fill_between(n_grid, 0, p, color='#1f77b4', alpha=0.15)
ax.axvline(map_val, color='#d62728', linestyle=':', lw=2, label=f'MAP (0-1 Loss) = {map_val}')
ax.axvline(med_val, color='#2ca02c', linestyle='--', lw=2, label=f'Median ($L_1$ Loss) = {med_val}')
ax.axvline(umvue_val, color='#9467bd', linestyle='-.', lw=2, label=f'UMVUE = {umvue_val:.0f}')
ax.axvline(mean_val, color='#ff7f0e', linestyle='-', lw=2, label=f'Mean ($L_2$ Loss) = {mean_val:.1f}')
ax.axvline(q90_val, color='#8c564b', linestyle='--', lw=1.5, label=f'Pinball ($q=0.90$) = {q90_val}')
ax.set_title('Bayesian Decision Theory: Point Estimators Across Loss Geometries', fontweight='bold')
ax.set_xlabel('Population Ceiling $N$')
ax.set_ylabel('Posterior Probability Mass')
ax.legend(loc='upper right', frameon=True, fontsize=9)
ax.set_xlim(95, 300)
plt.tight_layout()
plt.savefig('figures/expert_loss_functions.png', dpi=200)
plt.close()

# Standalone 2: Boundary admissibility
fig2, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(wald_x, wald_pdf, color='#d62728', lw=2, label=f'Frequentist Wald CI: [{wald_lower:.1f}, {wald_upper:.1f}]')
ax.fill_between(wald_x[wald_x < m_ex], 0, wald_pdf[wald_x < m_ex], color='#d62728', alpha=0.3, label=f'Inadmissible: $N < m$ ({mass_below*100:.1f}%)')
ax.plot(n_b, p_b, color='#1f77b4', lw=2.5, label='Exact Bayesian Discrete Posterior')
ax.axvline(m_ex, color='black', linestyle='--', lw=2, label=f'Boundary $m = {m_ex}$')
ax.axvline(exact_freq_upper, color='#2ca02c', linestyle=':', lw=2, label=f'Exact Freq 95% Upper = {exact_freq_upper:.1f}')
ax.set_title(r'Physical Boundary Admissibility: $m=50, k=5$', fontweight='bold')
ax.set_xlabel('Population Ceiling $N$')
ax.set_ylabel('Probability Density / Mass')
ax.legend(loc='upper right', frameon=True, fontsize=9)
ax.set_xlim(15, 120)
plt.tight_layout()
plt.savefig('figures/expert_boundary_wald_vs_bayes.png', dpi=200)
plt.close()

# Standalone 3: Deployment tilt
fig3, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(gammas, biases_pct, width=0.18, color=np.where(np.array(biases_pct) >= 0, '#d62728', '#1f77b4'), edgecolor='black', alpha=0.85)
ax.axhline(0, color='black', lw=1.5)
ax.set_title(f'Capture Heterogeneity: Deployment Tilt ($N={N_sim}, k={k_sim}, R={R_sim}$)', fontweight='bold')
ax.set_xlabel(r'Deployment Tilt $\gamma$ ($w(s) \propto s^\gamma$)')
ax.set_ylabel('Estimation Bias of Naive Estimator (%)')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/expert_deployment_tilt.png', dpi=200)
plt.close()
print("All standalone figures updated successfully!")
