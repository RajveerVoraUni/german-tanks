
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
import scipy.special as sc
import seaborn as sns

os.makedirs('figures', exist_ok=True)
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'figure.dpi': 200})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): Posterior under different loss functions
ax1 = axes[0, 0]
k = 5
m = 100
n_grid = np.arange(m, 350)
p = (k - 1) / k * sc.comb(m - 1, k - 1) / sc.comb(n_grid, k)
p /= np.sum(p)

mean_val = (k - 1) * (m - 1) / (k - 2)
med_val = int(m * (2.0 ** (1.0 / (k - 1))))
map_val = m
umvue_val = m * (1 + 1.0 / k) - 1
q90_val = int(m * (10.0 ** (1.0 / (k - 1))))

ax1.plot(n_grid, p, color='#1f77b4', lw=2.5, label=r'Posterior $P(N \mid m=100, k=5)$')
ax1.fill_between(n_grid, 0, p, color='#1f77b4', alpha=0.15)
ax1.axvline(map_val, color='#d62728', linestyle=':', lw=2, label=f'MAP (0-1 Loss) = {map_val}')
ax1.axvline(med_val, color='#2ca02c', linestyle='--', lw=2, label=f'Median (L1 Loss) = {med_val}')
ax1.axvline(umvue_val, color='#9467bd', linestyle='-.', lw=2, label=f'UMVUE = {umvue_val:.0f}')
ax1.axvline(mean_val, color='#ff7f0e', linestyle='-', lw=2, label=f'Mean (L2 Loss) = {mean_val:.1f}')
ax1.axvline(q90_val, color='#8c564b', linestyle='--', lw=1.5, label=f'Risk-Averse (q=0.90) = {q90_val}')

ax1.set_title('(a) Bayesian Decision Theory: Point Estimators Across Loss Functions', fontweight='bold')
ax1.set_xlabel('Population Ceiling $N$')
ax1.set_ylabel(r'Posterior Probability Mass $P(N \mid \mathcal{D})$')
ax1.legend(loc='upper right', frameon=True, fontsize=9)
ax1.set_xlim(95, 300)

# Panel (b): Boundary violation in Wald vs Exact Bayesian HPD
ax2 = axes[0, 1]
m_ex = 50
k_ex = 5
wald_mean = m_ex * (1 + 1.0/k_ex) - 1
wald_se = np.sqrt((wald_mean - k_ex) * (wald_mean + 1.0) / (k_ex * (k_ex + 2.0)))
wald_x = np.linspace(10, 120, 500)
wald_pdf = 1.0 / (np.sqrt(2*np.pi)*wald_se) * np.exp(-0.5 * ((wald_x - wald_mean)/wald_se)**2)

n_b = np.arange(m_ex, 150)
p_b = (k_ex - 1) / k_ex * sc.comb(m_ex - 1, k_ex - 1) / sc.comb(n_b, k_ex)
p_b /= np.sum(p_b)

ax2.plot(wald_x, wald_pdf, color='#d62728', lw=2, label='Frequentist Wald Gaussian Approx')
ax2.fill_between(wald_x[wald_x < m_ex], 0, wald_pdf[wald_x < m_ex], color='#d62728', alpha=0.3, label=r'Physical Impossibility: $N < m$ (30.8% mass)')
ax2.plot(n_b, p_b, color='#1f77b4', lw=2.5, label='Exact Bayesian Discrete Posterior')
ax2.axvline(m_ex, color='black', linestyle='--', lw=2, label=f'Physical Boundary $m = {m_ex}$')
ax2.set_title('(b) Physical Boundary Violation: Wald vs. Exact Bayesian Posterior', fontweight='bold')
ax2.set_xlabel('Population Ceiling $N$')
ax2.set_ylabel('Probability Density / Mass')
ax2.legend(loc='upper right', frameon=True, fontsize=9)
ax2.set_xlim(15, 120)

# Panel (c): Moment existence table
ax3 = axes[1, 0]
ax3.axis('off')
table_data = [
    ['Sample Size', r'Flat Prior $\pi(N) \propto 1$', r'Reciprocal Prior $\pi(N) \propto 1/N$'],
    ['k = 1', 'Improper (Sum Diverges)', 'Proper, Mean Diverges'],
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

# Panel (d): Non-Uniform Front-Line Deployment Bias
ax4 = axes[1, 1]
gammas = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
biases_pct = []
N_sim = 20000
k_sim = 20
serials_full = np.arange(1, N_sim + 1)
np.random.seed(42)

for g in gammas:
    pr = serials_full ** g
    pr /= np.sum(pr)
    samp_m = [np.max(np.random.choice(serials_full, size=k_sim, replace=False, p=pr)) for _ in range(500)]
    u_est = np.mean(samp_m) * (1 + 1.0/k_sim) - 1
    biases_pct.append((u_est - N_sim) / N_sim * 100)

bars = ax4.bar(gammas, biases_pct, width=0.18, color=np.where(np.array(biases_pct) >= 0, '#d62728', '#1f77b4'), edgecolor='black', alpha=0.85)
ax4.axhline(0, color='black', lw=1.5)
ax4.set_title(r'(d) Non-Uniform Capture: Front-Line Deployment Tilt ($w(s) \propto s^\gamma$)', fontweight='bold')
ax4.set_xlabel(r'Deployment Tilt Parameter $\gamma$ ($\gamma < 0$: Aging Fleet, $\gamma > 0$: Front-Line Concentration)')
ax4.set_ylabel('Estimation Bias of Naive Estimator (%)')
ax4.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/expert_bayesian_diagnostics.png', dpi=200)
plt.close()
print('Generated figures/expert_bayesian_diagnostics.png successfully!')
