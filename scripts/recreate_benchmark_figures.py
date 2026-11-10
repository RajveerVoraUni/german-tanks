import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

os.makedirs('figures', exist_ok=True)
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 200,
    'mathtext.fontset': 'stixsans'
})

df = pd.read_csv('results/wp1_dense_summary.csv')

methods_to_plot = [
    ('Frequentist UMVUE', 'UMVUE', 'Frequentist UMVUE', '#1f77b4', 'o', '--'),
    ('Bayes (Flat)', 'Posterior Mean (SE loss)', 'Bayes Flat (Mean)', '#ff7f0e', 's', '-'),
    ('Bayes (Flat)', 'Posterior Median (Abs loss)', 'Bayes Flat (Median)', '#e377c2', 'd', ':'),
    ('Bayes (1/n (Scale-Inv))', 'Posterior Mean (SE loss)', 'Bayes 1/N (Scale-Inv)', '#2ca02c', '^', '-'),
    ('Bayes (Poisson (mu=0.5*N_true))', 'Posterior Mean (SE loss)', 'Misspecified Poisson (mu=0.5N)', '#d62728', 'x', '-.')
]

# -------------------------------------------------------------
# 1. Figure: Benchmark Bias (Fix x-axis '$k$', clean title)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
for m, est, label, color, marker, ls in methods_to_plot:
    sub = df[(df['method'] == m) & (df['estimator'] == est)]
    if not sub.empty:
        ax.plot(sub['k'], sub['bias'], label=label, color=color, marker=marker, linestyle=ls, lw=2.2, markersize=8)

ax.axhline(0, color='black', linestyle='--', alpha=0.7, lw=1.5)
ax.set_title(r'Estimator Bias vs. Sample Size $k$ ($N_{\mathrm{true}} = 22,061$)', fontweight='bold', fontsize=13)
ax.set_xlabel(r'Sample Size $k$', fontsize=12)
ax.set_ylabel(r'Estimation Bias ($\hat{N} - N_{\mathrm{true}}$)', fontsize=12)
ax.set_xlim(0, 105)
ax.set_xticks([5, 10, 20, 50, 100])
ax.legend(loc='best', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('figures/benchmark_bias.png', dpi=200)
plt.close()
print("Saved figures/benchmark_bias.png")

# -------------------------------------------------------------
# 2. Figure: Benchmark RMSE (Fix x-axis '$k$', fix arbitrary y-axis scale)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
for m, est, label, color, marker, ls in methods_to_plot:
    sub = df[(df['method'] == m) & (df['estimator'] == est)]
    if not sub.empty:
        ax.plot(sub['k'], sub['rmse'], label=label, color=color, marker=marker, linestyle=ls, lw=2.2, markersize=8)

ax.set_title(r'Estimator RMSE vs. Sample Size $k$ (Log Scale)', fontweight='bold', fontsize=13)
ax.set_xlabel(r'Sample Size $k$', fontsize=12)
ax.set_ylabel(r'Root Mean Squared Error (RMSE)', fontsize=12)
ax.set_yscale('log')

# Eliminate arbitrary scientific ticks; set clean integer ticks
ax.yaxis.set_minor_locator(ticker.NullLocator())
y_ticks = [200, 500, 1000, 2000, 5000]
ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
ax.set_ylim(180, 6000)

ax.set_xlim(0, 105)
ax.set_xticks([5, 10, 20, 50, 100])

ax.legend(loc='upper right', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('figures/benchmark_rmse.png', dpi=200)
plt.close()
print("Saved figures/benchmark_rmse.png")

# -------------------------------------------------------------
# 3. Figure: Benchmark 95% Coverage (Fix x-axis, add kink / broken axis)
# -------------------------------------------------------------
coverage_methods = [
    ('Frequentist UMVUE', 'UMVUE', 'Frequentist 95% CI (Wald)', '#1f77b4', 'o', '--'),
    ('Bayes (Flat)', 'Posterior Mean (SE loss)', 'Bayes Flat 95% HPD', '#ff7f0e', 's', '-'),
    ('Bayes (Flat)', 'Posterior Median (Abs loss)', 'Bayes Flat 95% Equal-Tailed', '#e377c2', 'd', ':'),
    ('Bayes (1/n (Scale-Inv))', 'Posterior Mean (SE loss)', 'Bayes 1/N 95% HPD', '#2ca02c', '^', '-'),
    ('Bayes (Poisson (mu=0.5*N_true))', 'Posterior Mean (SE loss)', 'Misspecified Poisson 95% HPD', '#d62728', 'x', '-.')
]

# Broken y-axis: top panel for [80, 100], bottom panel for [0, 4]
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 5.0),
    gridspec_kw={'height_ratios': [3.2, 1.0], 'hspace': 0.12}
)

# Plot data on both axes
for m, est, label, color, marker, ls in coverage_methods:
    sub = df[(df['method'] == m) & (df['estimator'] == est)]
    if not sub.empty:
        ax_top.plot(sub['k'], sub['coverage_95'] * 100, label=label, color=color, marker=marker, linestyle=ls, lw=2.2, markersize=8)
        ax_bottom.plot(sub['k'], sub['coverage_95'] * 100, label=label, color=color, marker=marker, linestyle=ls, lw=2.2, markersize=8)

# Nominal 95% line on top axis
ax_top.axhline(95, color='black', linestyle='--', alpha=0.85, lw=1.5, label='Nominal 95% Level')

# Zoom ranges: top zooms 80-100%, bottom shows 0-3%
ax_top.set_ylim(79, 101)
ax_bottom.set_ylim(-0.5, 3.5)

ax_top.set_yticks([80, 85, 90, 95, 100])
ax_bottom.set_yticks([0, 2])

# Hide spines between top and bottom
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.xaxis.tick_top()
ax_top.tick_params(labeltop=False, top=False, bottom=False)
ax_bottom.xaxis.tick_bottom()

ax_bottom.set_xlim(0, 105)
ax_bottom.set_xticks([5, 10, 20, 50, 100])
ax_bottom.set_xlabel(r'Sample Size $k$', fontsize=12)

# Common y-label centered between subplots
fig.text(0.02, 0.52, 'Empirical Coverage Percentage (%)', va='center', rotation='vertical', fontsize=12)

# Standard kink break slashes on y-axes
d = 0.015  # size of diagonal tick lines
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=1.3)
ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right

kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d*3.2, 1 + d*3.2), **kwargs)        # bottom-left
ax_bottom.plot((1 - d, 1 + d), (1 - d*3.2, 1 + d*3.2), **kwargs)  # bottom-right

ax_top.set_title(r'95% Credible / Confidence Interval Coverage vs. Sample Size $k$', fontweight='bold', fontsize=13)
ax_top.legend(loc='lower left', frameon=True, fontsize=9.5)

# Clear annotations
ax_top.text(0.98, 0.15, 'Zoom: 80%–100% Coverage', transform=ax_top.transAxes,
            ha='right', va='bottom', fontsize=10, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9))

ax_bottom.annotate(
    'Misspecified Poisson Collapse (< 2.5%)',
    xy=(50, 1.15), xytext=(22, 2.2),
    fontsize=9.5, color='#d62728', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2)
)

fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.11)
plt.savefig('figures/benchmark_coverage.png', dpi=200)
plt.close()
print("Saved figures/benchmark_coverage.png")
