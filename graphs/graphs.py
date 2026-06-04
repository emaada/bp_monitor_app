import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set global plotting style for a clean, professional dark look
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'grid.color': '#2f2f35',
    'axes.edgecolor': '#4b5563',
    'axes.linewidth': 1.2
})

# =========================================================================
# GRAPH 1: SBP vs. DBP Error Correlation (Scatter Plot + Trendline)
# =========================================================================
sbp_errors = [1.65, 7.98, 10.61, 0.20, 1.06, 0.35, 1.41, 11.45, 3.33, 6.23, 0.43, 0.29]
dbp_errors = [0.36, 1.45, 1.93, 3.30, 3.53, 2.25, 1.57, 7.12, 0.78, 4.29, 0.67, 3.20]

plt.figure(figsize=(9, 5.5))
# Typo fixed here: alpha=0.8, edgecolor...
plt.scatter(sbp_errors, dbp_errors, color='#60a5fa', s=70, alpha=0.8, edgecolor='white', linewidth=0.5, label='Participant Evaluation')

m, b = np.polyfit(sbp_errors, dbp_errors, 1)
x_range = np.linspace(0, 14, 100)
plt.plot(x_range, m*x_range + b, color='#ef4444', linestyle='-', linewidth=2, label='Trendline (r = 0.430)')

plt.annotate('(11.45, 7.12)', xy=(11.45, 7.12), xytext=(10.2, 7.5),
             arrowprops=dict(arrowstyle="->", color='#9ca3af', lw=1),
             color='#9ca3af', fontsize=10, weight='bold')

plt.title('3. SBP vs. DBP Error Correlation', fontsize=16, pad=15, weight='bold', loc='left')
plt.xlabel('SBP Errors (mmHg)', fontsize=12, labelpad=10)
plt.ylabel('DBP Errors (mmHg)', fontsize=12, labelpad=10)
plt.xlim(0, 15)
plt.ylim(0, 10)
plt.grid(True, linestyle='--')
plt.legend(frameon=True, facecolor='#1e1e24', edgecolor='none')
plt.tight_layout()
plt.savefig('sbp_dbp_correlation.png', dpi=300)
plt.show()


# =========================================================================
# GRAPH 2: Mode-Based SBP Performance Breakdown (Bar Chart)
# =========================================================================
modes = ['Foot', 'Peak']
sbp_means = [6.340, 2.885]

plt.figure(figsize=(7, 5))
bars1 = plt.bar(modes, sbp_means, color='#457b9d', width=0.5, edgecolor='none', alpha=0.9)

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f'{yval:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')

plt.annotate('Lower Error\n(Superior)', xy=(1, 3.2), xytext=(0.4, 4.8),
             arrowprops=dict(facecolor='#22c55e', shrink=0.05, edgecolor='none'),
             ha='center', va='center', color='#22c55e', weight='bold')

plt.title('Mode-Based SBP Performance Breakdown (Mean MAE)', fontsize=14, pad=15, weight='bold', loc='left')
plt.ylabel('Mean MAE (mmHg)', fontsize=12, labelpad=10)
plt.ylim(0, 8)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('mode_sbp_performance.png', dpi=300)
plt.show()


# =========================================================================
# GRAPH 3: Mode-Based DBP Performance Breakdown (Bar Chart)
# =========================================================================
dbp_means = [1.350, 2.933]

plt.figure(figsize=(7, 5))
bars2 = plt.bar(modes, dbp_means, color='#457b9d', width=0.5, edgecolor='none', alpha=0.9)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')

plt.annotate('Lower Error\n(Superior)', xy=(0, 1.6), xytext=(0.6, 2.4),
             arrowprops=dict(facecolor='#22c55e', shrink=0.05, edgecolor='none'),
             ha='center', va='center', color='#22c55e', weight='bold')

plt.title('Mode-Based DBP Performance Breakdown (Mean MAE)', fontsize=14, pad=15, weight='bold', loc='left')
plt.ylabel('Mean MAE (mmHg)', fontsize=12, labelpad=10)
plt.ylim(0, 4)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('mode_dbp_performance.png', dpi=300)
plt.show()