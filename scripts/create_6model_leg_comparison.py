"""
Create comprehensive 6-model LEG interpretability comparison
Combines offline (BC, CQL, DQN) and online (DDQN-Att, DDQN-Res, SAC) results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import pandas as pd

# Set style for journal publication
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# ============================================================
# DATA: Offline models (from paper Results section)
# ============================================================
offline_models = {
    'CQL': {
        'max_saliency': 40.06,
        'top_feature': 'SysBP',
        'saliency_range': '±4 to ±40',
        'interpretability': 'Excellent',
        'survival_overall': 94.0,
        'survival_high_sofa': 88.5,
        'paradigm': 'Offline'
    },
    'BC': {
        'max_saliency': 0.78,
        'top_feature': 'qSOFA',
        'saliency_range': '±0.05 to ±0.78',
        'interpretability': 'Mixed',
        'survival_overall': 94.2,
        'survival_high_sofa': 88.6,
        'paradigm': 'Offline'
    },
    'DQN': {
        'max_saliency': 0.069,
        'top_feature': 'INR',
        'saliency_range': '±0.02 to ±0.07',
        'interpretability': 'Poor',
        'survival_overall': 94.0,
        'survival_high_sofa': 84.3,
        'paradigm': 'Offline (online trained)'
    }
}

# ============================================================
# DATA: Online models (from github_models/sepsis_leg_analysis JSON)
# ============================================================
online_json_path = Path(r"C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\github_models\sepsis_leg_analysis\multi_model_feature_importance.json")

with open(online_json_path, 'r') as f:
    online_data = json.load(f)

online_models = {
    'DDQN-Attention': {
        'max_saliency': 3.57,  # qSOFA from JSON
        'top_feature': 'qSOFA',
        'saliency_range': '±1.0 to ±3.6',
        'interpretability': 'Moderate',
        'survival_overall': 95.4,
        'survival_high_sofa': 90.5,
        'paradigm': 'Online'
    },
    'DDQN-Residual': {
        'max_saliency': 2.93,  # INR from JSON
        'top_feature': 'INR',
        'saliency_range': '±1.0 to ±2.9',
        'interpretability': 'Moderate',
        'survival_overall': 94.2,
        'survival_high_sofa': 87.0,
        'paradigm': 'Online'
    },
    'SAC': {
        'max_saliency': 1.17,  # INR from JSON
        'top_feature': 'INR',
        'saliency_range': '±0.5 to ±1.2',
        'interpretability': 'Limited',
        'survival_overall': 94.8,
        'survival_high_sofa': 88.7,
        'paradigm': 'Online'
    }
}

# Combine all models
all_models = {**offline_models, **online_models}

# ============================================================
# FIGURE 1: Comprehensive 6-Model Comparison
# ============================================================
fig = plt.figure(figsize=(16, 12))  # Increased height for better spacing
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)  # Increased spacing

# Color scheme
colors_paradigm = {'Offline': '#2ecc71', 'Online': '#3498db', 'Offline (online trained)': '#9b59b6'}
model_order = ['CQL', 'DDQN-Attention', 'DDQN-Residual', 'SAC', 'BC', 'DQN']

# Plot 1: Maximum LEG Saliency (Log Scale)
ax1 = fig.add_subplot(gs[0, :])
saliencies = [all_models[m]['max_saliency'] for m in model_order]
colors = [colors_paradigm[all_models[m]['paradigm']] for m in model_order]

bars = ax1.bar(range(len(model_order)), saliencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_yscale('log')
ax1.set_xlabel('Algorithm', fontweight='bold', fontsize=12)
ax1.set_ylabel('Max LEG Saliency (log)', fontweight='bold', fontsize=12)  # Shortened label
ax1.set_title('(A) LEG Interpretability Comparison: 6 RL Algorithms',
              fontweight='bold', fontsize=14, pad=15)
ax1.set_xticks(range(len(model_order)))
ax1.set_xticklabels(model_order, fontsize=11, rotation=0)
ax1.set_ylim([0.01, 100])  # Set y-axis limits to give more room
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, saliencies)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.3,  # Move labels slightly higher
             f'{val:.2f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add interpretability annotations - position adjusted for each bar
interpretability_levels = [all_models[m]['interpretability'] for m in model_order]
# Custom y positions: move "Poor" below the bar for DQN
y_positions = []
for i, (s, level) in enumerate(zip(saliencies, interpretability_levels)):
    if level == 'Poor':  # DQN case - put label below at 0.02
        y_positions.append(0.025)
    else:
        y_positions.append(s * 0.4)

for i, (level, y) in enumerate(zip(interpretability_levels, y_positions)):
    ax1.text(i, y, level, ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

# Add reference lines
ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold: 1.0')
ax1.axhline(y=10.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Threshold: 10.0')

# Legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors_paradigm['Offline'], label='Offline RL'),
                  plt.Rectangle((0,0),1,1, facecolor=colors_paradigm['Online'], label='Online RL'),
                  plt.Rectangle((0,0),1,1, facecolor=colors_paradigm['Offline (online trained)'], label='DQN (online trained)')]
ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

# Plot 2: Survival Rate Comparison
ax2 = fig.add_subplot(gs[1, 0])
survival_overall = [all_models[m]['survival_overall'] for m in model_order]
bars2 = ax2.bar(range(len(model_order)), survival_overall, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Algorithm', fontweight='bold', fontsize=11)
ax2.set_ylabel('Overall Survival Rate (%)', fontweight='bold', fontsize=11)
ax2.set_title('(B) Overall Performance (500 Episodes)', fontweight='bold', fontsize=12)
ax2.set_xticks(range(len(model_order)))
# Use abbreviated labels to avoid overflow
model_labels_short = ['CQL', 'DDQN-Att', 'DDQN-Res', 'SAC', 'BC', 'DQN']
ax2.set_xticklabels(model_labels_short, fontsize=9, rotation=30, ha='right')
ax2.set_ylim([93, 96])
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars2, survival_overall):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%',
             ha='center', va='bottom', fontsize=9)

# Plot 3: High-SOFA Survival Rate
ax3 = fig.add_subplot(gs[1, 1])
survival_high_sofa = [all_models[m]['survival_high_sofa'] for m in model_order]
bars3 = ax3.bar(range(len(model_order)), survival_high_sofa, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Algorithm', fontweight='bold', fontsize=11)
ax3.set_ylabel('Survival Rate (%)', fontweight='bold', fontsize=11)
ax3.set_title('(C) High-Severity Patients (SOFA >= 11)', fontweight='bold', fontsize=12)
ax3.set_xticks(range(len(model_order)))
# Use abbreviated labels to avoid overflow
ax3.set_xticklabels(model_labels_short, fontsize=9, rotation=30, ha='right')
ax3.set_ylim([82, 92])
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars3, survival_high_sofa):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%',
             ha='center', va='bottom', fontsize=9)

# Plot 4: Performance vs Interpretability Scatter
ax4 = fig.add_subplot(gs[2, :])
for model_name in model_order:
    model_data = all_models[model_name]
    ax4.scatter(model_data['max_saliency'], model_data['survival_overall'],
               s=300, alpha=0.7, color=colors_paradigm[model_data['paradigm']],
               edgecolors='black', linewidth=2, label=model_name)
    # Data point labels - restore to original position (right of points)
    ax4.annotate(model_name,
                (model_data['max_saliency'], model_data['survival_overall']),
                xytext=(10, 5), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8))

ax4.set_xscale('log')
ax4.set_xlabel('Maximum LEG Saliency (Interpretability, log scale)', fontweight='bold', fontsize=12)
ax4.set_ylabel('Overall Survival Rate (%)', fontweight='bold', fontsize=12)
ax4.set_title('(D) Performance-Interpretability Trade-off', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_ylim([93.5, 96])

# Add legend for paradigm colors in scatter plot - moved to upper left
legend_elements_scatter = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_paradigm['Offline'],
                                       markersize=12, label='Offline RL', markeredgecolor='black'),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_paradigm['Online'],
                                       markersize=12, label='Online RL', markeredgecolor='black'),
                          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_paradigm['Offline (online trained)'],
                                       markersize=12, label='DQN (Hybrid)', markeredgecolor='black')]
ax4.legend(handles=legend_elements_scatter, loc='upper left', framealpha=0.9, fontsize=10)

# Add annotation about key finding - moved to upper right corner
textstr = ('Key Finding: CQL achieves 94.0% survival with\n'
           '40.06 saliency (11x better than best online).\n'
           'DDQN-Att achieves 95.4% but only 3.57 saliency.')
ax4.text(0.98, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Adjust layout to prevent label cutoff
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.4, wspace=0.3)

# Save in multiple formats for journal submission
output_dir = Path(r"C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\results\figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Save PNG at 300 dpi
output_path_png = output_dir / "leg_6model_comparison.png"
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved PNG (300 dpi): leg_6model_comparison.png")

# Save PDF for vector graphics
output_path_pdf = output_dir / "leg_6model_comparison.pdf"
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved PDF (vector): leg_6model_comparison.pdf")

# Save TIFF at 300 dpi (some journals require TIFF)
output_path_tiff = output_dir / "leg_6model_comparison.tiff"
plt.savefig(output_path_tiff, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved TIFF (300 dpi): leg_6model_comparison.tiff")

plt.close()

# ============================================================
# Create Summary Table (CSV and LaTeX)
# ============================================================
summary_data = []
for model_name in model_order:
    model_data = all_models[model_name]
    summary_data.append({
        'Algorithm': model_name,
        'Paradigm': model_data['paradigm'],
        'Max Saliency': model_data['max_saliency'],
        'Top Feature': model_data['top_feature'],
        'Saliency Range': model_data['saliency_range'],
        'Interpretability': model_data['interpretability'],
        'Survival (%)': model_data['survival_overall'],
        'High-SOFA Survival (%)': model_data['survival_high_sofa']
    })

df = pd.DataFrame(summary_data)

# Save CSV
csv_path = Path(r"C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\results\tables\leg_6model_summary.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_path, index=False)
print("\nSaved summary table: leg_6model_summary.csv")

# Print LaTeX table
latex_table = df.to_latex(index=False, float_format="%.2f",
                          caption="Comprehensive 6-Model LEG Interpretability Comparison",
                          label="tab:6model-leg")
print("\n" + "="*60)
print("LaTeX Table:")
print("="*60)
print(latex_table)

# Print summary statistics
print("\n" + "="*60)
print("Summary Statistics:")
print("="*60)
print(f"\nInterpretability Ranking (by Max Saliency):")
for i, row in df.sort_values('Max Saliency', ascending=False).iterrows():
    ratio_to_dqn = row['Max Saliency'] / 0.069
    print(f"  {row['Algorithm']:15s}: {row['Max Saliency']:6.2f}  ({ratio_to_dqn:4.0f}× vs DQN)")

print(f"\nPerformance Ranking (by Overall Survival):")
for i, row in df.sort_values('Survival (%)', ascending=False).iterrows():
    print(f"  {row['Algorithm']:15s}: {row['Survival (%)']:5.1f}%")

print(f"\nHigh-SOFA Performance Ranking:")
for i, row in df.sort_values('High-SOFA Survival (%)', ascending=False).iterrows():
    print(f"  {row['Algorithm']:15s}: {row['High-SOFA Survival (%)']:5.1f}%")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
