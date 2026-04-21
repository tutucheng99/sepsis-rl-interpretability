"""
Script 6: Visualization Generation

Generates all publication-quality figures for the paper.
Includes policy heatmaps, comparison plots, SOFA stratification, etc.

Usage:
    python scripts/06_visualization.py
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# from src.visualization.policy_viz import create_policy_heatmap, plot_mortality_vs_dosage_diff  # Not needed for basic viz


def load_all_results():
    """Load all experiment results"""
    results_dir = project_root / "results"

    all_results = {}

    # Load baseline
    baseline_file = results_dir / "baseline_results.pkl"
    if baseline_file.exists():
        with open(baseline_file, 'rb') as f:
            baseline = pickle.load(f)
        all_results['baseline'] = baseline

    # Load RL results
    for algo_name in ['bc', 'cql', 'dqn']:
        result_file = results_dir / f"{algo_name}_results.pkl"
        if result_file.exists():
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            all_results[algo_name] = result

    # Load reward comparison
    reward_file = results_dir / "reward_comparison_results.pkl"
    if reward_file.exists():
        with open(reward_file, 'rb') as f:
            reward_comp = pickle.load(f)
        all_results['reward_comparison'] = reward_comp

    return all_results


def create_algorithm_comparison_figure(all_results):
    """
    Figure 1: Algorithm Comparison
    Compare all algorithms on simple reward
    """
    print("\nCreating Algorithm Comparison Figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect data
    algorithms = []
    survival_rates = []
    avg_returns = []
    std_returns = []
    sofa_low = []
    sofa_med = []
    sofa_high = []

    # Baseline
    if 'baseline' in all_results:
        for name in ['random', 'heuristic']:
            res = all_results['baseline'][name]
            algorithms.append(name.capitalize())
            survival_rates.append(res['survival_rate'] * 100)
            avg_returns.append(res['avg_return'])
            std_returns.append(res['std_return'])
            sofa_low.append(res['sofa_stratified']['low_sofa']['survival_rate'] * 100)
            sofa_med.append(res['sofa_stratified']['medium_sofa']['survival_rate'] * 100)
            sofa_high.append(res['sofa_stratified']['high_sofa']['survival_rate'] * 100)

    # RL algorithms
    for algo_name in ['bc', 'cql', 'dqn']:
        if algo_name in all_results:
            res = all_results[algo_name]['evaluation']
            algorithms.append(algo_name.upper())
            survival_rates.append(res['survival_rate'] * 100)
            avg_returns.append(res['avg_return'])
            std_returns.append(res['std_return'])
            sofa_low.append(res['sofa_stratified']['low_sofa']['survival_rate'] * 100)
            sofa_med.append(res['sofa_stratified']['medium_sofa']['survival_rate'] * 100)
            sofa_high.append(res['sofa_stratified']['high_sofa']['survival_rate'] * 100)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # Plot 1: Survival Rate
    axes[0, 0].bar(algorithms, survival_rates, color=colors[:len(algorithms)])
    axes[0, 0].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[0, 0].set_title('Overall Survival Rate', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(survival_rates):
        axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)

    # Plot 2: Average Return
    axes[0, 1].bar(algorithms, avg_returns, yerr=std_returns,
                   color=colors[:len(algorithms)], capsize=5)
    axes[0, 1].set_ylabel('Average Return', fontsize=12)
    axes[0, 1].set_title('Average Return', fontsize=14, fontweight='bold')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot 3: SOFA-Stratified Survival
    x = np.arange(3)
    width = 0.15
    sofa_categories = ['Low SOFA', 'Medium SOFA', 'High SOFA']

    for i, algo in enumerate(algorithms):
        offset = (i - len(algorithms)/2) * width
        values = [sofa_low[i], sofa_med[i], sofa_high[i]]
        axes[1, 0].bar(x + offset, values, width, label=algo, color=colors[i])

    axes[1, 0].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[1, 0].set_title('SOFA-Stratified Survival Rate', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(sofa_categories)
    axes[1, 0].legend(fontsize=9, ncol=2)
    axes[1, 0].set_ylim(0, 100)

    # Plot 4: Episode Length
    episode_lengths = [all_results['baseline']['random']['avg_episode_length'],
                       all_results['baseline']['heuristic']['avg_episode_length']]

    for algo_name in ['bc', 'cql', 'dqn']:
        if algo_name in all_results:
            episode_lengths.append(all_results[algo_name]['evaluation']['avg_episode_length'])

    axes[1, 1].bar(algorithms, episode_lengths, color=colors[:len(algorithms)])
    axes[1, 1].set_ylabel('Average Episode Length', fontsize=12)
    axes[1, 1].set_title('Average Episode Length', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "algorithm_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("[OK] Figure saved successfully")
    plt.close()


def create_reward_comparison_figure(all_results):
    """
    Figure 2: Reward Function Comparison
    Compare different reward functions
    """
    print("\nCreating Reward Function Comparison Figure...")

    if 'reward_comparison' not in all_results:
        print("[WARNING] Reward comparison results not found. Skipping.")
        return

    reward_comp = all_results['reward_comparison']
    algorithm = reward_comp['algorithm']
    results = reward_comp['results']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    reward_names = ['Simple', 'Paper', 'Hybrid']
    reward_keys = ['simple', 'paper', 'hybrid']

    # Plot 1: Survival Rate
    survival_rates = [results[r]['survival_rate'] * 100 for r in reward_keys]
    axes[0].bar(reward_names, survival_rates, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[0].set_title(f'{algorithm}: Survival Rate', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(survival_rates):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Plot 2: Average Return
    avg_returns = [results[r]['avg_return'] for r in reward_keys]
    std_returns = [results[r]['std_return'] for r in reward_keys]
    axes[1].bar(reward_names, avg_returns, yerr=std_returns,
                color=['#3498db', '#e74c3c', '#2ecc71'], capsize=5)
    axes[1].set_ylabel('Average Return', fontsize=12)
    axes[1].set_title(f'{algorithm}: Average Return', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: SOFA-Stratified
    x = np.arange(3)
    width = 0.25
    sofa_categories = ['Low\nSOFA', 'Med\nSOFA', 'High\nSOFA']

    for i, (reward_name, reward_key) in enumerate(zip(reward_names, reward_keys)):
        sofa_vals = [
            results[reward_key]['sofa_stratified']['low_sofa']['survival_rate'] * 100,
            results[reward_key]['sofa_stratified']['medium_sofa']['survival_rate'] * 100,
            results[reward_key]['sofa_stratified']['high_sofa']['survival_rate'] * 100
        ]
        axes[2].bar(x + i*width, sofa_vals, width, label=reward_name,
                    color=['#3498db', '#e74c3c', '#2ecc71'][i])

    axes[2].set_ylabel('Survival Rate (%)', fontsize=12)
    axes[2].set_title(f'{algorithm}: SOFA-Stratified', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels(sofa_categories)
    axes[2].legend()
    axes[2].set_ylim(0, 100)

    plt.tight_layout()

    # Save
    figures_dir = project_root / "results" / "figures"
    fig_path = figures_dir / "reward_function_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: reward_function_comparison.png")
    plt.close()


def create_summary_table(all_results):
    """
    Create summary table for paper
    """
    print("\nCreating Summary Table...")

    table_data = []

    # Baseline
    if 'baseline' in all_results:
        for name in ['random', 'heuristic']:
            res = all_results['baseline'][name]
            table_data.append({
                'Algorithm': name.capitalize(),
                'Survival Rate (%)': f"{res['survival_rate']*100:.1f}",
                'Avg Return': f"{res['avg_return']:.2f}",
                'Std Return': f"{res['std_return']:.2f}",
                'Avg Episode Length': f"{res['avg_episode_length']:.1f}"
            })

    # RL algorithms
    for algo_name in ['bc', 'cql', 'dqn']:
        if algo_name in all_results:
            res = all_results[algo_name]['evaluation']
            table_data.append({
                'Algorithm': algo_name.upper(),
                'Survival Rate (%)': f"{res['survival_rate']*100:.1f}",
                'Avg Return': f"{res['avg_return']:.2f}",
                'Std Return': f"{res['std_return']:.2f}",
                'Avg Episode Length': f"{res['avg_episode_length']:.1f}"
            })

    # Print table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Algorithm':<15} {'Survival (%)':<15} {'Avg Return':<15} {'Std Return':<15} {'Avg Length':<15}")
    print("-"*80)

    for row in table_data:
        print(f"{row['Algorithm']:<15} {row['Survival Rate (%)']:<15} "
              f"{row['Avg Return']:<15} {row['Std Return']:<15} {row['Avg Episode Length']:<15}")

    # Save as CSV
    results_dir = project_root / "results"
    csv_path = results_dir / "summary_table.csv"

    with open(csv_path, 'w') as f:
        # Header
        f.write("Algorithm,Survival Rate (%),Avg Return,Std Return,Avg Episode Length\n")
        # Data
        for row in table_data:
            f.write(f"{row['Algorithm']},{row['Survival Rate (%)']},{row['Avg Return']},"
                   f"{row['Std Return']},{row['Avg Episode Length']}\n")

    print(f"\n[OK] Table saved: summary_table.csv")


def main():
    """Generate all visualizations"""
    print("\n" + "="*60)
    print("VISUALIZATION GENERATION")
    print("="*60 + "\n")

    # Load all results
    print("Loading results...")
    all_results = load_all_results()

    if not all_results:
        print("[ERROR] No results found! Please run experiments first.")
        return 1

    print(f"[OK] Loaded results for: {list(all_results.keys())}")

    # Create figures
    create_algorithm_comparison_figure(all_results)
    create_reward_comparison_figure(all_results)
    create_summary_table(all_results)

    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated figures:")
    print(f"  results/figures/algorithm_comparison.png")
    print(f"  results/figures/reward_function_comparison.png")
    print(f"  results/summary_table.csv")
    print(f"\nNext step:")
    print(f"  python scripts/07_final_analysis.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
