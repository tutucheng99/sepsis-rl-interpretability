"""
Policy Visualization Tools

Creates publication-quality figures for the paper:
1. Policy heatmaps (like Raghu et al. Fig 1)
2. Mortality vs dosage deviation (like Raghu et al. Fig 2)
3. Training curves
4. SOFA-stratified comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def create_policy_heatmap(policy_fn: callable,
                          env,
                          n_episodes: int = 500,
                          sofa_category: str = 'all',
                          title: str = 'Policy',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create policy heatmap showing action distribution (IV vs VP)

    Similar to Raghu et al. (2017) Figure 1

    Args:
        policy_fn: Policy function
        env: Sepsis environment
        n_episodes: Number of episodes to sample
        sofa_category: 'all', 'low', 'medium', 'high'
        title: Plot title
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    SOFA_IDX = 37

    # Collect actions
    action_counts = np.zeros((5, 5))  # 5x5 action grid

    for _ in range(n_episodes):
        obs, info = env.reset()
        sofa = obs[SOFA_IDX]

        # Filter by SOFA category
        if sofa_category == 'low' and sofa >= 5:
            continue
        elif sofa_category == 'medium' and (sofa < 5 or sofa > 15):
            continue
        elif sofa_category == 'high' and sofa <= 15:
            continue

        done = False
        steps = 0
        while not done and steps < 50:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            # Decode action to (IV_bin, VP_bin)
            # action = 5 * IV_bin + VP_bin
            iv_bin = action // 5
            vp_bin = action % 5
            action_counts[iv_bin, vp_bin] += 1

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        action_counts,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Action Count'},
        ax=ax,
        square=True
    )

    ax.set_xlabel('Vasopressor Dose', fontsize=12)
    ax.set_ylabel('IV Fluid Dose', fontsize=12)
    ax.set_title(f'{title} - {sofa_category.capitalize()} SOFA', fontsize=14)
    ax.set_xticklabels(['0', '1', '2', '3', '4'])
    ax.set_yticklabels(['0', '1', '2', '3', '4'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap: {save_path}")

    return fig


def create_mortality_dosage_plot(policy_fn: callable,
                                  clinician_policy_fn: callable,
                                  env,
                                  n_episodes: int = 500,
                                  sofa_category: str = 'medium',
                                  title: str = 'Policy',
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot mortality vs dosage deviation from clinician

    Similar to Raghu et al. (2017) Figure 2

    Args:
        policy_fn: Learned policy
        clinician_policy_fn: Clinician policy (heuristic)
        env: Sepsis environment
        n_episodes: Number of episodes
        sofa_category: 'low', 'medium', 'high'
        title: Plot title
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    SOFA_IDX = 37

    # Collect data
    dosage_diffs = []
    mortalities = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        sofa = obs[SOFA_IDX]

        # Filter by SOFA category
        if sofa_category == 'low' and sofa >= 5:
            continue
        elif sofa_category == 'medium' and (sofa < 5 or sofa > 15):
            continue
        elif sofa_category == 'high' and sofa <= 15:
            continue

        episode_diffs = []
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 50:
            # Get both policies' actions
            learned_action = policy_fn(obs)
            clinician_action = clinician_policy_fn(obs)

            # Calculate dosage difference
            learned_iv, learned_vp = learned_action // 5, learned_action % 5
            clinician_iv, clinician_vp = clinician_action // 5, clinician_action % 5

            diff = abs(learned_iv - clinician_iv) + abs(learned_vp - clinician_vp)
            episode_diffs.append(diff)

            # Step environment with learned policy
            obs, reward, terminated, truncated, info = env.step(learned_action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        # Average difference for episode
        avg_diff = np.mean(episode_diffs) if episode_diffs else 0
        died = total_reward < 0

        dosage_diffs.append(avg_diff)
        mortalities.append(1 if died else 0)

    # Bin differences and compute mortality rate
    diff_bins = np.arange(0, max(dosage_diffs) + 1, 0.5)
    mortality_by_bin = []
    bin_centers = []

    for i in range(len(diff_bins) - 1):
        mask = (np.array(dosage_diffs) >= diff_bins[i]) & (np.array(dosage_diffs) < diff_bins[i+1])
        if np.sum(mask) > 0:
            mortality_rate = np.mean(np.array(mortalities)[mask])
            mortality_by_bin.append(mortality_rate)
            bin_centers.append((diff_bins[i] + diff_bins[i+1]) / 2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(bin_centers, mortality_by_bin, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax.fill_between(bin_centers, 0, mortality_by_bin, alpha=0.3, color='#e74c3c')

    ax.set_xlabel('Dosage Deviation from Clinician', fontsize=12)
    ax.set_ylabel('Mortality Rate', fontsize=12)
    ax.set_title(f'{title} - {sofa_category.capitalize()} SOFA Patients', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Annotate minimum
    min_idx = np.argmin(mortality_by_bin)
    ax.annotate(f'Min: {mortality_by_bin[min_idx]:.2f}',
                xy=(bin_centers[min_idx], mortality_by_bin[min_idx]),
                xytext=(bin_centers[min_idx] + 0.5, mortality_by_bin[min_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved mortality plot: {save_path}")

    return fig


def create_training_curves(log_dict: Dict,
                           algorithms: List[str],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training curves for multiple algorithms

    Args:
        log_dict: Dict mapping algorithm names to log dictionaries
        algorithms: List of algorithm names to plot
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

    for idx, algo in enumerate(algorithms):
        if algo not in log_dict:
            continue

        logs = log_dict[algo]

        # Plot loss
        if 'loss' in logs:
            axes[0].plot(logs['loss'], label=algo, color=colors[idx], linewidth=2)

        # Plot reward
        if 'reward' in logs:
            axes[1].plot(logs['reward'], label=algo, color=colors[idx], linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Training Reward')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves: {save_path}")

    return fig


def create_sofa_stratified_comparison(results_dict: Dict[str, Dict],
                                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create SOFA-stratified comparison plot

    Args:
        results_dict: Dict mapping policy names to evaluation results
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    categories = ['low_sofa', 'medium_sofa', 'high_sofa']
    category_names = ['Low SOFA\n(<5)', 'Medium SOFA\n(5-15)', 'High SOFA\n(>15)']

    policies = list(results_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(policies)))

    for cat_idx, (cat, cat_name) in enumerate(zip(categories, category_names)):
        survival_rates = []

        for policy in policies:
            sr = results_dict[policy]['sofa_stratified'][cat]['survival_rate'] * 100
            survival_rates.append(sr)

        x = np.arange(len(policies))
        axes[cat_idx].bar(x, survival_rates, color=colors, edgecolor='black', linewidth=1.5)
        axes[cat_idx].set_xticks(x)
        axes[cat_idx].set_xticklabels(policies, rotation=45, ha='right')
        axes[cat_idx].set_ylabel('Survival Rate (%)')
        axes[cat_idx].set_title(cat_name)
        axes[cat_idx].set_ylim(0, 100)
        axes[cat_idx].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, v in enumerate(survival_rates):
            axes[cat_idx].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SOFA comparison: {save_path}")

    return fig


# For testing
if __name__ == "__main__":
    print("Policy visualization module loaded successfully!")
    print("\nAvailable functions:")
    print("  - create_policy_heatmap(policy_fn, env, ...)")
    print("  - create_mortality_dosage_plot(policy_fn, clinician_fn, env, ...)")
    print("  - create_training_curves(log_dict, algorithms, ...)")
    print("  - create_sofa_stratified_comparison(results_dict, ...)")
