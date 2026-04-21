"""
Script 1: Baseline Evaluation

Evaluates random and heuristic policies on the sepsis environment.
Saves results for comparison with RL algorithms.

Usage:
    python scripts/01_baseline_evaluation.py
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.sepsis_wrapper import make_sepsis_env
from src.evaluation.metrics import evaluate_policy, print_evaluation_results, compare_policies


def random_policy(state):
    """Random baseline: select action uniformly at random"""
    return np.random.randint(0, 24)


def heuristic_policy(state):
    """
    Clinical rule-based heuristic policy

    Decision rules based on SOFA, lactate, blood pressure
    """
    # Feature indices
    LACTATE_IDX = 15
    MEAN_BP_IDX = 16
    SBP_IDX = 25
    SOFA_IDX = 37

    lactate = state[LACTATE_IDX]
    sbp = state[SBP_IDX]
    map_bp = state[MEAN_BP_IDX]
    sofa = state[SOFA_IDX]

    # Clinical decision rules (states are standardized)
    if sbp < -1.0 or map_bp < -1.0:  # Severe hypotension
        iv_bin, vp_bin = 4, 3
    elif lactate > 1.0:  # High lactate
        iv_bin, vp_bin = 3, 2
    elif sofa > 1.0:  # High SOFA
        iv_bin, vp_bin = 3, 3
    elif sbp < 0 or lactate > 0:  # Mild abnormalities
        iv_bin, vp_bin = 2, 1
    else:  # Stable
        iv_bin, vp_bin = 1, 1

    action = min(5 * iv_bin + vp_bin, 23)
    return action


def main():
    """Run baseline evaluation"""
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60 + "\n")

    # Create environment
    print("Creating environment...")
    env = make_sepsis_env(reward_fn_name='simple')
    print(f"[OK] Environment created")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")

    # Evaluate Random Policy
    print("\n" + "-"*60)
    print("Evaluating Random Policy...")
    print("-"*60)
    random_results = evaluate_policy(
        env=env,
        policy_fn=random_policy,
        n_episodes=500,
        max_steps=50,
        verbose=True
    )
    print_evaluation_results(random_results, policy_name="Random Policy")

    # Evaluate Heuristic Policy
    print("\n" + "-"*60)
    print("Evaluating Heuristic Policy...")
    print("-"*60)
    heuristic_results = evaluate_policy(
        env=env,
        policy_fn=heuristic_policy,
        n_episodes=500,
        max_steps=50,
        verbose=True
    )
    print_evaluation_results(heuristic_results, policy_name="Heuristic Policy")

    env.close()

    # Compare policies
    print("\n" + "-"*60)
    print("Comparison")
    print("-"*60)
    results_dict = {
        'Random': random_results,
        'Heuristic': heuristic_results
    }
    compare_policies(results_dict)

    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Survival Rate
    policies = ['Random', 'Heuristic']
    survival_rates = [
        random_results['survival_rate'] * 100,
        heuristic_results['survival_rate'] * 100
    ]
    axes[0].bar(policies, survival_rates, color=['#e74c3c', '#3498db'])
    axes[0].set_ylabel('Survival Rate (%)')
    axes[0].set_title('Survival Rate Comparison')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(survival_rates):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    # Plot 2: Average Return
    avg_returns = [random_results['avg_return'], heuristic_results['avg_return']]
    std_returns = [random_results['std_return'], heuristic_results['std_return']]
    axes[1].bar(policies, avg_returns, yerr=std_returns, color=['#e74c3c', '#3498db'], capsize=5)
    axes[1].set_ylabel('Average Return')
    axes[1].set_title('Average Return Comparison')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: SOFA-stratified
    categories = ['Low\nSOFA', 'Medium\nSOFA', 'High\nSOFA']
    x = np.arange(len(categories))
    width = 0.35

    # Handle NaN values (when no episodes in a category)
    random_sofa = [
        random_results['sofa_stratified']['low_sofa']['survival_rate'] * 100
        if not np.isnan(random_results['sofa_stratified']['low_sofa']['survival_rate']) else 0,
        random_results['sofa_stratified']['medium_sofa']['survival_rate'] * 100
        if not np.isnan(random_results['sofa_stratified']['medium_sofa']['survival_rate']) else 0,
        random_results['sofa_stratified']['high_sofa']['survival_rate'] * 100
        if not np.isnan(random_results['sofa_stratified']['high_sofa']['survival_rate']) else 0
    ]
    heuristic_sofa = [
        heuristic_results['sofa_stratified']['low_sofa']['survival_rate'] * 100
        if not np.isnan(heuristic_results['sofa_stratified']['low_sofa']['survival_rate']) else 0,
        heuristic_results['sofa_stratified']['medium_sofa']['survival_rate'] * 100
        if not np.isnan(heuristic_results['sofa_stratified']['medium_sofa']['survival_rate']) else 0,
        heuristic_results['sofa_stratified']['high_sofa']['survival_rate'] * 100
        if not np.isnan(heuristic_results['sofa_stratified']['high_sofa']['survival_rate']) else 0
    ]

    axes[2].bar(x - width/2, random_sofa, width, label='Random', color='#e74c3c')
    axes[2].bar(x + width/2, heuristic_sofa, width, label='Heuristic', color='#3498db')
    axes[2].set_ylabel('Survival Rate (%)')
    axes[2].set_title('SOFA-Stratified Survival Rate')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories)
    axes[2].legend()
    axes[2].set_ylim(0, 100)

    plt.tight_layout()

    # Save figure
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "baseline_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Figure saved to results/figures/baseline_comparison.png")
    plt.close()

    # Save results
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = {
        'random': random_results,
        'heuristic': heuristic_results
    }

    results_file = results_dir / "baseline_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(baseline_results, f)
    print(f"[OK] Results saved to results/baseline_results.pkl")

    print("\n" + "="*60)
    print("BASELINE EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nResults:")
    print(f"  Random Policy: {random_results['survival_rate']*100:.1f}% survival")
    print(f"  Heuristic Policy: {heuristic_results['survival_rate']*100:.1f}% survival")
    print(f"\nNext step:")
    print(f"  python scripts/02_train_bc.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
