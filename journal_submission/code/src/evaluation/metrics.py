"""
Evaluation Metrics for Sepsis Treatment RL

Provides comprehensive evaluation metrics including:
- Survival rate and mortality analysis
- SOFA-stratified performance (low/medium/high severity)
- Episode statistics and reward tracking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
import sys
from pathlib import Path

# Feature indices (must match sepsis_env.py)
SOFA_IDX = 37


def evaluate_policy(env,
                   policy_fn: Callable,
                   n_episodes: int = 100,
                   max_steps: int = 50,
                   verbose: bool = False) -> Dict:
    """
    Evaluate a policy on the environment

    Args:
        env: Sepsis environment (wrapped or original)
        policy_fn: Function that takes state and returns action
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        verbose: Print progress

    Returns:
        results: Dictionary containing:
            - survival_rate: Proportion of episodes where patient survived
            - avg_return: Average cumulative reward
            - avg_episode_length: Average number of steps
            - all_returns: List of all episode returns
            - all_lengths: List of all episode lengths
            - all_survivals: List of survival outcomes (True/False)
            - sofa_stratified: SOFA-stratified metrics
    """
    returns = []
    lengths = []
    survivals = []
    episode_data = []  # Store (initial_sofa, return, length, survived)

    for ep in range(n_episodes):
        obs, info = env.reset()
        initial_sofa = obs[SOFA_IDX]

        done = False
        episode_return = 0
        episode_length = 0

        while not done and episode_length < max_steps:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

        # Determine survival from terminal outcome
        # For simple reward (±15 terminal), check if final reward dominates
        # Assumption: terminal reward (±15) will dominate intermediate rewards
        if abs(reward) > 10:  # Terminal reward (±15) clearly dominates
            survived = reward > 0
        else:
            # Fallback: use cumulative return (only accurate for simple reward)
            survived = episode_return > 0

        returns.append(episode_return)
        lengths.append(episode_length)
        survivals.append(survived)
        episode_data.append((initial_sofa, episode_return, episode_length, survived))

        if verbose and (ep + 1) % 20 == 0:
            print(f"Evaluated {ep+1}/{n_episodes} episodes")

    # Compute SOFA-stratified metrics
    sofa_stratified = compute_sofa_stratified_metrics(episode_data)

    results = {
        'survival_rate': np.mean(survivals),
        'mortality_rate': 1 - np.mean(survivals),
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'avg_episode_length': np.mean(lengths),
        'std_episode_length': np.std(lengths),
        'all_returns': returns,
        'all_lengths': lengths,
        'all_survivals': survivals,
        'n_episodes': n_episodes,
        'sofa_stratified': sofa_stratified,
    }

    return results


def compute_sofa_stratified_metrics(episode_data: List[Tuple]) -> Dict:
    """
    Compute metrics stratified by SOFA score categories

    NOTE: SOFA scores in the environment are STANDARDIZED (z-score normalized).
    Original thresholds (5, 15) correspond to approximately (-0.45, 0.21)
    in standardized space based on the starting state distribution.

    Categories (standardized SOFA):
    - Low SOFA: < -0.45 (approximately lowest 33%)
    - Medium SOFA: -0.45 to 0.21 (approximately middle 33%)
    - High SOFA: > 0.21 (approximately highest 33%)

    Args:
        episode_data: List of (initial_sofa, return, length, survived) tuples

    Returns:
        stratified_metrics: Dict with 'low', 'medium', 'high' keys
    """
    # Thresholds for standardized SOFA scores
    LOW_THRESHOLD = -0.45
    HIGH_THRESHOLD = 0.21

    # Categorize episodes
    low_sofa = []      # SOFA < LOW_THRESHOLD
    medium_sofa = []   # LOW_THRESHOLD <= SOFA <= HIGH_THRESHOLD
    high_sofa = []     # SOFA > HIGH_THRESHOLD

    for initial_sofa, episode_return, length, survived in episode_data:
        if initial_sofa < LOW_THRESHOLD:
            low_sofa.append((episode_return, length, survived))
        elif initial_sofa <= HIGH_THRESHOLD:
            medium_sofa.append((episode_return, length, survived))
        else:
            high_sofa.append((episode_return, length, survived))

    # Compute metrics for each category
    def compute_category_metrics(data):
        if len(data) == 0:
            return {
                'n_episodes': 0,
                'survival_rate': np.nan,
                'avg_return': np.nan,
                'avg_length': np.nan,
            }

        returns, lengths, survivals = zip(*data)
        return {
            'n_episodes': len(data),
            'survival_rate': np.mean(survivals),
            'mortality_rate': 1 - np.mean(survivals),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
        }

    return {
        'low_sofa': compute_category_metrics(low_sofa),
        'medium_sofa': compute_category_metrics(medium_sofa),
        'high_sofa': compute_category_metrics(high_sofa),
    }


def print_evaluation_results(results: Dict, policy_name: str = "Policy"):
    """
    Pretty print evaluation results

    Args:
        results: Results dictionary from evaluate_policy
        policy_name: Name of the policy for display
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {policy_name}")
    print('='*60)

    # Overall metrics
    print(f"\nOverall Performance ({results['n_episodes']} episodes):")
    print(f"  Survival Rate:      {results['survival_rate']*100:.1f}%")
    print(f"  Mortality Rate:     {results['mortality_rate']*100:.1f}%")
    print(f"  Avg Return:         {results['avg_return']:.2f} +/- {results['std_return']:.2f}")
    print(f"  Avg Episode Length: {results['avg_episode_length']:.1f} +/- {results['std_episode_length']:.1f}")

    # SOFA-stratified metrics
    print(f"\nSOFA-Stratified Performance:")
    for category_name, category_display in [
        ('low_sofa', 'Low SOFA (least severe)'),
        ('medium_sofa', 'Medium SOFA'),
        ('high_sofa', 'High SOFA (most severe)')
    ]:
        cat_metrics = results['sofa_stratified'][category_name]
        n_eps = cat_metrics['n_episodes']

        if n_eps > 0:
            print(f"\n  {category_display} (n={n_eps}):")
            print(f"    Survival Rate: {cat_metrics['survival_rate']*100:.1f}%")
            print(f"    Avg Return:    {cat_metrics['avg_return']:.2f} +/- {cat_metrics['std_return']:.2f}")
            print(f"    Avg Length:    {cat_metrics['avg_length']:.1f} +/- {cat_metrics['std_length']:.1f}")
        else:
            print(f"\n  {category_display}: No episodes")

    print(f"\n{'='*60}\n")


def compare_policies(results_dict: Dict[str, Dict]) -> None:
    """
    Compare multiple policies side-by-side

    Args:
        results_dict: Dict mapping policy names to their evaluation results
    """
    print(f"\n{'='*80}")
    print("Policy Comparison")
    print('='*80)

    # Overall comparison
    print(f"\n{'Policy':<20} {'Survival %':<12} {'Avg Return':<15} {'Avg Length':<12}")
    print('-'*80)

    for policy_name, results in results_dict.items():
        print(f"{policy_name:<20} "
              f"{results['survival_rate']*100:>10.1f}% "
              f"{results['avg_return']:>10.2f} +/- {results['std_return']:<5.2f} "
              f"{results['avg_episode_length']:>8.1f}")

    # SOFA-stratified comparison
    for category_name, category_display in [
        ('low_sofa', 'Low SOFA (least severe)'),
        ('medium_sofa', 'Medium SOFA'),
        ('high_sofa', 'High SOFA (most severe)')
    ]:
        print(f"\n{category_display}:")
        print(f"{'Policy':<20} {'Survival %':<12} {'Avg Return':<15} {'n episodes':<12}")
        print('-'*80)

        for policy_name, results in results_dict.items():
            cat_metrics = results['sofa_stratified'][category_name]
            n_eps = cat_metrics['n_episodes']

            if n_eps > 0:
                print(f"{policy_name:<20} "
                      f"{cat_metrics['survival_rate']*100:>10.1f}% "
                      f"{cat_metrics['avg_return']:>10.2f} +/- {cat_metrics['std_return']:<5.2f} "
                      f"{n_eps:>8d}")
            else:
                print(f"{policy_name:<20} {'No episodes':>23}")

    print(f"\n{'='*80}\n")


# For testing
if __name__ == "__main__":
    print("Evaluation metrics module loaded successfully!")
    print("\nAvailable functions:")
    print("  - evaluate_policy(env, policy_fn, n_episodes)")
    print("  - compute_sofa_stratified_metrics(episode_data)")
    print("  - print_evaluation_results(results, policy_name)")
    print("  - compare_policies(results_dict)")