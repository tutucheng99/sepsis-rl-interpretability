"""
LEG Analysis for Online RL Models (Stable-Baselines3)

Linearly Estimated Gradient (LEG) analysis for interpreting
online RL policies trained with Stable-Baselines3 (DQN, A2C, PPO, etc.).

Usage:
    python scripts/leg_analysis_online.py --model results/models/dqn_simple_reward.zip --n_states 10
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import pandas as pd
import argparse
import os
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))

from src.envs.sepsis_wrapper import make_sepsis_env

# Import Stable-Baselines3
from stable_baselines3 import DQN

# Feature names from Gym-Sepsis environment
FEATURE_NAMES = [
    'ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
    'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
    'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
    'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
    'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
    'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
    'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
    'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
    'blood_culture_positive'
]

CRITICAL_FEATURES = ['LACTATE', 'MeanBP', 'SysBP', 'sofa', 'HeartRate', 'SpO2']


class LEGAnalyzer_Online:
    """Linearly Estimated Gradient analyzer for Stable-Baselines3 DQN."""

    def __init__(
        self,
        sb3_model,
        feature_names: List[str] = FEATURE_NAMES,
        n_samples: int = 1000,
        perturbation_std: float = 0.1
    ):
        """
        Initialize LEG analyzer for online RL models.

        Args:
            sb3_model: Trained Stable-Baselines3 DQN model
            feature_names: Names of state features
            n_samples: Number of perturbation samples for gradient estimation
            perturbation_std: Standard deviation for perturbations
        """
        self.sb3_model = sb3_model
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.perturbation_std = perturbation_std
        self.action_dim = 24  # Sepsis environment has 24 actions

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions given state.

        For Stable-Baselines3 DQN, we access the q_net directly.
        """
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        # Get Q-values from DQN's Q-network
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(state).to(self.sb3_model.device)
            q_values = self.sb3_model.q_net(obs_tensor).cpu().numpy()[0]

        return q_values

    def compute_saliency_scores(
        self,
        state: np.ndarray,
        feature_indices: List[int] = None,
        feature_ranges: Dict[int, Tuple[float, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute LEG saliency scores for all actions.

        Uses the formula: γ̂(π, s₀, F) = Σ⁻¹ (1/n) Σᵢ(ŷᵢZᵢ)

        Args:
            state: Current state (46-dimensional)
            feature_indices: Indices of features to perturb (None = all)
            feature_ranges: Valid ranges for each feature

        Returns:
            Dictionary mapping action indices to saliency score arrays
        """
        if feature_indices is None:
            feature_indices = list(range(len(state)))

        # Get Q-values for original state
        q_values_original = self.get_q_values(state)

        n_features = len(feature_indices)

        # Initialize saliency scores for each action
        saliency_scores = {action: np.zeros(len(state)) for action in range(self.action_dim)}

        # Generate perturbations from normal distribution
        perturbations = np.random.normal(0, self.perturbation_std,
                                        (self.n_samples, n_features))

        # Compute covariance matrix
        Sigma = np.cov(perturbations.T)
        # Add small ridge for numerical stability
        Sigma += np.eye(Sigma.shape[0]) * 1e-6
        Sigma_inv = np.linalg.pinv(Sigma)

        # Store differences
        Z_list = []
        y_diff_list = {action: [] for action in range(self.action_dim)}

        for i in range(self.n_samples):
            # Create perturbed state
            perturbed_state = state.copy()
            perturbed_state[feature_indices] += perturbations[i]

            # Clip to valid ranges if provided
            if feature_ranges:
                for idx, feature_idx in enumerate(feature_indices):
                    if feature_idx in feature_ranges:
                        min_val, max_val = feature_ranges[feature_idx]
                        perturbed_state[feature_idx] = np.clip(
                            perturbed_state[feature_idx], min_val, max_val
                        )

            # Get Q-values for perturbed state
            q_values_perturbed = self.get_q_values(perturbed_state)

            # Store Z (perturbation)
            Z_list.append(perturbations[i])

            # Store y differences for each action
            for action in range(self.action_dim):
                y_diff = q_values_perturbed[action] - q_values_original[action]
                y_diff_list[action].append(y_diff)

        # Convert to arrays
        Z_array = np.array(Z_list)

        # Compute LEG gradient estimate for each action (Equation 1)
        for action in range(self.action_dim):
            y_diff_array = np.array(y_diff_list[action])

            # γ̂(π, s₀, F) = Σ⁻¹ (1/n) Σᵢ(ŷᵢZᵢ)
            gradient_reduced = Sigma_inv @ (Z_array.T @ y_diff_array) / self.n_samples

            # Map back to full feature space
            for idx, feature_idx in enumerate(feature_indices):
                saliency_scores[action][feature_idx] = gradient_reduced[idx]

        return saliency_scores

    def analyze_state(
        self,
        state: np.ndarray,
        top_k: int = 15,
        feature_subset: List[str] = None
    ) -> Dict:
        """
        Perform comprehensive LEG analysis on a state.

        Args:
            state: Current state
            top_k: Number of top features to return
            feature_subset: Subset of feature names to analyze

        Returns:
            Dictionary with analysis results
        """
        # Determine which features to analyze
        if feature_subset:
            feature_indices = [
                i for i, name in enumerate(self.feature_names)
                if name in feature_subset
            ]
        else:
            # Exclude discrete/categorical features for perturbation
            exclude_features = ['is_male', 'race_white', 'race_black',
                              'race_hispanic', 'race_other']
            feature_indices = [
                i for i, name in enumerate(self.feature_names)
                if name not in exclude_features
            ]

        print(f'Analyzing {len(feature_indices)} features...')

        # Compute saliency scores
        saliency_scores = self.compute_saliency_scores(state, feature_indices)

        # Get Q-values and selected action
        q_values = self.get_q_values(state)
        selected_action = int(np.argmax(q_values))

        # Find top-k most important features for selected action
        action_saliency = saliency_scores[selected_action]
        top_indices = np.argsort(np.abs(action_saliency))[-top_k:][::-1]

        # Decode action to IV and VP bins
        iv_bin = selected_action // 5
        vp_bin = selected_action % 5

        results = {
            'state': state,
            'q_values': q_values,
            'selected_action': selected_action,
            'iv_bin': iv_bin,
            'vp_bin': vp_bin,
            'saliency_scores': saliency_scores,
            'top_features': {
                'indices': top_indices,
                'names': [self.feature_names[i] for i in top_indices],
                'scores': action_saliency[top_indices],
                'values': state[top_indices]
            }
        }

        return results


class LEGVisualizer:
    """Visualization tools for LEG analysis."""

    def __init__(self, feature_names: List[str] = FEATURE_NAMES):
        self.feature_names = feature_names
        sns.set_style("whitegrid")

    def plot_saliency_heatmap(
        self,
        saliency_scores: Dict[int, np.ndarray],
        state: np.ndarray,
        selected_actions: List[int] = None,
        top_k: int = 15,
        figsize: Tuple[int, int] = (16, 10),
        save_path: str = None
    ):
        """Create comprehensive saliency heatmap."""
        if selected_actions is None:
            # Show representative actions
            selected_actions = [0, 6, 12, 18, 23]

        n_actions = len(selected_actions)

        # Find top-k features across selected actions
        all_scores = np.array([saliency_scores[a] for a in selected_actions])
        importance = np.abs(all_scores).sum(axis=0)
        top_indices = np.argsort(importance)[-top_k:][::-1]

        # Prepare data
        data = []
        feature_labels = []
        state_values = []

        for idx in top_indices:
            feature_labels.append(self.feature_names[idx])
            state_values.append(state[idx])
            row = [saliency_scores[a][idx] for a in selected_actions]
            data.append(row)

        data = np.array(data)

        # Normalize for visualization
        max_abs = np.abs(data).max()
        if max_abs > 0:
            data_norm = data / max_abs
        else:
            data_norm = data

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(data_norm, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(n_actions))
        action_labels = []
        for a in selected_actions:
            iv, vp = a // 5, a % 5
            action_labels.append(f'Action {a}\n(IV:{iv}, VP:{vp})')
        ax.set_xticklabels(action_labels, fontsize=10)

        ax.set_yticks(range(top_k))
        ax.set_yticklabels(feature_labels, fontsize=9)

        # Add value annotations
        for i in range(top_k):
            for j in range(n_actions):
                score = data[i, j]
                text_color = 'white' if abs(data_norm[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{score:.3f}',
                       ha='center', va='center',
                       color=text_color, fontsize=8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Normalized LEG Saliency Score', fontsize=11)

        # Title
        ax.set_title('LEG Saliency Analysis for Sepsis Treatment Policy (Online DQN)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Treatment Actions', fontsize=11, fontweight='bold')
        ax.set_ylabel('Physiological Features', fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Saved saliency heatmap to {save_path}')

        plt.close()

    def plot_top_features_detailed(
        self,
        analysis_results: Dict,
        figsize: Tuple[int, int] = (14, 8),
        save_path: str = None
    ):
        """Plot detailed analysis of top features."""
        top_features = analysis_results['top_features']
        selected_action = analysis_results['selected_action']
        iv_bin = analysis_results['iv_bin']
        vp_bin = analysis_results['vp_bin']

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Plot 1: Saliency scores
        ax1 = fig.add_subplot(gs[0, :])
        colors = ['red' if s < 0 else 'steelblue' for s in top_features['scores']]
        ax1.barh(range(len(top_features['names'])),
                top_features['scores'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(top_features['names'])))
        ax1.set_yticklabels(top_features['names'], fontsize=10)
        ax1.set_xlabel('LEG Saliency Score', fontsize=11, fontweight='bold')
        ax1.set_title('Feature Importance for Selected Action',
                     fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()

        # Plot 2: Feature values
        ax2 = fig.add_subplot(gs[1, 0])
        colors_val = ['green' if f in CRITICAL_FEATURES else 'gray'
                      for f in top_features['names']]
        ax2.barh(range(len(top_features['names'])),
                top_features['values'], color=colors_val, alpha=0.6)
        ax2.set_yticks(range(len(top_features['names'])))
        ax2.set_yticklabels(top_features['names'], fontsize=10)
        ax2.set_xlabel('Feature Value (Current State)', fontsize=10, fontweight='bold')
        ax2.set_title('Current Feature Values', fontsize=11, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()

        # Plot 3: Q-values
        ax3 = fig.add_subplot(gs[1, 1])
        q_values = analysis_results['q_values']
        actions = list(range(len(q_values)))

        colors_q = ['red' if a == selected_action else 'gray' for a in actions]
        ax3.bar(actions, q_values, color=colors_q, alpha=0.7, width=0.8)
        ax3.set_xlabel('Action', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Q-Value', fontsize=10, fontweight='bold')
        ax3.set_title('Q-Values for All Actions', fontsize=11, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_xticks(range(0, 24, 3))

        # Super title
        fig.suptitle(
            f'LEG Analysis (Online DQN) - Action {selected_action} (IV={iv_bin}, VP={vp_bin})',
            fontsize=14, fontweight='bold', y=0.98
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Saved detailed analysis to {save_path}')

        plt.close()


def load_online_model(model_path: str):
    """Load trained Stable-Baselines3 DQN model"""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load DQN model from .zip file
    if model_path.suffix.lower() == '.zip':
        model = DQN.load(str(model_path), device='cpu')
        print(f"Loaded SB3 DQN model from {model_path}")
        return model
    else:
        raise ValueError(f"Expected .zip file for SB3 DQN model, got {model_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description='LEG Analysis for Online RL Models')
    parser.add_argument('--model', type=str,
                        default='results/models/dqn_simple_reward.zip',
                        help='Path to trained SB3 DQN model (.zip file)')
    parser.add_argument('--n_states', type=int, default=10,
                        help='Number of states to analyze')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of perturbation samples for LEG')
    parser.add_argument('--output_dir', type=str, default='results/figures/leg_online',
                        help='Output directory for figures')
    args = parser.parse_args()

    print("=" * 60)
    print("LEG ANALYSIS FOR ONLINE RL MODEL (DQN)")
    print("=" * 60 + "\n")

    # Load model
    model = load_online_model(args.model)
    model_name = Path(args.model).stem

    # Create output directory
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer and visualizer
    leg_analyzer = LEGAnalyzer_Online(
        sb3_model=model,
        feature_names=FEATURE_NAMES,
        n_samples=args.n_samples,
        perturbation_std=0.1
    )

    visualizer = LEGVisualizer(FEATURE_NAMES)

    # Sample states from environment
    print(f"\nSampling {args.n_states} states from environment...")
    env = make_sepsis_env(reward_fn_name='simple', verbose=False)

    analyses = []
    for i in range(args.n_states):
        obs, _ = env.reset()

        print(f'\n--- Analyzing State {i+1}/{args.n_states} ---')

        # Perform analysis
        analysis = leg_analyzer.analyze_state(obs, top_k=15)
        analyses.append(analysis)

        # Print summary
        print(f'Selected Action: {analysis["selected_action"]} '
              f'(IV={analysis["iv_bin"]}, VP={analysis["vp_bin"]})')
        print(f'Top 5 Features:')
        for j in range(min(5, len(analysis['top_features']['names']))):
            name = analysis['top_features']['names'][j]
            score = analysis['top_features']['scores'][j]
            value = analysis['top_features']['values'][j]
            print(f'  {j+1}. {name}: score={score:.4f}, value={value:.2f}')

        # Visualize
        visualizer.plot_top_features_detailed(
            analysis,
            save_path=output_dir / f'analysis_state_{i+1}.png'
        )

        visualizer.plot_saliency_heatmap(
            analysis['saliency_scores'],
            obs,
            selected_actions=[0, 6, 12, 18, 23],
            top_k=15,
            save_path=output_dir / f'saliency_state_{i+1}.png'
        )

    env.close()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {output_dir}")
    print(f"Total states analyzed: {len(analyses)}")
    print(f"Total figures generated: {len(analyses) * 2}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
