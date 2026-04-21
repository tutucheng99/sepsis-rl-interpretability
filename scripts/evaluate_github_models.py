"""
Evaluate GitHub models (DoubleDQN+Attention, DoubleDQN+Residual, SAC) with strict 500-episode protocol

This script:
1. Loads the three online models from results/models/ directory
2. Defines custom encoder architectures (Attention, Residual)
3. Evaluates with 500 episodes + SOFA stratification
4. Compares with your existing BC/CQL/DQN results

Usage:
    python scripts/evaluate_github_models.py
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn as nn
from typing import Sequence

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.sepsis_wrapper import make_sepsis_env
from src.evaluation.metrics import evaluate_policy, print_evaluation_results

# Import d3rlpy
import d3rlpy
from d3rlpy.models.encoders import Encoder, EncoderFactory, register_encoder_factory
from d3rlpy.types import Shape


# ============================================================================
# STEP 1: Define Custom Encoder Architectures (from notebook)
# ============================================================================

class DeepResidualEncoder(Encoder):
    """Deep encoder with residual connections for better gradient flow."""

    def __init__(self, observation_shape: Shape, hidden_units: Sequence[int] = [256, 256, 256],
                 activation: str = 'relu', dropout_rate: float = 0.1):
        super().__init__()
        self.observation_shape = observation_shape
        self.hidden_units = hidden_units
        self._feature_size = hidden_units[-1]

        # Handle Shape type (can be 1D or 2D)
        input_size = observation_shape[0] if isinstance(observation_shape[0], int) else observation_shape[0][0]

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_units[0])

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(len(hidden_units) - 1):
            self.hidden_layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))

            # Residual projection if dimensions change
            if hidden_units[i] != hidden_units[i + 1]:
                self.residual_layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            else:
                self.residual_layers.append(nn.Identity())

            self.dropout_layers.append(nn.Dropout(dropout_rate))

        # Activation function
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()

        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_units[i]) for i in range(len(hidden_units))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layer
        h = self.activation(self.layer_norms[0](self.input_layer(x)))

        # Hidden layers with residual connections
        for i, (hidden_layer, residual_layer, dropout, norm) in enumerate(
            zip(self.hidden_layers, self.residual_layers, self.dropout_layers, self.layer_norms[1:])
        ):
            residual = residual_layer(h)
            h = hidden_layer(h)
            h = norm(h + residual)  # Residual connection + normalization
            h = self.activation(h)
            h = dropout(h)

        return h

    def get_feature_size(self) -> int:
        return self._feature_size


class DeepResidualEncoderFactory(EncoderFactory):
    """Factory for creating DeepResidualEncoder instances."""

    def __init__(self, hidden_units: Sequence[int] = [256, 256, 256],
                 activation: str = 'relu', dropout_rate: float = 0.1):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate

    def create(self, observation_shape: Shape) -> DeepResidualEncoder:
        return DeepResidualEncoder(
            observation_shape=observation_shape,
            hidden_units=self.hidden_units,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )


class AttentionEncoder(Encoder):
    """Encoder with self-attention mechanism."""

    def __init__(self, observation_shape: Shape, hidden_units: Sequence[int] = [256, 128],
                 n_heads: int = 4, activation: str = 'relu'):
        super().__init__()
        self.observation_shape = observation_shape
        self._feature_size = hidden_units[-1]

        # Handle Shape type (can be 1D or 2D)
        input_size = observation_shape[0] if isinstance(observation_shape[0], int) else observation_shape[0][0]

        # Project input to match hidden size for attention (MUST BE input_proj, not input_layer!)
        self.input_proj = nn.Linear(input_size, hidden_units[0])

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_units[0],
            num_heads=n_heads,
            batch_first=True
        )

        # Feedforward layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            self.fc_layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))

        # Activation
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_units[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        h = self.input_proj(x)

        # Add sequence dimension for attention (batch, seq_len=1, features)
        h = h.unsqueeze(1)

        # Self-attention
        attn_output, _ = self.attention(h, h, h)

        # Remove sequence dimension
        h = attn_output.squeeze(1)

        # Residual connection + layer norm
        h = self.layer_norm(h + self.input_proj(x))
        h = self.activation(h)

        # Feedforward layers
        for fc in self.fc_layers:
            h = self.activation(fc(h))

        return h

    def get_feature_size(self) -> int:
        return self._feature_size


class AttentionEncoderFactory(EncoderFactory):
    """Factory for creating AttentionEncoder instances."""

    def __init__(self, hidden_units: Sequence[int] = [256, 128],
                 n_heads: int = 4, activation: str = 'relu'):
        self.hidden_units = hidden_units
        self.n_heads = n_heads
        self.activation = activation

    def create(self, observation_shape: Shape) -> AttentionEncoder:
        return AttentionEncoder(
            observation_shape=observation_shape,
            hidden_units=self.hidden_units,
            n_heads=self.n_heads,
            activation=self.activation
        )


# ============================================================================
# STEP 2: Load and Evaluate Models
# ============================================================================

def evaluate_github_model(model_path: Path, model_name: str, n_episodes: int = 500):
    """
    Evaluate a single GitHub model with strict protocol

    Args:
        model_path: Path to .d3 model file
        model_name: Display name for the model
        n_episodes: Number of evaluation episodes (default: 500)

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}\n")

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return None

    try:
        # Load model
        print(f"Loading model from: {model_path.name}")
        model = d3rlpy.load_learnable(str(model_path))
        print(f"[OK] Model loaded successfully")
        print(f"     Model type: {type(model).__name__}")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print(f"\nTroubleshooting:")
        print(f"  - Make sure custom encoders are registered")
        print(f"  - Check if model file is corrupted")
        return None

    # Create environment
    env = make_sepsis_env(reward_fn_name='simple', verbose=False)

    # Define policy function
    def policy_fn(state):
        state_batch = np.array([state], dtype=np.float32)
        action = model.predict(state_batch)[0]
        return int(action)

    # Evaluate
    print(f"\nEvaluating with {n_episodes} episodes...")
    print("(This will take ~5-10 minutes)")
    print()

    results = evaluate_policy(
        env=env,
        policy_fn=policy_fn,
        n_episodes=n_episodes,
        max_steps=50,
        verbose=True
    )

    env.close()

    # Print results
    print_evaluation_results(results, policy_name=model_name)

    return results


def load_baseline_results():
    """Load existing BC/CQL/DQN results for comparison"""
    results_dir = project_root / "results"

    baseline_results = {}

    # Try to load each model's results
    for model_name in ['bc', 'cql', 'dqn']:
        results_file = results_dir / f"{model_name}_results.pkl"
        if results_file.exists():
            try:
                with open(results_file, 'rb') as f:
                    data = pickle.load(f)
                    baseline_results[model_name.upper()] = data['evaluation']
                    print(f"[OK] Loaded {model_name.upper()} baseline results")
            except Exception as e:
                print(f"[WARNING] Could not load {model_name.upper()} results: {e}")

    # Load baseline policies (Random, Heuristic)
    baseline_file = results_dir / "baseline_results.pkl"
    if baseline_file.exists():
        try:
            with open(baseline_file, 'rb') as f:
                data = pickle.load(f)
                if 'random_policy' in data:
                    baseline_results['Random'] = data['random_policy']
                if 'heuristic_policy' in data:
                    baseline_results['Heuristic'] = data['heuristic_policy']
                print(f"[OK] Loaded baseline policies (Random, Heuristic)")
        except Exception as e:
            print(f"[WARNING] Could not load baseline results: {e}")

    return baseline_results


def print_comparison_table(github_results: dict, baseline_results: dict):
    """Print comprehensive comparison table"""
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON: GITHUB MODELS vs YOUR PROJECT")
    print("="*100 + "\n")

    # Header
    print(f"{'Model':<30} {'Overall Survival':<20} {'High SOFA Survival':<20} {'High SOFA n':<15}")
    print("-" * 100)

    # All models
    all_results = {**baseline_results, **github_results}

    for model_name in ['Random', 'Heuristic', 'BC', 'CQL', 'DQN',
                       'DDQN-Attention', 'DDQN-Residual', 'SAC']:
        if model_name in all_results:
            res = all_results[model_name]
            overall = res['survival_rate'] * 100

            if 'sofa_stratified' in res and 'high_sofa' in res['sofa_stratified']:
                high = res['sofa_stratified']['high_sofa']['survival_rate'] * 100
                high_n = res['sofa_stratified']['high_sofa']['n_episodes']
            else:
                high = np.nan
                high_n = 0

            print(f"{model_name:<30} {overall:>6.1f}%             {high:>6.1f}%             {high_n:<15}")

    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100 + "\n")

    # Calculate rankings
    github_models = ['DDQN-Attention', 'DDQN-Residual', 'SAC']
    github_survivals = [(name, all_results[name]['survival_rate'] * 100)
                        for name in github_models if name in all_results]
    github_survivals.sort(key=lambda x: x[1], reverse=True)

    print("GitHub Models Ranking (Overall Survival):")
    for rank, (name, survival) in enumerate(github_survivals, 1):
        print(f"  {rank}. {name}: {survival:.1f}%")

    # Compare with baselines
    if 'Random' in all_results:
        random_survival = all_results['Random']['survival_rate'] * 100
        print(f"\nRandom Baseline: {random_survival:.1f}%")
        print("  → Shows environment difficulty")

        for name, survival in github_survivals:
            diff = survival - random_survival
            if diff > 1.0:
                print(f"  → {name} beats Random by {diff:+.1f}% ✓")
            elif diff < -1.0:
                print(f"  → {name} worse than Random by {diff:.1f}% ✗")
            else:
                print(f"  → {name} ≈ Random (diff: {diff:+.1f}%)")


def main():
    """Main evaluation pipeline"""
    print("\n" + "="*80)
    print("GITHUB MODELS EVALUATION WITH STRICT PROTOCOL")
    print("="*80)
    print("\nProtocol:")
    print("  - 500 episodes per model")
    print("  - SOFA-stratified analysis (Low/Medium/High)")
    print("  - Comparison with your existing BC/CQL/DQN results")
    print("  - Expected time: ~20-30 minutes total\n")

    # IMPORTANT: Register custom encoder factories
    print("Registering custom encoders...")
    try:
        register_encoder_factory(DeepResidualEncoderFactory, 'deep_residual')
        register_encoder_factory(AttentionEncoderFactory, 'attention')
        print("[OK] Custom encoders registered: deep_residual, attention\n")
    except Exception as e:
        print(f"[WARNING] Error registering encoders: {e}")
        print("Continuing anyway (encoders may already be registered)\n")

    # Model paths
    models_dir = project_root / "results" / "models"
    models = {
        'DDQN-Attention': models_dir / "ddqn_attention.d3",
        'DDQN-Residual': models_dir / "ddqn_residual.d3",
        'SAC': models_dir / "sac.d3"
    }

    n_episodes = 500

    # Evaluate each model
    github_results = {}

    for i, (model_name, model_path) in enumerate(models.items(), 1):
        print(f"\n{'='*80}")
        print(f"STEP {i}/3: {model_name}")
        print(f"{'='*80}")

        results = evaluate_github_model(model_path, model_name, n_episodes)

        if results is not None:
            github_results[model_name] = results
            print(f"\n[OK] {model_name} evaluation complete")
        else:
            print(f"\n[ERROR] {model_name} evaluation failed")

    # Load baseline results
    print(f"\n{'='*80}")
    print("LOADING BASELINE RESULTS")
    print(f"{'='*80}\n")

    baseline_results = load_baseline_results()

    # Print comparison
    if github_results:
        print_comparison_table(github_results, baseline_results)

        # Save results
        print(f"\n{'='*80}")
        print("SAVING RESULTS")
        print(f"{'='*80}\n")

        results_file = project_root / "results" / "github_models_evaluation.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'github_results': github_results,
                'baseline_results': baseline_results,
                'n_episodes': n_episodes,
                'evaluation_date': '2025-10-28'
            }, f)
        print(f"[OK] Results saved to: {results_file}")

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"\nEvaluated {len(github_results)}/3 GitHub models")
        print(f"All models tested with {n_episodes} episodes")
        print("\nNext steps:")
        print("  1. Review comparison table above")
        print("  2. Check if GitHub models truly outperform your models")
        print("  3. Investigate why (training data, architecture, algorithm)")
        print("  4. Update methodology_comparison.md with findings")

        return 0
    else:
        print(f"\n[ERROR] No models were successfully evaluated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
