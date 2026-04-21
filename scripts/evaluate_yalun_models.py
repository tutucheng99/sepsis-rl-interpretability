"""
Evaluate Yalun's GitHub Models with Strict 500-Episode Protocol

This script evaluates the three online RL models from results/models/:
- ddqn_attention.d3 (DoubleDQN + Attention)
- ddqn_residual.d3 (DoubleDQN + Residual)
- sac.d3 (DiscreteSAC)

Using the same rigorous protocol as your BC/CQL/DQN evaluation:
- 500 episodes per model
- SOFA-stratified analysis
- Comparison with your baseline results

Author: Claude Code
Date: 2025-10-28
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "gym-sepsis"))

# Import d3rlpy and PyTorch
import d3rlpy
import torch
import torch.nn as nn
from typing import Sequence
from d3rlpy.models.encoders import Encoder, EncoderFactory
from d3rlpy.types import Shape

# Import project evaluation tools
from src.envs.sepsis_wrapper import make_sepsis_env
from src.evaluation.metrics import evaluate_policy, print_evaluation_results


# ============================================================================
# PART 1: Custom Encoder Definitions (from Yalun's notebook)
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

        # IMPORTANT: Must be named input_proj (not input_layer) to match trained model
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
# PART 2: Register Encoders with d3rlpy
# ============================================================================

def register_custom_encoders():
    """Register custom encoder factories with d3rlpy's CONFIG_LIST"""
    # Direct registration without importing internal functions

    # Register deep_residual
    try:
        if not hasattr(d3rlpy.models.encoders, 'CONFIG_LIST'):
            print("[ERROR] CONFIG_LIST not found in d3rlpy.models.encoders")
            return False

        d3rlpy.models.encoders.CONFIG_LIST['deep_residual'] = DeepResidualEncoderFactory
        print("[OK] Registered 'deep_residual' encoder")
    except Exception as e:
        print(f"[WARNING] deep_residual registration: {e}")

    # Register attention
    try:
        d3rlpy.models.encoders.CONFIG_LIST['attention'] = AttentionEncoderFactory
        print("[OK] Registered 'attention' encoder")
    except Exception as e:
        print(f"[WARNING] attention registration: {e}")

    return True


# ============================================================================
# PART 3: Model Loading and Evaluation
# ============================================================================

def load_yalun_model(model_path: Path, model_name: str):
    """
    Load a single Yalun model with proper encoder support

    Args:
        model_path: Path to .d3 model file
        model_name: Display name for the model

    Returns:
        Loaded d3rlpy model or None if failed
    """
    print(f"\n{'='*80}")
    print(f"Loading: {model_name}")
    print(f"{'='*80}")

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    try:
        # Load model using d3rlpy
        model = d3rlpy.load_learnable(str(model_path), device='cpu')
        print(f"[OK] Successfully loaded")
        print(f"     Model type: {type(model).__name__}")
        return model

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("\nDiagnostics:")
        import traceback
        traceback.print_exc()
        return None


def evaluate_yalun_model(model, model_name: str, n_episodes: int = 500):
    """
    Evaluate a Yalun model using your project's strict protocol

    Args:
        model: Loaded d3rlpy model
        model_name: Display name
        n_episodes: Number of evaluation episodes (default: 500)

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}\n")

    # Create environment with simple reward (matching your evaluation)
    env = make_sepsis_env(reward_fn_name='simple', verbose=False)

    # Define policy function
    def policy_fn(state):
        state_batch = np.array([state], dtype=np.float32)
        action = model.predict(state_batch)[0]
        return int(action)

    # Evaluate using your project's metrics
    print(f"Running {n_episodes} episodes...")
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

    # Print results using your project's format
    print_evaluation_results(results, policy_name=model_name)

    return results


def load_your_baseline_results():
    """Load your existing BC/CQL/DQN/Random/Heuristic results for comparison"""
    results_dir = project_root / "results"
    baseline_results = {}

    # Load RL algorithm results
    for model_name in ['bc', 'cql', 'dqn']:
        results_file = results_dir / f"{model_name}_results.pkl"
        if results_file.exists():
            try:
                with open(results_file, 'rb') as f:
                    data = pickle.load(f)
                    baseline_results[model_name.upper()] = data['evaluation']
                    print(f"[OK] Loaded {model_name.upper()} results")
            except Exception as e:
                print(f"[WARNING] Could not load {model_name.upper()}: {e}")

    # Load baseline policies
    baseline_file = results_dir / "baseline_results.pkl"
    if baseline_file.exists():
        try:
            with open(baseline_file, 'rb') as f:
                data = pickle.load(f)
                if 'random_policy' in data:
                    baseline_results['Random'] = data['random_policy']
                if 'heuristic_policy' in data:
                    baseline_results['Heuristic'] = data['heuristic_policy']
                print(f"[OK] Loaded Random and Heuristic baselines")
        except Exception as e:
            print(f"[WARNING] Could not load baselines: {e}")

    return baseline_results


def print_comprehensive_comparison(yalun_results: dict, your_results: dict):
    """Print comprehensive comparison table"""
    print("\n" + "="*120)
    print("COMPREHENSIVE COMPARISON: YALUN'S MODELS vs YOUR PROJECT")
    print("="*120 + "\n")

    # Header
    print(f"{'Model':<35} {'Overall Survival':<20} {'Avg Return':<20} {'High SOFA Survival':<20} {'High SOFA n'}")
    print("-" * 120)

    # Order: Your models first, then Yalun's models
    model_order = [
        'Random', 'Heuristic',  # Baselines
        'BC', 'CQL', 'DQN',  # Your offline/online RL
        'DDQN-Attention', 'DDQN-Residual', 'SAC'  # Yalun's models
    ]

    all_results = {**your_results, **yalun_results}

    for model_name in model_order:
        if model_name not in all_results:
            continue

        res = all_results[model_name]
        overall_survival = res['survival_rate'] * 100
        avg_return = res['avg_return']

        # High SOFA metrics
        if 'sofa_stratified' in res and 'high_sofa' in res['sofa_stratified']:
            high_sofa = res['sofa_stratified']['high_sofa']['survival_rate'] * 100
            high_n = res['sofa_stratified']['high_sofa']['n_episodes']
        else:
            high_sofa = np.nan
            high_n = 0

        print(f"{model_name:<35} {overall_survival:>6.1f}%             {avg_return:>7.2f}±{res['std_return']:>5.2f}       "
              f"{high_sofa:>6.1f}%             {high_n:<10}")

    # Analysis section
    print("\n" + "="*120)
    print("KEY INSIGHTS")
    print("="*120 + "\n")

    # Compare Yalun's best with your Random baseline
    if 'Random' in your_results:
        random_survival = your_results['Random']['survival_rate'] * 100
        print(f"Your Random Baseline: {random_survival:.1f}%")
        print("This reveals the true environment difficulty\n")

        yalun_models = ['DDQN-Attention', 'DDQN-Residual', 'SAC']
        yalun_survivals = [(name, yalun_results[name]['survival_rate'] * 100)
                           for name in yalun_models if name in yalun_results]

        if yalun_survivals:
            yalun_survivals.sort(key=lambda x: x[1], reverse=True)
            print("Yalun's Models vs Random Baseline:")
            for name, survival in yalun_survivals:
                diff = survival - random_survival
                status = "✓ Better" if diff > 1.0 else ("✗ Worse" if diff < -1.0 else "≈ Similar")
                print(f"  {name:20s}: {survival:>5.1f}% (diff: {diff:+.1f}%) [{status}]")

    # Compare with your best model
    print("\nComparison with Your Best Model:")
    your_models = ['BC', 'CQL', 'DQN']
    your_survivals = [(name, your_results[name]['survival_rate'] * 100)
                      for name in your_models if name in your_results]

    if your_survivals:
        your_best_name, your_best_survival = max(your_survivals, key=lambda x: x[1])
        print(f"Your Best: {your_best_name} at {your_best_survival:.1f}%\n")

        if yalun_survivals:
            yalun_best_name, yalun_best_survival = yalun_survivals[0]
            diff = yalun_best_survival - your_best_survival
            if diff > 1.0:
                print(f"→ Yalun's {yalun_best_name} outperforms by {diff:+.1f}%")
            elif diff < -1.0:
                print(f"→ Your {your_best_name} outperforms by {-diff:+.1f}%")
            else:
                print(f"→ Performance is similar (diff: {diff:+.1f}%)")

    # High SOFA comparison
    print("\nHigh SOFA Patient Performance (Most Critical):")
    for model_name in model_order:
        if model_name in all_results:
            res = all_results[model_name]
            if 'sofa_stratified' in res and 'high_sofa' in res['sofa_stratified']:
                high = res['sofa_stratified']['high_sofa']['survival_rate'] * 100
                high_n = res['sofa_stratified']['high_sofa']['n_episodes']
                print(f"  {model_name:20s}: {high:>5.1f}% (n={high_n})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main evaluation pipeline"""
    print("\n" + "="*80)
    print("EVALUATING YALUN'S GITHUB MODELS")
    print("="*80)
    print("\nProtocol:")
    print("  - 500 episodes per model (matching your evaluation)")
    print("  - SOFA-stratified analysis (Low/Medium/High)")
    print("  - Direct comparison with your BC/CQL/DQN results")
    print("  - Expected time: ~20-30 minutes total\n")

    # Step 1: Register custom encoders
    print("="*80)
    print("STEP 1: Registering Custom Encoders")
    print("="*80)
    register_custom_encoders()

    # Step 2: Define model paths
    models_dir = project_root / "results" / "models"
    models_to_evaluate = {
        'DDQN-Attention': models_dir / "ddqn_attention.d3",
        'DDQN-Residual': models_dir / "ddqn_residual.d3",
        'SAC': models_dir / "sac.d3"
    }

    n_episodes = 500

    # Step 3: Load and evaluate each model
    yalun_results = {}

    for i, (model_name, model_path) in enumerate(models_to_evaluate.items(), 1):
        print(f"\n{'='*80}")
        print(f"STEP {i+1}/4: {model_name}")
        print(f"{'='*80}")

        # Load model
        model = load_yalun_model(model_path, model_name)

        if model is None:
            print(f"[ERROR] Skipping {model_name} evaluation\n")
            continue

        # Evaluate model
        try:
            results = evaluate_yalun_model(model, model_name, n_episodes)
            yalun_results[model_name] = results
            print(f"\n[OK] {model_name} evaluation complete\n")
        except Exception as e:
            print(f"\n[ERROR] {model_name} evaluation failed: {e}\n")
            import traceback
            traceback.print_exc()

    # Step 4: Load your baseline results
    print(f"\n{'='*80}")
    print("STEP 5/5: Loading Your Baseline Results")
    print(f"{'='*80}\n")

    your_results = load_your_baseline_results()

    # Step 5: Print comparison
    if yalun_results:
        print_comprehensive_comparison(yalun_results, your_results)

        # Save results
        print(f"\n{'='*80}")
        print("SAVING RESULTS")
        print(f"{'='*80}\n")

        results_file = project_root / "results" / "yalun_models_evaluation.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'yalun_results': yalun_results,
                'your_results': your_results,
                'n_episodes': n_episodes,
                'evaluation_date': '2025-10-28',
                'notes': 'Evaluation of Yalun\'s GitHub models using your strict 500-episode protocol'
            }, f)
        print(f"[OK] Results saved to: {results_file}")

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"\nSuccessfully evaluated {len(yalun_results)}/3 models")
        print(f"All models tested with {n_episodes} episodes")
        print("\nNext steps:")
        print("  1. Review comparison table above")
        print("  2. Analyze why Yalun's models perform differently")
        print("  3. Update methodology_comparison.md")
        print("  4. Decide if you want to include these in your paper")

        return 0
    else:
        print(f"\n{'='*80}")
        print("[ERROR] No models were successfully evaluated")
        print(f"{'='*80}")
        print("\nTroubleshooting:")
        print("  1. Check that model files exist in results/models/")
        print("  2. Verify d3rlpy version compatibility")
        print("  3. Ensure all dependencies are installed")

        return 1


if __name__ == "__main__":
    sys.exit(main())
