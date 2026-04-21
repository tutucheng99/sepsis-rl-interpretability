"""
Script 4: Train Deep Q-Network (DQN)

Trains DQN algorithm using Stable-Baselines3.
This is an online RL method that learns through interaction.

Usage:
    python scripts/04_train_dqn.py
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
from src.evaluation.metrics import evaluate_policy, print_evaluation_results

# Stable-Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """
    Custom callback for displaying training progress.
    """
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        """Initialize progress bar"""
        self.pbar = tqdm(total=self.total_timesteps, desc="Training DQN")

    def _on_step(self):
        """Update progress bar"""
        if self.pbar:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        """Close progress bar"""
        if self.pbar:
            self.pbar.close()


def train_dqn(reward_fn_name='simple', total_timesteps=50000, learning_rate=1e-4):
    """Train DQN algorithm with optimized settings"""
    import time

    print(f"\n{'='*60}")
    print(f"TRAINING DQN WITH {reward_fn_name.upper()} REWARD")
    print(f"{'='*60}\n")

    # Create environment
    print("Creating environment...")
    env = make_sepsis_env(reward_fn_name=reward_fn_name, verbose=False)
    print(f"✅ Environment created")

    # Configure DQN with optimized parameters
    print("\nConfiguring DQN algorithm...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=20000,          # Reduced: episodes are short (~10 steps)
        learning_starts=2000,       # Start learning after 2000 steps
        batch_size=1024,            # Increased: 4x larger for efficiency (matches BC/CQL)
        tau=0.005,
        gamma=0.99,
        train_freq=8,               # Train every 8 steps (less frequent = faster)
        target_update_interval=500, # Update target more frequently for short episodes
        exploration_fraction=0.2,   # Explore for 20% of training
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        device='cpu'
    )

    print(f"✅ DQN configured:")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Buffer size: 20,000 (optimized for short episodes)")
    print(f"   Batch size: 1,024 (4x larger for efficiency)")
    print(f"   Train freq: 8 (reduced for speed)")
    print(f"   Device: CPU")
    print(f"   Total timesteps: {total_timesteps:,} (reduced from 200K)")

    # Train
    print(f"\n⚡ Optimized Training Plan:")
    print(f"   Episodes are ~10 steps, so {total_timesteps:,} steps ≈ {total_timesteps//10:,} episodes")
    print(f"   Estimated time: 20-30 minutes (vs 11 hours original)")

    print(f"\n{'='*60}")
    print("Training started...")
    print(f"{'='*60}\n")

    start_time = time.time()

    callback = ProgressCallback(total_timesteps=total_timesteps)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False  # Using custom callback
    )

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("✅ DQN training complete!")
    print(f"{'='*60}")
    print(f"   Training time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"   Steps completed: {total_timesteps:,}")

    env.close()

    return model


def evaluate_dqn_policy(dqn_model, reward_fn_name='simple', n_episodes=200):
    """Evaluate trained DQN policy"""
    print(f"\n{'='*60}")
    print(f"EVALUATING DQN POLICY")
    print(f"{'='*60}\n")

    # Create environment
    env = make_sepsis_env(reward_fn_name=reward_fn_name, verbose=False)

    # Define policy function
    def dqn_policy(state):
        # SB3 expects observation, deterministic=True for evaluation
        action, _ = dqn_model.predict(state, deterministic=True)
        return int(action)

    # Evaluate
    results = evaluate_policy(
        env=env,
        policy_fn=dqn_policy,
        n_episodes=n_episodes,
        max_steps=50,
        verbose=True
    )

    env.close()

    print_evaluation_results(results, policy_name=f"DQN ({reward_fn_name} reward)")

    return results


def main():
    """Run DQN training"""
    print("\n" + "="*60)
    print("DEEP Q-NETWORK TRAINING")
    print("="*60 + "\n")

    # Train DQN with optimized settings
    reward_fn_name = 'simple'
    dqn_model = train_dqn(
        reward_fn_name=reward_fn_name,
        total_timesteps=50000,   # Reduced: 4x less (episodes are ~10 steps)
        learning_rate=1e-4
    )

    # Save model
    models_dir = project_root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"dqn_{reward_fn_name}_reward.zip"
    dqn_model.save(str(model_path))
    print(f"\n✅ Model saved: {model_path}")

    # Evaluate
    results = evaluate_dqn_policy(
        dqn_model,
        reward_fn_name=reward_fn_name,
        n_episodes=200
    )

    # Save results
    results_dir = project_root / "results"
    results_file = results_dir / "dqn_results.pkl"

    dqn_results = {
        'model_path': str(model_path),
        'reward_fn': reward_fn_name,
        'evaluation': results
    }

    with open(results_file, 'wb') as f:
        pickle.dump(dqn_results, f)
    print(f"✅ Results saved: {results_file}")

    # Summary
    print("\n" + "="*60)
    print("DQN TRAINING COMPLETE!")
    print("="*60)
    print(f"\nPerformance:")
    print(f"  Survival Rate: {results['survival_rate']*100:.1f}%")
    print(f"  Average Return: {results['avg_return']:.2f} ± {results['std_return']:.2f}")
    print(f"\nNext step:")
    print(f"  python scripts/05_reward_comparison.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
