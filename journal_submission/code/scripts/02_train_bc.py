"""
Script 2: Train Behavior Cloning (BC)

Trains BC algorithm on offline dataset using d3rlpy.
Evaluates performance and saves model.

Usage:
    python scripts/02_train_bc.py
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

# d3rlpy imports
from d3rlpy.algos import DiscreteBCConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import TDErrorEvaluator, DiscountedSumOfAdvantageEvaluator


def load_offline_dataset():
    """Load offline dataset from pickle file"""
    data_file = project_root / "data" / "offline_dataset.pkl"

    if not data_file.exists():
        raise FileNotFoundError(
            f"❌ Offline dataset not found: {data_file}\n"
            f"Please run data generation first."
        )

    print(f"Loading dataset from {data_file}...")
    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)

    print(f"✅ Dataset loaded:")
    print(f"   Observations: {len(dataset['observations'])}")
    print(f"   Actions: {len(dataset['actions'])}")
    print(f"   Rewards: {len(dataset['rewards'])}")
    print(f"   Terminals: {len(dataset['terminals'])}")

    return dataset


def create_mdp_dataset(dataset_dict):
    """Convert our dataset to d3rlpy MDPDataset format"""
    print("\nConverting to d3rlpy MDPDataset format...")

    observations = np.array(dataset_dict['observations'], dtype=np.float32)
    actions = np.array(dataset_dict['actions'], dtype=np.int32)
    rewards = np.array(dataset_dict['rewards'], dtype=np.float32)
    terminals = np.array(dataset_dict['terminals'], dtype=np.float32)

    # d3rlpy expects discrete actions to be 0-indexed integers
    mdp_dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )

    print(f"✅ MDPDataset created:")
    print(f"   Episodes: {len(mdp_dataset.episodes)}")
    # d3rlpy exposes transition_count for total transitions
    print(f"   Total steps: {mdp_dataset.transition_count}")


    return mdp_dataset


def train_bc(mdp_dataset, reward_fn_name='simple', n_epochs=10, batch_size=1024):
    """Train BC algorithm with optimized settings"""
    import time

    print(f"\n{'='*60}")
    print(f"TRAINING BC WITH {reward_fn_name.upper()} REWARD")
    print(f"{'='*60}\n")

    # Configure BC with optimized hyperparameters
    print("Configuring BC algorithm...")
    bc = DiscreteBCConfig(
        batch_size=batch_size,
        learning_rate=1e-3,  # Higher learning rate for larger batch
    ).create(device='cpu')

    print(f"✅ BC configured:")
    print(f"   Batch size: {batch_size} (4x larger for efficiency)")
    print(f"   Learning rate: 1e-3 (optimized for large batch)")
    print(f"   Device: CPU")
    print(f"   Epochs: {n_epochs} (BC converges quickly)")

    # Optimize training steps
    # BC doesn't need to see all data multiple times
    full_steps_per_epoch = mdp_dataset.transition_count
    steps_per_epoch = min(full_steps_per_epoch, 5000)  # Cap at 5000 steps/epoch
    total_steps = n_epochs * steps_per_epoch

    print(f"\n⚡ Optimized Training Plan:")
    print(f"   Full dataset size: {full_steps_per_epoch:,} transitions")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Estimated time: 5-10 minutes")

    print(f"\n{'='*60}")
    print("Training started...")
    print(f"{'='*60}\n")

    start_time = time.time()

    bc.fit(
        mdp_dataset,
        n_steps=total_steps,
        n_steps_per_epoch=steps_per_epoch,
        show_progress=True
    )

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("✅ BC training complete!")
    print(f"{'='*60}")
    print(f"   Training time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"   Steps completed: {total_steps:,}")

    return bc


def evaluate_bc_policy(bc_model, reward_fn_name='simple', n_episodes=200):
    """Evaluate trained BC policy"""
    print(f"\n{'='*60}")
    print(f"EVALUATING BC POLICY")
    print(f"{'='*60}\n")

    # Create environment
    env = make_sepsis_env(reward_fn_name=reward_fn_name, verbose=False)

    # Define policy function
    def bc_policy(state):
        # d3rlpy expects batch dimension
        state_batch = np.array([state], dtype=np.float32)
        action = bc_model.predict(state_batch)[0]
        return int(action)

    # Evaluate
    results = evaluate_policy(
        env=env,
        policy_fn=bc_policy,
        n_episodes=n_episodes,
        max_steps=50,
        verbose=True
    )

    env.close()

    print_evaluation_results(results, policy_name=f"BC ({reward_fn_name} reward)")

    return results


def main():
    """Run BC training"""
    print("\n" + "="*60)
    print("BEHAVIOR CLONING TRAINING")
    print("="*60 + "\n")

    # Load dataset
    dataset_dict = load_offline_dataset()

    # Convert to MDPDataset
    mdp_dataset = create_mdp_dataset(dataset_dict)

    # Train BC with optimized settings
    reward_fn_name = 'simple'
    bc_model = train_bc(
        mdp_dataset,
        reward_fn_name=reward_fn_name,
        n_epochs=10,      # Reduced: BC converges quickly
        batch_size=1024   # Increased: more efficient training
    )

    # Save model
    models_dir = project_root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"bc_{reward_fn_name}_reward.d3"
    bc_model.save(str(model_path))
    print(f"\n✅ Model saved: {model_path}")

    # Evaluate
    results = evaluate_bc_policy(
        bc_model,
        reward_fn_name=reward_fn_name,
        n_episodes=200
    )

    # Save results
    results_dir = project_root / "results"
    results_file = results_dir / "bc_results.pkl"

    bc_results = {
        'model_path': str(model_path),
        'reward_fn': reward_fn_name,
        'evaluation': results
    }

    with open(results_file, 'wb') as f:
        pickle.dump(bc_results, f)
    print(f"✅ Results saved: {results_file}")

    # Summary
    print("\n" + "="*60)
    print("BC TRAINING COMPLETE!")
    print("="*60)
    print(f"\nPerformance:")
    print(f"  Survival Rate: {results['survival_rate']*100:.1f}%")
    print(f"  Average Return: {results['avg_return']:.2f} ± {results['std_return']:.2f}")
    print(f"\nNext step:")
    print(f"  python scripts/03_train_cql.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
