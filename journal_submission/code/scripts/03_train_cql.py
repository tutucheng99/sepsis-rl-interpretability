"""
Script 3: Train Conservative Q-Learning (CQL)

Trains CQL algorithm on offline dataset using d3rlpy.
Evaluates performance and saves model.

Usage:
    python scripts/03_train_cql.py
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
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset


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


def train_cql(mdp_dataset, reward_fn_name='simple', n_epochs=20, batch_size=1024):
    """Train CQL algorithm with optimized settings"""
    import time

    print(f"\n{'='*60}")
    print(f"TRAINING CQL WITH {reward_fn_name.upper()} REWARD")
    print(f"{'='*60}\n")

    # Configure CQL
    print("Configuring CQL algorithm...")
    cql = DiscreteCQLConfig(
        batch_size=batch_size,
        learning_rate=3e-4,
        target_update_interval=2000,
        alpha=1.0,  # Conservative penalty
    ).create(device='cpu')

    print(f"✅ CQL configured:")
    print(f"   Batch size: {batch_size} (4x larger for efficiency)")
    print(f"   Learning rate: 3e-4")
    print(f"   Alpha (conservatism): 1.0")
    print(f"   Device: CPU")
    print(f"   Epochs: {n_epochs} (optimized)")

    # Optimize training steps
    # CQL needs more training than BC, but not 100 epochs
    full_steps_per_epoch = mdp_dataset.transition_count
    steps_per_epoch = min(full_steps_per_epoch, 10000)  # CQL needs more steps than BC
    total_steps = n_epochs * steps_per_epoch

    print(f"\n⚡ Optimized Training Plan:")
    print(f"   Full dataset size: {full_steps_per_epoch:,} transitions")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Estimated time: 15-20 minutes")

    print(f"\n{'='*60}")
    print("Training started...")
    print(f"{'='*60}\n")

    start_time = time.time()

    cql.fit(
        mdp_dataset,
        n_steps=total_steps,
        n_steps_per_epoch=steps_per_epoch,
        show_progress=True
    )

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("✅ CQL training complete!")
    print(f"{'='*60}")
    print(f"   Training time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"   Steps completed: {total_steps:,}")

    return cql


def evaluate_cql_policy(cql_model, reward_fn_name='simple', n_episodes=200):
    """Evaluate trained CQL policy"""
    print(f"\n{'='*60}")
    print(f"EVALUATING CQL POLICY")
    print(f"{'='*60}\n")

    # Create environment
    env = make_sepsis_env(reward_fn_name=reward_fn_name, verbose=False)

    # Define policy function
    def cql_policy(state):
        # d3rlpy expects batch dimension
        state_batch = np.array([state], dtype=np.float32)
        action = cql_model.predict(state_batch)[0]
        return int(action)

    # Evaluate
    results = evaluate_policy(
        env=env,
        policy_fn=cql_policy,
        n_episodes=n_episodes,
        max_steps=50,
        verbose=True
    )

    env.close()

    print_evaluation_results(results, policy_name=f"CQL ({reward_fn_name} reward)")

    return results


def main():
    """Run CQL training"""
    print("\n" + "="*60)
    print("CONSERVATIVE Q-LEARNING TRAINING")
    print("="*60 + "\n")

    # Load dataset
    dataset_dict = load_offline_dataset()

    # Convert to MDPDataset
    mdp_dataset = create_mdp_dataset(dataset_dict)

    # Train CQL with optimized settings
    reward_fn_name = 'simple'
    cql_model = train_cql(
        mdp_dataset,
        reward_fn_name=reward_fn_name,
        n_epochs=20,      # Reduced: CQL still needs more than BC
        batch_size=1024   # Increased: more efficient training
    )

    # Save model
    models_dir = project_root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"cql_{reward_fn_name}_reward.d3"
    cql_model.save(str(model_path))
    print(f"\n✅ Model saved: {model_path}")

    # Evaluate
    results = evaluate_cql_policy(
        cql_model,
        reward_fn_name=reward_fn_name,
        n_episodes=200
    )

    # Save results
    results_dir = project_root / "results"
    results_file = results_dir / "cql_results.pkl"

    cql_results = {
        'model_path': str(model_path),
        'reward_fn': reward_fn_name,
        'evaluation': results
    }

    with open(results_file, 'wb') as f:
        pickle.dump(cql_results, f)
    print(f"✅ Results saved: {results_file}")

    # Summary
    print("\n" + "="*60)
    print("CQL TRAINING COMPLETE!")
    print("="*60)
    print(f"\nPerformance:")
    print(f"  Survival Rate: {results['survival_rate']*100:.1f}%")
    print(f"  Average Return: {results['avg_return']:.2f} ± {results['std_return']:.2f}")
    print(f"\nNext step:")
    print(f"  python scripts/04_train_dqn.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
