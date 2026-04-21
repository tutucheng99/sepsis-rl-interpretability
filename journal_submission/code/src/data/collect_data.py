"""
Data Collection Script for Sepsis RL Project

Collects offline dataset using heuristic policy
Target: 20,000 transitions (~2,100 episodes)
Estimated time: ~1 hour
"""
import sys
import os
import time
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.env_wrapper import make_sepsis_env
import numpy as np


def heuristic_policy(state):
    """
    Clinical rule-based heuristic policy
    
    Args:
        state: 46-dim standardized state vector
    
    Returns:
        action: integer 0-23
    """
    # Key features (standardized)
    lactate = state[15]
    sbp = state[25]
    map_value = state[16]
    sofa = state[37]
    
    # Clinical decision rules
    if sbp < -1.0 or map_value < -1.0:
        iv_bin, vp_bin = 4, 3
    elif lactate > 1.0:
        iv_bin, vp_bin = 3, 2
    elif sofa > 1.0:
        iv_bin, vp_bin = 3, 3
    elif sbp < 0 or lactate > 0:
        iv_bin, vp_bin = 2, 1
    else:
        iv_bin, vp_bin = 1, 1
    
    action = min(5 * iv_bin + vp_bin, 23)
    return action


def collect_episodes(n_episodes, output_dir='data', save_interval=500):
    """
    Collect episodes and save incrementally
    
    Args:
        n_episodes: Number of episodes to collect
        output_dir: Directory to save data
        save_interval: Save checkpoint every N episodes
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env = make_sepsis_env()
    
    # Storage for all data
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_next_observations = []
    
    # Statistics
    total_transitions = 0
    episode_rewards = []
    episode_lengths = []
    
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Starting data collection: {n_episodes} episodes")
    print(f"Target: ~20,000 transitions")
    print(f"{'='*60}\n")
    
    for ep in range(n_episodes):
        # Reset environment
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'next_observations': []
        }
        
        done = False
        ep_reward = 0
        ep_length = 0
        
        while not done:
            # Get action from policy
            action = heuristic_policy(obs)
            
            # Take step
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
            
            # Store transition
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['terminals'].append(done)
            episode_data['next_observations'].append(next_obs)
            
            obs = next_obs
            ep_reward += reward
            ep_length += 1
            total_transitions += 1
        
        # Add episode data to collection
        all_observations.extend(episode_data['observations'])
        all_actions.extend(episode_data['actions'])
        all_rewards.extend(episode_data['rewards'])
        all_terminals.extend(episode_data['terminals'])
        all_next_observations.extend(episode_data['next_observations'])
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        # Progress reporting
        if (ep + 1) % 100 == 0 or (ep + 1) == n_episodes:
            elapsed = time.time() - start_time
            avg_time = elapsed / (ep + 1)
            remaining = avg_time * (n_episodes - ep - 1)
            survival_rate = sum(r > 0 for r in episode_rewards) / len(episode_rewards)
            
            print(f"Episode {ep+1}/{n_episodes}")
            print(f"  Transitions: {total_transitions}")
            print(f"  Avg episode length: {np.mean(episode_lengths):.1f}")
            print(f"  Survival rate: {survival_rate*100:.1f}%")
            print(f"  Elapsed: {elapsed/60:.1f}min, Remaining: ~{remaining/60:.1f}min")
            print()
        
        # Save checkpoint
        if (ep + 1) % save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_ep{ep+1}.pkl'
            save_dataset(
                all_observations, all_actions, all_rewards, 
                all_terminals, all_next_observations,
                checkpoint_path
            )
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    env.close()
    
    # Final statistics
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Data Collection Complete!")
    print(f"{'='*60}")
    print(f"Total episodes: {n_episodes}")
    print(f"Total transitions: {total_transitions}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Survival rate: {sum(r > 0 for r in episode_rewards) / len(episode_rewards) * 100:.1f}%")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Total time: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
    print(f"{'='*60}\n")
    
    return {
        'observations': np.array(all_observations),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'terminals': np.array(all_terminals),
        'next_observations': np.array(all_next_observations),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def save_dataset(observations, actions, rewards, terminals, next_observations, filepath):
    """
    Save dataset in format compatible with d3rlpy
    """
    dataset = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'terminals': np.array(terminals),
        'next_observations': np.array(next_observations)
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved: {filepath}")
    print(f"  Shape: {dataset['observations'].shape}")


def verify_dataset(filepath):
    """
    Load and verify dataset
    """
    print(f"\nVerifying dataset: {filepath}")
    
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset contents:")
    for key, value in dataset.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: length={len(value)}")
    
    # Basic checks
    n_transitions = len(dataset['observations'])
    assert len(dataset['actions']) == n_transitions
    assert len(dataset['rewards']) == n_transitions
    assert len(dataset['terminals']) == n_transitions
    
    print(f"\nDataset verification passed!")
    print(f"Total transitions: {n_transitions}")
    
    return dataset


if __name__ == "__main__":
    # Configuration
    TARGET_TRANSITIONS = 20000
    TRANSITIONS_PER_EPISODE = 9.5  # From our test
    N_EPISODES = int(TARGET_TRANSITIONS / TRANSITIONS_PER_EPISODE)
    
    print(f"Configuration:")
    print(f"  Target transitions: {TARGET_TRANSITIONS}")
    print(f"  Estimated episodes: {N_EPISODES}")
    print(f"  Estimated time: ~1.0 hour")
    
    # Collect data
    dataset = collect_episodes(
        n_episodes=N_EPISODES,
        output_dir='data',
        save_interval=500
    )
    
    # Save final dataset
    final_path = Path('data') / 'offline_dataset.pkl'
    save_dataset(
        dataset['observations'],
        dataset['actions'],
        dataset['rewards'],
        dataset['terminals'],
        dataset['next_observations'],
        final_path
    )
    
    # Verify
    verify_dataset(final_path)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Data collection complete.")
    print(f"Next step: Train BC model with: python src/training/run_bc.py")
    print(f"{'='*60}")