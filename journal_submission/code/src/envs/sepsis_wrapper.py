"""
Gym-Sepsis Environment Wrapper with Custom Reward Functions

Wraps the original gym-sepsis environment to support different reward functions
while maintaining full compatibility with gym and RL libraries (d3rlpy, SB3).
"""

import gym
import numpy as np
from typing import Optional, Tuple, Dict, Any
import sys
import os
from pathlib import Path

# Add gym-sepsis to path
# Navigate from code/src/envs/ up to repository root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'gym-sepsis'))

# Import gym-sepsis environment
try:
    from gym_sepsis.envs.sepsis_env import SepsisEnv
except ImportError:
    raise ImportError("Cannot import gym-sepsis. Make sure it's in the project directory.")

# Import our reward functions
try:
    from src.envs.reward_functions import get_reward_function
except ImportError:
    # If running as script
    from reward_functions import get_reward_function


class SepsisEnvWrapper(gym.Wrapper):
    """
    Wrapper for Gym-Sepsis environment with customizable reward functions

    Features:
    - Switch between 'simple', 'paper', 'hybrid' reward functions
    - Track previous state for reward computation
    - Compatible with gym, d3rlpy, and stable-baselines3
    - Preserves all original environment functionality

    Args:
        reward_fn_name: Name of reward function ('simple', 'paper', 'hybrid')
        starting_state: Optional starting state (None for random)
        verbose: Print debug information
    """

    def __init__(self,
                 reward_fn_name: str = 'simple',
                 starting_state: Optional[np.ndarray] = None,
                 verbose: bool = False):

        # Create base environment
        base_env = SepsisEnv(starting_state=starting_state, verbose=verbose)
        super().__init__(base_env)

        # Set reward function
        self.reward_fn_name = reward_fn_name
        self.reward_fn = get_reward_function(reward_fn_name)

        # Track previous state for reward computation
        self.prev_state = None
        self.verbose = verbose

        if self.verbose:
            print(f"SepsisEnvWrapper initialized with reward function: {reward_fn_name}")

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and initialize state tracking"""
        obs, info = self.env.reset(**kwargs)
        self.prev_state = obs.copy()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take environment step with custom reward computation

        Args:
            action: Action to take (integer 0-23)

        Returns:
            observation: Next state (46-dim array)
            reward: Custom reward based on reward function
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Take step in base environment
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Determine if patient survived (from original reward)
        # Original: +15 if survived, -15 if died
        outcome_survived = original_reward > 0 if terminated else False

        # Compute custom reward
        custom_reward = self.reward_fn(
            prev_state=self.prev_state,
            curr_state=obs,
            done=terminated,
            outcome_survived=outcome_survived
        )

        # Update previous state
        self.prev_state = obs.copy()

        if self.verbose and terminated:
            print(f"Episode ended. Original reward: {original_reward}, "
                  f"Custom reward: {custom_reward}, Survived: {outcome_survived}")

        return obs, custom_reward, terminated, truncated, info

    def get_reward_fn_name(self) -> str:
        """Get name of current reward function"""
        return self.reward_fn_name


def make_sepsis_env(reward_fn_name: str = 'simple',
                    starting_state: Optional[np.ndarray] = None,
                    verbose: bool = False) -> SepsisEnvWrapper:
    """
    Factory function to create wrapped Sepsis environment

    Args:
        reward_fn_name: Name of reward function ('simple', 'paper', 'hybrid')
        starting_state: Optional starting state
        verbose: Print debug information

    Returns:
        env: Wrapped Sepsis environment

    Example:
        >>> env = make_sepsis_env(reward_fn_name='paper')
        >>> obs, info = env.reset()
        >>> obs, reward, done, truncated, info = env.step(10)
    """
    return SepsisEnvWrapper(
        reward_fn_name=reward_fn_name,
        starting_state=starting_state,
        verbose=verbose
    )


# For testing
if __name__ == "__main__":
    print("Testing Sepsis Environment Wrapper...")

    # Test all reward functions
    for reward_name in ['simple', 'paper', 'hybrid']:
        print(f"\n{'='*60}")
        print(f"Testing reward function: {reward_name}")
        print('='*60)

        env = make_sepsis_env(reward_fn_name=reward_name, verbose=False)

        # Reset
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")

        # Run one episode
        total_reward = 0
        step_count = 0
        done = False

        while not done and step_count < 20:  # Max 20 steps for testing
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            if reward != 0 or done:  # Print non-zero rewards and terminal
                print(f"  Step {step_count}: action={action}, reward={reward:.3f}, done={done}")

        print(f"Episode finished in {step_count} steps with total reward: {total_reward:.3f}")
        env.close()

    print("\n[OK] Environment wrapper tests passed!")