"""
Reward Functions for Sepsis Treatment RL

Implements three reward function variants:
1. Simple: Terminal reward only (±15)
2. Paper: Continuous SOFA + lactate feedback (Raghu et al. 2017)
3. Hybrid: Small intermediate + large terminal rewards

Feature indices (from sepsis_env.py):
- SOFA score: features[37]
- LACTATE: features[15]
"""

import numpy as np
from typing import Dict, Callable


# Feature indices (matching sepsis_env.py line 24-32)
SOFA_IDX = 37
LACTATE_IDX = 15


def simple_reward(prev_state: np.ndarray,
                  curr_state: np.ndarray,
                  done: bool,
                  outcome_survived: bool) -> float:
    """
    Simple reward function: Terminal reward only

    Intermediate steps: reward = 0
    Terminal step: reward = +15 if survived, -15 if died

    Args:
        prev_state: Previous state (46-dim)
        curr_state: Current state (46-dim)
        done: Whether episode is done
        outcome_survived: Whether patient survived (only valid if done=True)

    Returns:
        reward: Scalar reward value
    """
    if done:
        return 15.0 if outcome_survived else -15.0
    else:
        return 0.0


def paper_reward(prev_state: np.ndarray,
                 curr_state: np.ndarray,
                 done: bool,
                 outcome_survived: bool,
                 C0: float = -0.025,
                 C1: float = -0.125,
                 C2: float = -2.0) -> float:
    """
    Paper reward function from Raghu et al. (2017)

    Intermediate steps:
        r = C0·1(SOFA_t == SOFA_{t-1} & SOFA_t > 0)
          + C1·(SOFA_t - SOFA_{t-1})
          + C2·tanh(Lactate_t - Lactate_{t-1})

    Terminal step: reward = +15 if survived, -15 if died

    Rationale:
    - Penalize high SOFA and stagnant SOFA
    - Penalize increases in SOFA and lactate
    - Reward decreases in SOFA and lactate

    Args:
        prev_state: Previous state (46-dim)
        curr_state: Current state (46-dim)
        done: Whether episode is done
        outcome_survived: Whether patient survived
        C0, C1, C2: Reward coefficients (defaults from paper)

    Returns:
        reward: Scalar reward value
    """
    if done:
        # Terminal reward
        return 15.0 if outcome_survived else -15.0

    # Extract SOFA and lactate from states
    sofa_prev = prev_state[SOFA_IDX]
    sofa_curr = curr_state[SOFA_IDX]
    lactate_prev = prev_state[LACTATE_IDX]
    lactate_curr = curr_state[LACTATE_IDX]

    # Initialize reward
    reward = 0.0

    # C0 term: Penalize stagnant SOFA when SOFA > 0
    if sofa_curr > 0 and np.isclose(sofa_curr, sofa_prev, atol=1e-3):
        reward += C0

    # C1 term: Penalize SOFA increases, reward decreases
    sofa_change = sofa_curr - sofa_prev
    reward += C1 * sofa_change

    # C2 term: Penalize lactate increases, reward decreases (with tanh saturation)
    lactate_change = lactate_curr - lactate_prev
    reward += C2 * np.tanh(lactate_change)

    return float(reward)


def hybrid_reward(prev_state: np.ndarray,
                  curr_state: np.ndarray,
                  done: bool,
                  outcome_survived: bool,
                  intermediate_scale: float = 0.1) -> float:
    """
    Hybrid reward function: Intermediate guidance + strong terminal signal

    Combines:
    - Scaled paper reward for intermediate steps
    - Full terminal reward (±15)

    This balances learning signal density with outcome importance.

    Args:
        prev_state: Previous state (46-dim)
        curr_state: Current state (46-dim)
        done: Whether episode is done
        outcome_survived: Whether patient survived
        intermediate_scale: Scaling factor for intermediate rewards

    Returns:
        reward: Scalar reward value
    """
    if done:
        # Full terminal reward
        return 15.0 if outcome_survived else -15.0
    else:
        # Scaled intermediate reward based on paper formula
        paper_r = paper_reward(prev_state, curr_state, done=False,
                              outcome_survived=False)
        return intermediate_scale * paper_r


# Registry of reward functions
REWARD_FUNCTIONS: Dict[str, Callable] = {
    'simple': simple_reward,
    'paper': paper_reward,
    'hybrid': hybrid_reward,
}


def get_reward_function(name: str) -> Callable:
    """
    Get reward function by name

    Args:
        name: One of 'simple', 'paper', 'hybrid'

    Returns:
        reward_function: Callable reward function

    Raises:
        ValueError: If name is not recognized
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {name}. "
                        f"Available: {list(REWARD_FUNCTIONS.keys())}")
    return REWARD_FUNCTIONS[name]


# For testing
if __name__ == "__main__":
    print("Testing reward functions...")

    # Create dummy states
    prev_state = np.random.randn(46)
    curr_state = np.random.randn(46)

    # Set specific values for SOFA and lactate
    prev_state[SOFA_IDX] = 10.0
    curr_state[SOFA_IDX] = 8.0  # Improvement
    prev_state[LACTATE_IDX] = 3.0
    curr_state[LACTATE_IDX] = 2.5  # Improvement

    print("\nTest case: SOFA 10->8, Lactate 3.0->2.5 (improvement)")
    print(f"Simple (intermediate): {simple_reward(prev_state, curr_state, False, False):.3f}")
    print(f"Paper (intermediate):  {paper_reward(prev_state, curr_state, False, False):.3f}")
    print(f"Hybrid (intermediate): {hybrid_reward(prev_state, curr_state, False, False):.3f}")

    print("\nTest case: Terminal (survived)")
    print(f"Simple (terminal): {simple_reward(prev_state, curr_state, True, True):.3f}")
    print(f"Paper (terminal):  {paper_reward(prev_state, curr_state, True, True):.3f}")
    print(f"Hybrid (terminal): {hybrid_reward(prev_state, curr_state, True, True):.3f}")

    print("\n[OK] Reward functions module created successfully!")