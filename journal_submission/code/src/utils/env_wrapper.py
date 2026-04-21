"""
Environment Compatibility Wrapper

Handles compatibility between gym-sepsis (legacy gym 0.21) and
modern gymnasium/Stable-Baselines3 requirements.
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gym')
import numpy as np

try:
    import gymnasium as gym_new
    USING_GYMNASIUM = True
except ImportError:
    import gym as gym_new
    USING_GYMNASIUM = False

import gym as old_gym  # gym-sepsis uses legacy gym
import gym_sepsis


def make_sepsis_env():
    """
    Create Gym-Sepsis environment with automatic gym/gymnasium compatibility handling

    Returns:
        env: Sepsis environment instance
    """
    # gym-sepsis is registered in legacy gym, version v0
    env = old_gym.make('sepsis-v0')

    print(f"✅ Sepsis environment created (sepsis-v0)")
    print(f"   State space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n} actions")

    return env


def get_feature_names():
    """
    Return names of the 46 features in order
    Source: Project documentation
    """
    return [
        'albumin', 'anion_gap', 'band_neutrophils', 'bicarbonate', 'bilirubin',
        'bun', 'chloride', 'creatinine', 'dbp', 'glucose_1', 'glucose_2',
        'heart_rate', 'hematocrit', 'hemoglobin', 'inr', 'lactate',
        'map', 'paco2', 'platelet_count', 'potassium', 'pt', 'ptt',
        'respiratory_rate', 'sodium', 'spo2', 'sbp', 'temp_c', 'wbc',
        'age', 'gender', 'race_white', 'race_black', 'race_hispanic', 'race_other',
        'height', 'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa',
        'qsofa_sbp', 'qsofa_gcs', 'qsofa_rr', 'elixhauser', 'blood_culture_positive'
    ]


def print_state_info(state):
    """
    Print key clinical features from state (for debugging)

    Args:
        state: 46-dimensional state vector
    """
    feature_names = get_feature_names()

    # Key features and their indices
    key_features = {
        'lactate': 15,
        'sbp': 25,
        'heart_rate': 11,
        'map': 16,
        'sofa': 37,
        'wbc': 27,
        'respiratory_rate': 22
    }

    print("\n🩺 Key Clinical Features:")
    for name, idx in key_features.items():
        value = state[idx] if idx < len(state) else 'N/A'
        print(f"   {name:20s}: {value:.2f}" if isinstance(value, (int, float)) else f"   {name:20s}: {value}")


def test_environment():
    """
    Test whether environment works correctly
    """
    env = make_sepsis_env()

    # Run a simple episode
    # Handle different gym version return values
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result

    print(f"\n📊 Initial observation:")
    print(f"   Shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
    print(f"   Expected: (46,)")

    # Print key features of initial state
    print_state_info(obs)

    done = False
    step_count = 0
    total_reward = 0

    print("\n🏃 Running test episode...")
    while not done and step_count < 100:  # Limit maximum steps
        action = env.action_space.sample()  # Random action

        # Handle different gym version step return values
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        total_reward += reward
        step_count += 1

        # Print progress every 10 steps
        if step_count % 10 == 0:
            print(f"   Step {step_count}: reward={reward}, done={done}")

    print(f"\n✅ Test episode completed:")
    print(f"   Steps: {step_count}")
    print(f"   Total reward: {total_reward}")
    print(f"   Outcome: {'Survived ✅' if total_reward > 0 else 'Died ❌'}")

    # Print final state
    print_state_info(obs)

    env.close()
    return True


if __name__ == "__main__":
    print("🧪 Testing Gym-Sepsis environment...\n")
    test_environment()
