"""
Evaluation metrics for the Sepsis Safe Offline RL project.

This module implements:
- System A (training) reward computation
- System B (evaluation) clinical metrics
- Safety metrics
- Interpretability metrics
- Fairness metrics
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


# =============================================================================
# System A: Training Rewards
# =============================================================================

def compute_r_phys(
    state: Dict[str, float],
    next_state: Dict[str, float],
    config: Dict
) -> float:
    """
    Compute physiological reward component.

    Args:
        state: Current state
        next_state: Next state
        config: Reward configuration

    Returns:
        Physiological reward value
    """
    reward = 0.0

    for target in config.get("targets", []):
        feature = target["feature"]
        weight = target["weight"]

        if target.get("target_value") is not None:
            # Target a specific value (e.g., MAP >= 65)
            target_value = target["target_value"]
            current_value = next_state.get(feature, 0)

            if target["shaping"] == "sigmoid":
                scale = target["shaping_params"]["scale"]
                reward += weight * _sigmoid((current_value - target_value) / scale)
            else:
                # Linear distance to target
                reward += weight * max(0, current_value - target_value)

        elif target.get("target_direction") == "decrease":
            # Reward for decreasing value (e.g., lactate, SOFA)
            delta = state.get(feature, 0) - next_state.get(feature, 0)
            reward += weight * delta

    return reward


def compute_smoothness_penalty(
    action: int,
    prev_action: int,
    action_space_dims: Tuple[int, int]
) -> float:
    """
    Compute smoothness penalty (L1 norm of action change).

    Args:
        action: Current action (flattened index)
        prev_action: Previous action
        action_space_dims: (n_vasopressor_levels, n_fluid_levels)

    Returns:
        Smoothness penalty (positive value to be subtracted)
    """
    n_vaso, n_fluid = action_space_dims

    # Convert flat action to 2D
    vaso_curr, fluid_curr = action // n_fluid, action % n_fluid
    vaso_prev, fluid_prev = prev_action // n_fluid, prev_action % n_fluid

    # L1 distance
    penalty = abs(vaso_curr - vaso_prev) + abs(fluid_curr - fluid_prev)

    return float(penalty)


def compute_risk_penalty(
    state: Dict[str, float],
    action: int,
    config: Dict
) -> float:
    """
    Compute risk penalty for high-risk states/actions.

    Args:
        state: Current state
        action: Current action
        config: Risk penalty configuration

    Returns:
        Risk penalty (positive value to be subtracted)
    """
    penalty = 0.0

    for condition in config.get("high_risk_conditions", []):
        # Evaluate condition (simplified - use proper parser in production)
        if _evaluate_condition(condition["condition"], state, action):
            penalty += condition["penalty"]

    return penalty


def compute_system_a_reward(
    state: Dict[str, float],
    action: int,
    next_state: Dict[str, float],
    prev_action: int,
    config: Dict
) -> float:
    """
    Compute full System A (training) reward.

    Args:
        state: Current state
        action: Current action
        next_state: Next state
        prev_action: Previous action
        config: Full reward configuration

    Returns:
        Total reward value
    """
    # R_phys
    r_phys = compute_r_phys(state, next_state, config["r_phys"])

    # Smoothness penalty
    smoothness = compute_smoothness_penalty(
        action, prev_action, (5, 5)  # 5x5 action space
    )

    # Risk penalty
    risk = compute_risk_penalty(state, action, config["risk"])

    # Weights
    w1 = config.get("w1", 1.0)
    w2 = config.get("w2", 0.1)
    w3 = config.get("w3", 0.5)

    # Total reward
    total_reward = w1 * r_phys - w2 * smoothness - w3 * risk

    return total_reward


# =============================================================================
# System B: Evaluation Metrics
# =============================================================================

def compute_survival_rate(outcomes: np.ndarray, horizon_days: int = 90) -> float:
    """
    Compute survival rate at specified horizon.

    Args:
        outcomes: Binary array (1 = survived, 0 = died)
        horizon_days: Survival horizon (default: 90 days)

    Returns:
        Survival rate [0, 1]
    """
    return float(np.mean(outcomes))


def compute_icu_los(lengths: np.ndarray) -> Dict[str, float]:
    """
    Compute ICU length of stay statistics.

    Args:
        lengths: Array of ICU stay lengths (in days)

    Returns:
        Dictionary with mean, median, std
    """
    return {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
    }


def compute_vasopressor_exposure(
    trajectories: List[List[Dict]],
) -> Dict[str, float]:
    """
    Compute vasopressor exposure metrics.

    Args:
        trajectories: List of trajectories (each is list of state-action dicts)

    Returns:
        Dictionary with total dose, duration metrics
    """
    total_doses = []
    durations = []

    for traj in trajectories:
        total_dose = sum(step["vasopressor_dose"] for step in traj)
        duration = sum(1 for step in traj if step["vasopressor_dose"] > 0)
        total_doses.append(total_dose)
        durations.append(duration)

    return {
        "mean_total_dose": float(np.mean(total_doses)),
        "mean_duration": float(np.mean(durations)),
    }


# =============================================================================
# Safety Metrics
# =============================================================================

def compute_hard_violation_rate(
    trajectories: List[List[Dict]],
    safety_rules: List[str]
) -> float:
    """
    Compute hard constraint violation rate.

    Args:
        trajectories: List of trajectories
        safety_rules: List of safety rules to check

    Returns:
        Violation rate [0, 1]
    """
    n_violations = 0
    n_total = 0

    for traj in trajectories:
        for step in traj:
            n_total += 1
            for rule in safety_rules:
                if _evaluate_condition(rule, step["state"], step["action"]):
                    n_violations += 1
                    break  # Count at most one violation per step

    return n_violations / n_total if n_total > 0 else 0.0


def compute_intervention_rate(
    intervention_log: List[Dict],
    layer: str
) -> float:
    """
    Compute safety layer intervention rate.

    Args:
        intervention_log: List of logged interventions
        layer: "L1" or "L2"

    Returns:
        Intervention rate [0, 1]
    """
    n_interventions = sum(
        1 for log in intervention_log if log["layer"] == layer
    )
    n_total = len(intervention_log)

    return n_interventions / n_total if n_total > 0 else 0.0


# =============================================================================
# Statistical Utilities
# =============================================================================

def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval.

    Args:
        successes: Number of successes
        total: Total trials
        confidence: Confidence level (default: 0.95)

    Returns:
        (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 0.0)

    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

    return (center - margin, center + margin)


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cohen's d
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    pooled_std = np.sqrt(
        ((len(group1) - 1) * np.var(group1, ddof=1) +
         (len(group2) - 1) * np.var(group2, ddof=1)) /
        (len(group1) + len(group2) - 2)
    )

    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0


def benjamini_hochberg_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values
        alpha: Significance level

    Returns:
        Boolean array (True = reject null hypothesis)
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Compute critical values
    critical_values = (np.arange(1, n + 1) / n) * alpha

    # Find largest i where p_i <= critical_value_i
    rejections = sorted_p_values <= critical_values
    if np.any(rejections):
        max_i = np.where(rejections)[0][-1]
        # Reject all hypotheses up to max_i
        reject = np.zeros(n, dtype=bool)
        reject[sorted_indices[:max_i + 1]] = True
    else:
        reject = np.zeros(n, dtype=bool)

    return reject


# =============================================================================
# Helper Functions
# =============================================================================

def _sigmoid(x: float) -> float:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def _evaluate_condition(
    condition: str,
    state: Dict[str, float],
    action: Optional[int] = None
) -> bool:
    """
    Evaluate a simple condition string.

    Args:
        condition: Condition string (e.g., "MAP < 55")
        state: Current state
        action: Current action (optional)

    Returns:
        True if condition is met

    Note:
        This is a simplified implementation.
        Production code should use a proper parser.
    """
    # TODO: Implement proper condition parser
    # For now, return False as placeholder
    return False
