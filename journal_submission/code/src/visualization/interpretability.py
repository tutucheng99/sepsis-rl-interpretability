"""
Interpretability Analysis for RL Policies in Sepsis Treatment

Provides tools to understand and explain RL policy decisions:
1. Q-value analysis - understand why specific actions are chosen
2. Feature importance - which patient features drive decisions
3. Action agreement - how often RL agrees with clinicians
4. Decision paths - trace decision logic for specific cases
5. Counterfactual analysis - what if different treatments were used
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Callable
import pandas as pd
from collections import defaultdict


# Feature names (from sepsis_env.py)
FEATURE_NAMES = [
    'ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
    'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
    'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
    'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
    'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
    'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
    'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
    'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
    'blood_culture_positive'
]

CRITICAL_FEATURES = ['LACTATE', 'MeanBP', 'SysBP', 'sofa', 'HeartRate', 'SpO2']



def analyze_q_values(model, state: np.ndarray, top_k: int = 5) -> Dict:
    """
    Analyze Q-values for all actions in a given state

    Helps understand:
    - Which actions are considered best/worst
    - How confident the model is
    - Action rankings

    Args:
        model: Trained RL model (DQN, BC, CQL)
        state: Patient state
        top_k: Number of top actions to show

    Returns:
        analysis: Dict with Q-values, rankings, confidence
    """
    # Get Q-values for all actions
    state_batch = state.reshape(1, -1).astype(np.float32)

    q_values = None
    if hasattr(model, "predict_value"):
        q_list = []
        try:
            for action in range(24):
                q_val = model.predict_value(state_batch, np.array([[action]]))
                q_list.append(float(q_val[0]))
            q_values = np.asarray(q_list, dtype=float)
        except NotImplementedError:
            q_values = None  # e.g., Behavior Cloning raises this

    if q_values is None:
        # Fall back to ranking based on the policy's chosen action
        try:
            pred_action = int(model.predict(state_batch)[0])
        except Exception:
            pred_action = None
        if pred_action is None:
            q_values = np.zeros(24, dtype=float)
        else:
            # Highest score for the chosen action, penalize others by distance
            q_values = np.array(
                [0.0 if a == pred_action else -abs(a - pred_action) for a in range(24)],
                dtype=float,
            )

    # Rankings
    action_ranking = np.argsort(-q_values)  # Descending order

    # Decode top actions
    top_actions_decoded = []
    for i in range(min(top_k, len(action_ranking))):
        action = action_ranking[i]
        iv_dose = action // 5
        vp_dose = action % 5
        q_val = q_values[action]
        top_actions_decoded.append(
            {
                "action": int(action),
                "iv_dose": int(iv_dose),
                "vp_dose": int(vp_dose),
                "q_value": float(q_val),
                "rank": i + 1,
            }
        )

    # Confidence (difference between best and second-best)
    if len(q_values) >= 2:
        confidence = float(q_values[action_ranking[0]] - q_values[action_ranking[1]])
    else:
        confidence = 0.0

    return {
        "all_q_values": q_values.tolist(),
        "best_action": int(action_ranking[0]),
        "top_k_actions": top_actions_decoded,
        "confidence": confidence,
        "q_value_std": float(np.std(q_values)),
        "q_value_range": float(np.max(q_values) - np.min(q_values)),
    }


def feature_importance_simple(model, state: np.ndarray,
                               feature_names: List[str] = FEATURE_NAMES) -> pd.DataFrame:
    """
    Simple feature importance using perturbation

    Shows which patient features most influence the policy's decision

    Args:
        model: Trained RL model
        state: Patient state
        feature_names: Names of features

    Returns:
        importance_df: DataFrame with feature importances
    """
    baseline_action = model.predict(state.reshape(1, -1))[0]

    importances = []

    for i, feat_name in enumerate(feature_names):
        # Perturb feature by ±1 std
        perturbed_state = state.copy()
        perturbed_state[i] += 0.5  # Add noise

        perturbed_action = model.predict(perturbed_state.reshape(1, -1))[0]

        # Measure action change
        action_change = abs(int(perturbed_action) - int(baseline_action))

        importances.append({
            'feature': feat_name,
            'importance': action_change,
            'value': float(state[i])
        })

    df = pd.DataFrame(importances)
    df = df.sort_values('importance', ascending=False)

    return df


def compare_with_clinician(rl_policy: Callable,
                           clinician_policy: Callable,
                           env,
                           n_episodes: int = 100) -> Dict:
    """
    Compare RL policy decisions with clinician (heuristic) policy

    Provides insights into:
    - Agreement rate
    - When they disagree (patient characteristics)
    - Outcomes when they disagree

    Args:
        rl_policy: RL policy function
        clinician_policy: Clinician heuristic function
        env: Sepsis environment
        n_episodes: Number of episodes

    Returns:
        comparison: Dict with agreement analysis
    """
    agreements = []
    disagreements = []

    for ep in range(n_episodes):
        obs, info = env.reset()

        episode_agreements = []
        episode_disagreements = []

        done = False
        step = 0

        while not done and step < 50:
            rl_action = rl_policy(obs)
            clinician_action = clinician_policy(obs)

            agree = (rl_action == clinician_action)

            if agree:
                episode_agreements.append(step)
            else:
                # Record disagreement details
                disagreements.append({
                    'episode': ep,
                    'step': step,
                    'rl_action': int(rl_action),
                    'clinician_action': int(clinician_action),
                    'sofa': float(obs[37]),  # SOFA index
                    'lactate': float(obs[15]),  # Lactate index
                    'map_bp': float(obs[16])  # Mean BP index
                })
                episode_disagreements.append(step)

            obs, reward, terminated, truncated, info = env.step(rl_action)
            done = terminated or truncated
            step += 1

        if episode_agreements or episode_disagreements:
            agreement_rate = len(episode_agreements) / (len(episode_agreements) + len(episode_disagreements))
        else:
            agreement_rate = 0.0

        agreements.append(agreement_rate)

    return {
        'overall_agreement_rate': float(np.mean(agreements)),
        'agreement_std': float(np.std(agreements)),
        'n_disagreements': len(disagreements),
        'disagreement_details': disagreements[:20],  # Sample
        'agreement_by_episode': agreements
    }


def explain_single_decision(model, state: np.ndarray,
                            feature_names: List[str] = FEATURE_NAMES,
                            critical_features: List[str] = CRITICAL_FEATURES) -> str:
    """
    Generate human-readable explanation for a single decision

    Args:
        model: Trained model
        state: Patient state
        feature_names: All feature names
        critical_features: Most clinically relevant features

    Returns:
        explanation: Text explanation
    """
    # Get decision
    action = model.predict(state.reshape(1, -1))[0]
    iv_dose = action // 5
    vp_dose = action % 5

    # Get Q-value analysis
    q_analysis = analyze_q_values(model, state, top_k=3)

    # Get critical feature values
    critical_vals = {}
    for feat in critical_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            critical_vals[feat] = state[idx]

    # Build explanation
    explanation = f"""
DECISION EXPLANATION
{'='*60}

Recommended Treatment:
  - IV Fluid Dose: {iv_dose} (0=none, 4=max)
  - Vasopressor Dose: {vp_dose} (0=none, 4=max)
  - Action Code: {action}

Confidence:
  - Decision Confidence: {q_analysis['confidence']:.3f}
  - Q-value Range: {q_analysis['q_value_range']:.3f}

Critical Patient Features:
"""

    for feat, val in critical_vals.items():
        # Interpret values (standardized)
        if val > 1.0:
            status = "HIGH"
        elif val < -1.0:
            status = "LOW"
        else:
            status = "NORMAL"
        explanation += f"  - {feat}: {val:.2f} ({status})\n"

    explanation += "\nAlternative Actions Considered:\n"
    for alt in q_analysis['top_k_actions'][:3]:
        if alt['rank'] > 1:
            explanation += f"  {alt['rank']}. IV={alt['iv_dose']}, VP={alt['vp_dose']} (Q={alt['q_value']:.3f})\n"

    explanation += f"\n{'='*60}\n"

    return explanation


def plot_q_value_landscape(model, state: np.ndarray,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize Q-values across all action space

    Args:
        model: Trained model
        state: Patient state
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    # Get Q-values
    q_analysis = analyze_q_values(model, state, top_k=24)
    q_values = np.array(q_analysis['all_q_values'])

    q_grid = np.full((5, 5), np.nan, dtype=float)
    for action, value in enumerate(q_values):
        iv_idx = min(action // 5, 4)
        vp_idx = action % 5
        q_grid[iv_idx, vp_idx] = value

    if np.isnan(q_grid).any():
        fallback = float(np.nanmean(q_grid)) if np.isnan(q_grid).sum() < q_grid.size else 0.0
        q_grid = np.where(np.isnan(q_grid), fallback, q_grid)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        q_grid,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=q_grid.mean(),
        cbar_kws={'label': 'Q-Value'},
        ax=ax,
        square=True
    )

    # Mark best action
    best_action = q_analysis['best_action']
    best_iv = best_action // 5
    best_vp = best_action % 5
    ax.add_patch(plt.Rectangle((best_vp, best_iv), 1, 1,
                                fill=False, edgecolor='blue', lw=3))

    ax.set_xlabel('Vasopressor Dose', fontsize=12)
    ax.set_ylabel('IV Fluid Dose', fontsize=12)
    ax.set_title('Q-Value Landscape for Current State', fontsize=14)
    ax.set_xticklabels(['0', '1', '2', '3', '4'])
    ax.set_yticklabels(['0', '1', '2', '3', '4'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_feature_importance(importance_df: pd.DataFrame,
                            top_n: int = 10,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance

    Args:
        importance_df: DataFrame from feature_importance_simple
        top_n: Number of top features to show
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    top_features = importance_df.head(top_n)

    colors = ['red' if x in CRITICAL_FEATURES else 'steelblue'
              for x in top_features['feature']]

    ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance (Action Change Magnitude)')
    ax.set_title('Feature Importance for Policy Decision')
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Critical Clinical Feature'),
        Patch(facecolor='steelblue', label='Other Feature')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# For testing
if __name__ == "__main__":
    print("Interpretability module loaded successfully!")
    print("\nAvailable functions:")
    print("  - analyze_q_values(model, state)")
    print("  - feature_importance_simple(model, state)")
    print("  - compare_with_clinician(rl_policy, clinician_policy, env)")
    print("  - explain_single_decision(model, state)")
    print("  - plot_q_value_landscape(model, state)")
    print("  - plot_feature_importance(importance_df)")
