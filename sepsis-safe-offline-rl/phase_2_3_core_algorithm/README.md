# Phase 2 & 3: Core Algorithm Implementation & Training

## Overview

**Duration**: 5-7 weeks
**Objective**: Implement RF-CQL + Filtered-MDP, assemble safety architecture, and train candidate policies.

## Why This Phase Matters

This phase implements the **core innovation** of the research:
1. **Filtered-MDP**: Safety constraints integrated into Bellman backup (NOT post-hoc filtering!)
2. **Adaptive Conservatism**: State-dependent pessimism based on OOD detection
3. **Hierarchical Inference**: L1→L2→L3 safety-aware decision making

**⚠️ Critical Implementation Requirements**:
- Safety masks MUST be integrated into training (not just inference)
- Train-inference isomorphism is essential
- OPE evaluation must be stable (sufficient ESS)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    RF-CQL Agent                           │
├──────────────────────────────────────────────────────────┤
│  Training Time              │    Inference Time           │
├─────────────────────────────┼─────────────────────────────┤
│  1. Filtered-MDP Bellman    │  1. L1 Safety Check        │
│     y = r + γ max Q(s',a')  │     Filter A_safe(s)       │
│         a'∈A_safe(s')       │                             │
│                              │  2. L2 Cognitive Check     │
│  2. Adaptive CQL Loss       │     OOD ≤ δ, U_del ≤ κ     │
│     α(s) = α_0 + γ·OOD(s)   │                             │
│                              │  3. L3 Optimization        │
│  3. Action Masking          │     argmax Q(s,a)          │
│     Apply A_safe during     │     a ∈ filtered set       │
│     action selection        │                             │
│                              │  4. Fallback if needed     │
└─────────────────────────────┴─────────────────────────────┘
```

---

## Phase 2: Algorithm Implementation

### Task 2.1: Adaptive Conservatism

**Goal**: Make CQL more conservative in uncertain/OOD regions.

**Standard CQL Loss**:
```
L_CQL = E[max_a Q(s,a) - Q(s,a_data)] + L_TD
```

**Modified: State-Dependent α**:
```
α(s) = α_0 + γ · OOD_score(s)
L_CQL = E[α(s) · (max_a Q(s,a) - Q(s,a_data))] + L_TD
```

**Implementation Requirements**:
1. Modify CQL to accept state-dependent α
2. Compute OOD_score(s) using Phase 1's OOD detector
3. Validate that high OOD leads to higher conservatism

**File**: `rf_cql.py`

**Test**: Unit test that verifies α increases with OOD score.

---

### Task 2.2: Filtered-MDP Training (CRITICAL!)

**The Key Innovation**: Integrate safety masks into Bellman backup.

**❌ WRONG (Post-hoc filtering)**:
```python
# Train without constraints
y = r + gamma * torch.max(Q_target(next_state, all_actions))

# Filter only at inference
def act(state):
    safe_actions = get_safe_actions(state)
    return argmax(Q(state, safe_actions))  # First time seeing mask!
```

**✅ CORRECT (Filtered-MDP)**:
```python
# Train WITH constraints
def compute_target(state, action, reward, next_state):
    safe_actions = get_safe_actions(next_state)  # L1 mask
    safe_q_values = Q_target(next_state, safe_actions)
    y = reward + gamma * torch.max(safe_q_values)  # Max over SAFE actions only
    return y

# Inference uses same logic
def act(state):
    safe_actions = get_safe_actions(state)
    return argmax(Q(state, safe_actions))  # Consistent with training!
```

**Why This Matters**:
- Post-hoc filtering causes train-test mismatch
- Agent never learns that some actions are forbidden
- At inference, filtering creates OOD states the agent has never seen

**Implementation Steps**:
1. Implement `get_safe_actions(state)` using L1 semantic safety layer
2. Modify Bellman backup to apply mask before max operation
3. Apply same mask during training action selection (e.g., ε-greedy)
4. Unit test: Verify that forbidden actions never appear in Bellman target

**File**: `filtered_mdp.py`

**Validation**: Train two agents (with/without filtering), compare Q-values on states where some actions are masked.

---

### Task 2.3: Inference-Time Hierarchical Arbiter

**Goal**: Implement L1→L2→L3 decision pipeline.

**Architecture**:
```python
def select_action(state, Q_network, safety_layers):
    # L1: Semantic Safety (Hard Constraints)
    candidate_actions = safety_layers.l1_filter(state)
    if len(candidate_actions) == 0:
        return fallback_policy(state), "L1_BLOCKED"

    # L2: Cognitive Safety (Uncertainty/OOD)
    ood_score = safety_layers.ood_detector(state)
    uncertainty = safety_layers.u_del(state, candidate_actions)

    safe_actions = [a for a in candidate_actions
                    if ood_score <= delta and uncertainty[a] <= kappa]

    if len(safe_actions) == 0:
        return fallback_policy(state), "L2_BLOCKED"

    # L3: RL Optimization
    q_values = Q_network(state, safe_actions)
    best_action = safe_actions[argmax(q_values)]

    return best_action, "L3_OPTIMIZED"
```

**Fallback Policy Options**:
1. **Behavioral nearest-neighbor**: Find most similar state in training data, use clinician action
2. **Conservative BC**: Use high-quality BC policy trained in Phase 0
3. **Human-in-the-loop**: Flag for clinician decision

**Logging Requirements**:
- Log every intervention (L1/L2 blocks, L3 successes)
- Record fallback invocations
- Track intervention rates over episodes

**File**: `inference_arbiter.py`

**Test**: Create test states that should trigger each layer, verify correct routing.

---

## Phase 3: Training & Model Selection

### Task 3.1: Training Protocol

**Hyperparameter Search Space**:
```yaml
alpha_0: [0.1, 0.5, 1.0, 2.0]      # Base CQL conservatism
gamma_adapt: [0.5, 1.0, 2.0]       # OOD scaling factor
delta: [0.05, 0.1, 0.2]            # OOD threshold
kappa: [0.1, 0.3, 0.5]             # U_del threshold
```

**Training Procedure**:
1. **Use validation set** for hyperparameter tuning (NOT test set!)
2. Train with **multiple random seeds** (≥ 5)
3. Use **OPE (FQE primary)** for early stopping
   - Monitor ESS (effective sample size)
   - If ESS < 50, OPE estimates are unreliable
   - Track weight variance in WIS

**Early Stopping Criterion**:
- FQE value on validation set (not training!)
- Stop if no improvement for N epochs
- Prevent overfitting to training distribution

**File**: `train.py`

**Monitoring Dashboard**:
- Training loss curves
- FQE validation scores
- ESS over time
- L1/L2 intervention rates

---

### Task 3.2: Confounding Pessimization & Model Selection

**Goal**: Select models that are robust to potential confounding.

**Pessimistic Q-Value (for model selection ONLY)**:
```python
def pessimistic_q(state, action, Q_network, u_del_network, beta):
    q_hat = Q_network(state, action)
    uncertainty = u_del_network(state, action)
    q_tilde = q_hat - beta * uncertainty
    return q_tilde
```

**⚠️ IMPORTANT**: Use pessimistic Q ONLY for model selection, NOT for training gradients!

**Model Selection Procedure**:
1. Train multiple candidate policies (different hyperparameters)
2. Evaluate each on validation set using:
   - System B metrics (primary)
   - Pessimistic Q-values
   - Safety metrics (L1/L2 violation rates)
   - Interpretability proxies
3. Construct **Pareto frontier** over (Performance, Safety, Interpretability)
4. Select non-dominated policies for Phase 4 evaluation

**Pareto Dimensions**:
- **Performance**: FQE value on validation set
- **Safety**: Weighted sum of L1 violations (hard) + L2 interventions (soft)
- **Interpretability**: LEG/SHAP stability score

**File**: `model_selection.py`

**Output**: Set of 3-5 Pareto-optimal policies for comprehensive evaluation in Phase 4.

---

## Deliverables

- [ ] RF-CQL implementation (`rf_cql.py`)
- [ ] Filtered-MDP implementation (`filtered_mdp.py`)
- [ ] Inference arbiter (`inference_arbiter.py`)
- [ ] Training script (`train.py`)
- [ ] Model selection script (`model_selection.py`)
- [ ] Trained candidate policies (saved models)
- [ ] Training logs and hyperparameter records

---

## Checkpoint 3

Before proceeding to Phase 4:

### Critical Validation Checks

1. **Is Filtered-MDP correctly implemented?**
   - [ ] Safety mask integrated into Bellman backup
   - [ ] Mask applied during training action selection
   - [ ] Unit tests pass
   - [ ] Visual inspection of Bellman targets confirms masking

2. **Are training and inference isomorphic?**
   - [ ] Same masking logic used in both
   - [ ] No "first-time" mask application at inference
   - [ ] Logged intervention rates are non-zero during training

3. **Is OPE evaluation stable?**
   - [ ] ESS > 50 for all candidates
   - [ ] WIS weight variance < 10
   - [ ] FQE and WIS rankings roughly agree

4. **Are candidate policies diverse?**
   - [ ] Cover different points on Pareto frontier
   - [ ] Hyperparameters span search space

### Decision

- [ ] Proceed to Phase 4 (Main Evaluation)
- [ ] Debug and re-train (specify issues): _______

---

## File Structure

```
phase_2_3_core_algorithm/
├── README.md                          # This file
├── rf_cql.py                          # Adaptive CQL implementation
├── filtered_mdp.py                    # Filtered-MDP Bellman backup
├── inference_arbiter.py               # L1→L2→L3 decision logic
├── train.py                           # Training script
├── model_selection.py                 # Pareto frontier construction
├── config/
│   └── hyperparameters.yaml           # Search space definition
├── models/                            # Saved checkpoints
│   ├── candidate_1/
│   ├── candidate_2/
│   └── ...
├── logs/                              # Training logs
│   └── tensorboard/
├── results/
│   ├── pareto_frontier.png
│   ├── ope_scores.csv
│   └── intervention_rates.csv
└── tests/
    ├── test_filtered_mdp.py
    ├── test_arbiter.py
    └── test_adaptive_cql.py
```

---

## Key Metrics

| Metric | Target | Status |
|--------|--------|--------|
| ESS (effective sample size) | > 50 | - |
| WIS weight variance | < 10 | - |
| L1 violation rate (training) | ≈ 0 | - |
| L2 intervention rate | 10-30% | - |
| Pareto frontier size | 3-5 policies | - |
| FQE validation score | > Baseline BC | - |

---

## Common Pitfalls

1. **Post-hoc filtering only**: Train without masks, filter at inference → train-test mismatch!
2. **Ignoring ESS**: Low ESS means OPE is unreliable, can't trust model selection
3. **Training on test set**: Use validation set for hyperparameter tuning!
4. **Conflicting mechanisms**: Don't combine CQL with CMDP Lagrangian (they conflict)
5. **Forgetting to log interventions**: Need this data for Phase 4 evaluation

---

## Implementation Tips

### Debugging Filtered-MDP

1. **Verify masks are applied**:
   ```python
   # Add assertions
   assert all(a in safe_actions for a in selected_training_actions)
   ```

2. **Visualize Q-values**:
   - Plot Q(s,a) for masked vs unmasked actions
   - Masked actions should have lower Q-values (pessimism)

3. **Check for Q-value collapse**:
   - If all Q-values → same value, conservatism is too high
   - Reduce α_0

### Speeding Up Training

1. **Use smaller replay buffer** for hyperparameter search
2. **Parallelize** across hyperparameters (not critical path)
3. **Early termination** if ESS drops too low

### OPE Best Practices

1. **Always report ESS** alongside value estimates
2. **Use FQE as primary**, WIS/DR as sanity checks
3. **If methods disagree wildly**, data quality issue (revisit Phase -1)

---

## References

- Conservative Q-Learning: Kumar et al. (2020). "Conservative Q-Learning for Offline RL"
- Filtered-MDP: This work (novel contribution)
- OPE Survey: Voloshin et al. (2021). "Empirical Study of Off-Policy Evaluation"
- Pareto Multi-Objective Optimization: Deb et al. (2002). "NSGA-II"

---

**Status**: 🔵 Not Started
**Last Updated**: 2025-11-11
