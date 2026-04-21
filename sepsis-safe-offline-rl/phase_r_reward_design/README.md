# Phase R: Dual Reward System Design

## Overview

**Duration**: 2-3 weeks
**Objective**: Design training reward (System A) and evaluation metrics (System B), ensuring consistency.

## Why This Phase Matters

**The Problem**: Sparse survival rewards don't provide enough learning signal for RL.

**The Solution**: Use a dense, shaped reward (System A) for training, but evaluate with clinical metrics (System B).

**The Risk**: If A and B are misaligned, the agent will optimize the wrong objective!

**⚠️ Critical Success Factor**: Systems A and B must pass consistency validation.

---

## Task Architecture

```
┌─────────────────────────────────────────────────┐
│                 Reward Design                    │
├──────────────────┬──────────────────────────────┤
│    System A      │         System B              │
│   (Training)     │      (Evaluation)             │
├──────────────────┼──────────────────────────────┤
│ Dense, shaped    │  Sparse, clinical             │
│ Drives gradients │  Final judgment               │
│ Needs tuning     │  Ground truth                 │
└──────────────────┴──────────────────────────────┘
```

---

## Task R.1: System B (Evaluation) Metrics

**Purpose**: Define what we actually care about clinically.

**Metrics**:

### Terminal Metrics
- **90-day survival rate** (primary outcome)
- **ICU length of stay** (process metric)
- **28-day survival rate** (secondary outcome)
- **Hospital length of stay**

### Process Safety Indicators
- **Vasopressor exposure** (total dose, duration)
- **Fluid overload** (cumulative fluid balance)
- **AKI proxies** (creatinine elevation, urine output)
- **Hypotension episodes** (MAP < 65 for > 1 hour)

**Implementation**: `system_b_evaluation.py`

**Key Principle**: System B does NOT participate in gradient updates. It's purely for evaluation.

---

## Task R.2: System A (Training) Reward Shaping

**Purpose**: Provide dense learning signal that aligns with System B.

### Reward Function Structure

```
r_t = w1 * R_phys(s_t, s_{t+1})     [Physiological improvement]
    - w2 * ||a_t - a_{t-1}||_1       [Smoothness penalty]
    - w3 * 1[high_risk_state]        [Risk penalty]
```

### Component 1: R_phys (Physiological Reward)

**Target physiological states**:
- MAP ≥ 65 mmHg (sustained)
- Lactate trending downward
- SOFA score not worsening
- SpO2 > 92%

**Implementation strategy**:
- Use **continuous shaping** (no hard thresholds)
- Example: `R_MAP = sigmoid((MAP - 65) / 5)` instead of binary reward
- Weight by distance from target

**Why continuous?**: Avoids chattering near thresholds and provides smooth gradients.

### Component 2: Smoothness Penalty

**Rationale**: Frequent drastic changes in treatment are:
- Clinically undesirable
- May indicate policy instability
- Increase implementation burden

**Implementation**: L1 norm of action difference
- Penalize both vasopressor and fluid changes
- Scale by clinical significance (e.g., 1 mcg/kg/min change vs 250mL fluid)

### Component 3: Risk Penalty

**Immediate penalties for**:
- Severe hypotension (MAP < 55)
- Extreme dosages (e.g., norepinephrine > 0.5 mcg/kg/min)
- Contraindicated combinations

**Why immediate?**: Prevents exploration into dangerous regions.

### Component 4: Saturation Suppression

**Problem**: Terminal/irreversible states can dominate optimization.

**Solution**: Downweight rewards for states predicted as "terminal"
- Train a simple "futility" classifier on historical data
- For states with P(death|state) > 0.9, multiply reward by (1 - P(death|state))

**Implementation**: `system_a_training.py`

---

## Task R.3: Weight Calibration & Consistency Check

**Purpose**: Ensure System A actually aligns with System B.

### Step 1: Weight Search

**Search space**:
- w1 (phys reward): [0.5, 2.0]
- w2 (smoothness): [0.01, 0.2]
- w3 (risk penalty): [0.1, 1.0]

**Method**: Grid search or Bayesian optimization on **validation set**

**Evaluation**: Train small policies with different weight combinations, evaluate with both System A and B.

### Step 2: Consistency Validation (CRITICAL!)

**Metrics**:
1. **Spearman correlation** between cumulative System A reward and System B survival rate
   - Target: ρ > 0.7
2. **Pearson correlation** for linear relationship
   - Target: r > 0.6
3. **Monotonicity check**: Policies ranked by A should align with ranking by B

**Implementation**:
```python
def consistency_check(policies, val_data):
    system_a_scores = [evaluate_system_a(p, val_data) for p in policies]
    system_b_scores = [evaluate_system_b(p, val_data) for p in policies]

    spearman = scipy.stats.spearmanr(system_a_scores, system_b_scores)
    pearson = scipy.stats.pearsonr(system_a_scores, system_b_scores)

    return spearman, pearson
```

### Step 3: Callback Mechanism

**If consistency check fails** (ρ < 0.7 or r < 0.6):
1. **Document the issue**: Which policies? What's the gap?
2. **Analyze discrepancies**: Are there specific states where A and B disagree?
3. **Adjust**:
   - Modify R_phys to better capture clinical outcomes
   - Re-tune weights
   - Consider adding/removing reward components
4. **Re-run consistency check**

**Do NOT proceed to Phase 2 until consistency is validated!**

---

## Deliverables

- [ ] System A implementation (`system_a_training.py`)
- [ ] System B implementation (`system_b_evaluation.py`)
- [ ] Optimal weight configuration (`config/reward_weights.yaml`)
- [ ] Consistency validation report (`results/consistency_report.md`)

---

## Checkpoint 2

Before proceeding to Phase 0&1:

1. **Do System A and B pass consistency check?**
   - Spearman ρ > 0.7: [ ]
   - Pearson r > 0.6: [ ]

2. **If not, callback documented?**
   - Issue description: _______
   - Corrective actions: _______
   - Re-validation results: _______

3. **Decision**:
   - [ ] Proceed to Phase 0&1
   - [ ] Revise reward design and re-validate

---

## File Structure

```
phase_r_reward_design/
├── README.md                          # This file
├── system_a_training.py               # Training reward implementation
├── system_b_evaluation.py             # Evaluation metrics implementation
├── weight_tuning.py                   # Weight search script
├── consistency_check.py               # Validation script
├── config/
│   └── reward_weights.yaml            # Optimal weights
├── results/
│   ├── consistency_report.md
│   ├── weight_search_results.csv
│   └── correlation_plots.png
└── notebooks/
    ├── 01_reward_exploration.ipynb
    └── 02_consistency_analysis.ipynb
```

---

## Key Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Spearman correlation (A vs B) | > 0.7 | - |
| Pearson correlation (A vs B) | > 0.6 | - |
| Rank disagreement rate | < 20% | - |
| Optimal w1 (phys) | - | - |
| Optimal w2 (smooth) | - | - |
| Optimal w3 (risk) | - | - |

---

## Common Pitfalls

1. **Using sparse terminal survival reward for training**: Doesn't work! RL needs dense signals.
2. **Ignoring consistency check**: Leads to policies that score high on A but fail on B.
3. **Over-engineering reward**: Keep it simple; complexity makes tuning harder.
4. **Hard thresholds in R_phys**: Causes chattering and unstable gradients.

---

## References

- Reward shaping: Ng et al. (1999). "Policy Invariance Under Reward Transformations"
- Clinical reward design: Komorowski et al. (2018). "The Artificial Intelligence Clinician"
- Consistency validation: Thomas et al. (2019). "Preventing Undesirable Behavior of Intelligent Machines"

---

**Status**: 🔵 Not Started
**Last Updated**: 2025-11-11
