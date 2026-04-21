# Phase -1: Data Layer Causal-Fairness Audit

## Overview

**Duration**: 3-4 weeks
**Objective**: Assess data suitability before training, identify bias and confounding risks, and define RL validity boundaries.

## Why This Phase Matters

This phase is **CRITICAL** for ensuring that:
1. The data supports causal inference for policy learning
2. Hidden confounders are identified and quantified
3. Fairness issues are detected before model training
4. We avoid wasting compute on domains where RL cannot safely learn

**⚠️ This must be completed BEFORE any RL training begins to avoid selection bias.**

---

## Tasks

### Task 1.1: Behavioral Policy Estimation & Overlap Analysis

**Purpose**: Understand clinician behavior and identify where we have sufficient data support.

**Steps**:
1. Fit behavioral policy π_beh(a|s,t) using training data
   - Consider using gradient boosting or neural network classifier
   - Model should output P(action | state, time)
2. Compute propensity scores for all (state, action) pairs
3. Assess overlap/positivity
   - Identify regions where P_beh(a|s) is very low
   - These are areas where RL will struggle
4. Document findings in audit report

**Implementation**: `behavior_policy.py`

**Output**:
- Trained π_beh model (saved to `models/`)
- Propensity score distribution plots
- List of low-overlap regions

---

### Task 1.2: Causal Graph Construction & Confounding Sensitivity

**Purpose**: Make confounding explicit and quantify its potential impact.

**Steps**:
1. **Construct Causal DAG**:
   - Define nodes: S (state), A (action), R (reward), T (transition)
   - Add shadow variables: measurement frequency, missingness indicators
   - Identify potential confounders (e.g., disease severity, frailty)

2. **Sensitivity Analysis**:
   - Implement Rosenbaum bounds or E-value method
   - Parameterize: "How strong would unmeasured confounding need to be to overturn our conclusions?"
   - Compute E-values for key associations

3. **Quantify Uncertainty**:
   - Implement Delphi uncertainty U_del(s,a)
   - Maps confounding risk to numerical uncertainty
   - Will be used for pessimistic evaluation in Phase 3

**Implementation**: `causal_analysis.py`

**Output**:
- Causal DAG diagram (saved as image)
- Sensitivity analysis report
- U_del(s,a) function implementation

---

### Task 1.3: Fairness Audit & Stitchability Gate

**Purpose**: Ensure the learned policy will be fair and that data is "stitchable" (combinable across behavioral patterns).

**Steps**:
1. **Representativeness & Bias Audit**:
   - Stratify data by: age groups, gender, baseline SOFA score
   - Report sample sizes and coverage per stratum
   - Compare behavioral policies across groups
   - Flag any groups with insufficient representation

2. **Stitchability Assessment**:
   - Cluster trajectories by behavioral pattern
   - Check state overlap between clusters
   - Verify myopic value consistency across clusters
   - If stitchability is low, RL may produce unstable policies

3. **Usability Gate Decision**:
   - For low-overlap domains: label as "RL-restricted zones"
   - For these zones, use high-quality BC baseline instead
   - Document boundaries between RL-viable and BC-only domains

**Implementation**: `fairness_audit.py`

**Output**:
- Fairness audit report with stratified statistics
- Stitchability analysis report
- RL/BC domain boundary specification

---

## Deliverables

- [ ] Causal-fairness audit report (PDF/Markdown)
- [ ] Trained π_beh model
- [ ] U_del(s,a) implementation
- [ ] RL/BC domain boundaries

---

## Checkpoint 1

Before proceeding to Phase R, answer:

1. **Has data audit identified major biases or confounding risks?**
   - If YES: Document mitigation strategies or consider additional data collection

2. **Do RL-viable domains sufficiently cover target population?**
   - If NO: Consider expanding BC usage or revising project scope

3. **Decision**:
   - [ ] Proceed to Phase R
   - [ ] Revise data strategy
   - [ ] Abort project (data unsuitable for safe RL)

---

## File Structure

```
phase_-1_data_audit/
├── README.md                          # This file
├── behavior_policy.py                 # Task 1.1 implementation
├── causal_analysis.py                 # Task 1.2 implementation
├── fairness_audit.py                  # Task 1.3 implementation
├── utils.py                           # Shared utilities
├── models/                            # Saved models
│   └── pi_beh.pkl
├── results/                           # Analysis results
│   ├── propensity_scores.csv
│   ├── causal_dag.png
│   ├── fairness_report.md
│   └── domain_boundaries.json
└── notebooks/                         # Exploratory notebooks
    ├── 01_behavior_exploration.ipynb
    ├── 02_causal_graph.ipynb
    └── 03_fairness_analysis.ipynb
```

---

## Key Metrics to Track

| Metric | Target | Rationale |
|--------|--------|-----------|
| Min propensity score | > 0.01 | Avoid extreme importance weights |
| Low-overlap region % | < 20% | Ensure sufficient RL-viable domain |
| E-value (key effects) | > 2.0 | Robustness to unmeasured confounding |
| Min group sample size | > 50 | Adequate representation |
| Stitchability score | > 0.7 | Data supports policy combination |

---

## Common Pitfalls

1. **Ignoring measurement-driven confounding**: Sicker patients get measured more frequently, creating spurious associations
2. **Assuming overlap**: Just because you have data in a region doesn't mean you have good support for all actions
3. **Skipping fairness audit**: Biases in data will amplify in learned policies
4. **Rushing to training**: Time spent here saves weeks of debugging bad policies later

---

## References

- Rosenbaum bounds: Rosenbaum, P. R. (2002). Observational Studies
- E-value: VanderWeele & Ding (2017). "Sensitivity Analysis in Observational Research"
- Stitchability: Miao et al. (2024). "Stitched Trajectories for Offline RL"

---

**Status**: 🔵 Not Started
**Last Updated**: 2025-11-11
