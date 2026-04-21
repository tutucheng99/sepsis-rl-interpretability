# Project Execution Checklist

## Current Status
- **Current Phase**: Phase 0 (Preparation)
- **Start Date**: 2025-11-11
- **Last Updated**: 2025-11-11
- **Overall Progress**: 0% (0/8 phases completed)

---

## Phase 0: Infrastructure & Data Preparation (1-2 weeks)

**Status**: 🔵 Not Started
**Start Date**: TBD
**Completion Date**: TBD

### Task Checklist

#### 1. Environment & Toolchain Configuration
- [ ] Configure Gym-Sepsis environment
  - [ ] Verify state space (S, with SOFA core features)
  - [ ] Verify action space (A, discrete 5x5 grid)
- [ ] Unify computational environment
  - [ ] Fix Python version: _______
  - [ ] Fix d3rlpy version: _______
  - [ ] Fix PyTorch/TensorFlow version: _______
- [ ] Set global numerical precision (FP32)
- [ ] Set up Git version control
- [ ] Set up experiment tracking system (MLflow/Weights & Biases)
  - [ ] Tool selected: _______
  - [ ] Configuration completed

#### 2. Data Splitting & Isolation
- [ ] Extract ~10k patient trajectories
- [ ] Execute strict split:
  - [ ] Training set (~9k trajectories)
  - [ ] Validation set (~0.5k trajectories)
  - [ ] Test set (~0.5k trajectories)
- [ ] **SEAL TEST SET** (Do not touch until Phase 4!)
- [ ] Document data split methodology
- [ ] Verify no data leakage between splits

#### 3. Feature Engineering & Preprocessing Standardization
- [ ] Define feature normalization strategy
- [ ] Define missing value imputation strategy
- [ ] Fit preprocessor on training set only
- [ ] Apply preprocessor to all datasets
- [ ] Lock preprocessing logic version
- [ ] Verify train-inference preprocessing consistency

### Deliverables
- [ ] Standardized datasets (train/val/test)
- [ ] Locked preprocessing scripts
- [ ] Configured experiment environment
- [ ] Environment setup documentation

### Notes
_Record any important observations or decisions here_

---

## Phase -1: Data Layer Causal-Fairness Audit (3-4 weeks)

**Status**: 🔵 Not Started
**Start Date**: TBD
**Completion Date**: TBD

### Task Checklist

#### Task 1.1: Behavioral Policy Estimation & Overlap Analysis
- [ ] Fit behavioral policy π_beh(a|s,t) on training set
- [ ] Compute propensity scores
- [ ] Evaluate overlap/positivity for state-action pairs
- [ ] Identify regions with insufficient data support
- [ ] Document low-overlap regions

#### Task 1.2: Causal Graph Construction & Confounding Sensitivity
- [ ] Construct minimal causal DAG
  - [ ] Clarify S, A, R, T relationships
  - [ ] Incorporate shadow variables (measurement frequency, missingness indicators)
- [ ] Perform hidden confounding sensitivity analysis
  - [ ] Implement Rosenbaum bounds or E-value method
  - [ ] Quantify "confounding strength needed to overturn conclusions"
- [ ] Quantify uncertainty
  - [ ] Implement Delphi uncertainty U_del(s,a)
  - [ ] Document uncertainty quantification methodology

#### Task 1.3: Fairness Audit & Stitchability Gate
- [ ] Perform representativeness & bias audit
  - [ ] Stratify by key groups (age, gender, baseline SOFA)
  - [ ] Report sample coverage rates
  - [ ] Report behavioral differences across groups
- [ ] Evaluate stitchability
  - [ ] Assess state overlap across behavioral patterns (clusters)
  - [ ] Evaluate myopic value consistency
- [ ] Make usability gate decision
  - [ ] Label "RL-restricted zones" (low overlap regions)
  - [ ] Define BC baseline for restricted zones

### Deliverables
- [ ] Causal-fairness audit report
- [ ] π_beh model implementation
- [ ] U_del(s,a) implementation
- [ ] RL/BC domain boundary list

### Checkpoint 1
- [ ] Has data audit identified major biases or confounding risks?
- [ ] Do RL-viable domains sufficiently cover target population?
- [ ] Decision: Proceed / Revise data strategy / Abort

### Notes
_Record audit findings and decisions_

---

## Phase R: Dual Reward System Design (2-3 weeks)

**Status**: 🔵 Not Started
**Start Date**: TBD
**Completion Date**: TBD

### Task Checklist

#### Task R.1: System B (Evaluation) Metrics Definition
- [ ] Define clinical endpoints
  - [ ] Survival rate
  - [ ] ICU length of stay
  - [ ] Other: _______
- [ ] Define process safety indicators
  - [ ] Drug exposure metrics
  - [ ] AKI proxies
  - [ ] Other: _______
- [ ] Ensure System B does not participate in gradient updates

#### Task R.2: System A (Training) Reward Shaping
- [ ] Implement reward function structure
  - [ ] Component 1: R_phys (physiological indicators)
  - [ ] Component 2: Smoothness penalty (action changes)
  - [ ] Component 3: Risk penalty (high-risk states)
- [ ] Implement R_phys
  - [ ] MAP ≥ 65 target
  - [ ] Lactate reduction target
  - [ ] Use continuous shaping (avoid hard thresholds)
- [ ] Implement smoothness penalty
- [ ] Implement risk penalty
- [ ] Implement saturation suppression for terminal states

#### Task R.3: Weight Calibration & Consistency Check
- [ ] Define weight search space (w1, w2, w3)
- [ ] Optimize weights on validation set
- [ ] **Critical: Consistency check**
  - [ ] Compute correlation between System A cumulative reward and System B metrics
  - [ ] Compute monotonicity metrics (Spearman/Pearson)
  - [ ] Document correlation coefficients: _______
- [ ] If decoupling detected (high A, low B):
  - [ ] Document issue
  - [ ] Adjust weights or R_phys design
  - [ ] Re-run consistency check

### Deliverables
- [ ] System A and B implementation code
- [ ] Optimal weight configuration
- [ ] Consistency validation report

### Checkpoint 2
- [ ] Do System A and B pass consistency check?
- [ ] If not, is there a clear callback record and correction plan?
- [ ] Decision: Proceed / Revise reward design

### Notes
_Record reward design decisions and consistency check results_

---

## Phase 0 & 1: Baseline Establishment & Safety Layer Definition (3-4 weeks)

**Status**: 🔵 Not Started
**Start Date**: TBD
**Completion Date**: TBD

### Task Checklist

#### Task 0.1: Baseline Reproduction & OPE Configuration (Phase 0)
- [ ] Implement baseline algorithms
  - [ ] BC (on high-quality subset from Phase -1)
  - [ ] Standard CQL (fixed α)
- [ ] Configure OPE suite
  - [ ] Implement FQE (primary)
  - [ ] Implement WIS
  - [ ] Implement DR
- [ ] Establish evaluation standards
  - [ ] Fixed random seeds
  - [ ] Wilson confidence intervals
- [ ] Ensure OPE reports ESS and weight variance

#### Task 1.1: L1 Semantic Safety Layer (Hard Constraints) (Phase 1)
- [ ] Define semantic catalog
  - [ ] Review clinical guidelines
  - [ ] Consult domain experts
  - [ ] Document hard contraindications
  - [ ] Document dosage boundaries
  - [ ] Document absolute incompatibilities
- [ ] Implement action mask generator
  - [ ] Function to generate A_safe(s) for any state s
  - [ ] Unit tests for mask generator

#### Task 1.2: L2 Cognitive Safety Layer (Uncertainty/OOD) (Phase 1)
- [ ] Train risk proxy models
  - [ ] Model for ΔSOFA deterioration
  - [ ] Model for sustained low MAP
  - [ ] Calibrate models (e.g., isotonic regression)
- [ ] Implement OOD/overlap detector
  - [ ] Choose method: density estimation / ensemble / other: _______
  - [ ] Implement detector
  - [ ] Validate detector performance
- [ ] Set L2 thresholds on validation set
  - [ ] OOD threshold (δ): _______
  - [ ] Confounding uncertainty threshold (κ): _______

### Deliverables
- [ ] Baseline model performance report
- [ ] OPE suite implementation
- [ ] L1 catalog and A_safe(s) implementation
- [ ] L2 detectors and thresholds

### Notes
_Record baseline performance and safety threshold decisions_

---

## Phase 2 & 3: Core Algorithm Implementation & Training (5-7 weeks)

**Status**: 🔵 Not Started
**Start Date**: TBD
**Completion Date**: TBD

### Task Checklist

#### Task 2.1: Adaptive Conservatism Implementation (Phase 2)
- [ ] Modify CQL to support state-dependent α
- [ ] Implement modulation logic: α_eff(s) = α_0 + γ·OOD(s)
- [ ] Unit test adaptive conservatism
- [ ] Validate that higher OOD leads to higher conservatism

#### Task 2.2: Filtered-MDP Training Implementation (Phase 2) **CRITICAL**
- [ ] Modify Bellman backup to integrate L1 action mask
  - [ ] Implement: y = r + γ max_{a'∈A_safe(s')} Q(s',a')
  - [ ] Verify mask is applied during backup computation
- [ ] Ensure train-inference isomorphism
  - [ ] Apply mask during action selection in training
  - [ ] Avoid first-time mask application at inference
- [ ] Unit test Filtered-MDP Bellman operator
- [ ] Validate against baseline (non-filtered) CQL

#### Task 2.3: Inference-Time Hierarchical Arbiter (Phase 2)
- [ ] Implement hierarchical decision logic (L1→L2→L3)
  - [ ] L1 (Semantic): Check A_safe(s)
  - [ ] L2 (Cognitive): Check OOD ≤ δ and U_del ≤ κ
  - [ ] L3 (RL Optimization): Execute argmax Q(s,a)
- [ ] Define fallback policy
  - [ ] Strategy: Behavioral nearest-neighbor safe action / Human request / Other: _______
  - [ ] Implement fallback logic
- [ ] Implement detailed intervention logging
  - [ ] Log L1 interventions
  - [ ] Log L2 interventions
  - [ ] Log fallback invocations

#### Task 3.1: Training Protocol & Hyperparameter Optimization (Phase 3)
- [ ] Define hyperparameter search space
  - [ ] α_0 range: _______
  - [ ] γ range: _______
  - [ ] δ range: _______
  - [ ] κ range: _______
- [ ] Tune hyperparameters on validation set
- [ ] Train with multiple random seeds
  - [ ] Number of seeds: _______
  - [ ] Seeds used: _______
- [ ] Use OPE (FQE primary) for early stopping
- [ ] Monitor ESS and weight variance
  - [ ] Set minimum ESS threshold: _______

#### Task 3.2: Confounding Pessimization & Model Selection (Phase 3)
- [ ] Implement confounding pessimization for model selection
  - [ ] Implement pessimistic Q: Q̃(s,a) = Q̂(s,a) - β·U_del(s,a)
  - [ ] Choose β: _______
  - [ ] **DO NOT use for training gradients**
- [ ] Construct Pareto frontier
  - [ ] Dimensions: Performance, Safety, Interpretability
  - [ ] Select non-dominated candidate policies
  - [ ] Document Pareto frontier

### Deliverables
- [ ] RF-CQL + Filtered-MDP complete implementation
- [ ] Inference arbiter implementation
- [ ] Trained candidate policy set
- [ ] Training logs and hyperparameter records

### Checkpoint 3
- [ ] Is Filtered-MDP correctly implemented (mask in Bellman backup)?
- [ ] Are training and inference isomorphic?
- [ ] Is OPE evaluation stable (sufficient ESS)?
- [ ] Decision: Proceed to Phase 4 / Debug and re-train

### Notes
_Record training process, convergence behavior, and any issues_

---

## Phase 4: Main Evaluation (4-5 weeks)

**Status**: 🔵 Not Started
**Start Date**: TBD
**Completion Date**: TBD

**⚠️ IMPORTANT: Unseal test set for this phase only!**

### Task Checklist

#### 1. Efficacy (Performance) Evaluation (H1)
- [ ] Report survival rate / return (System B metrics)
  - [ ] Overall population: _______
  - [ ] High-risk subgroup (SOFA ≥ 11): _______
- [ ] Statistical reporting
  - [ ] Provide 95% Wilson confidence intervals
  - [ ] Perform significance testing
  - [ ] Report effect sizes
  - [ ] Apply multiple comparison correction (BH-FDR)

#### 2. Safety Evaluation (H1)
- [ ] **Hard safety**: Hard contraindication violation rate
  - [ ] Rate: _______ (MUST be ≈ 0 - RED LINE)
- [ ] **Soft safety**: Risk violation rate and OOD behavior rate
  - [ ] Risk violation rate: _______
  - [ ] OOD behavior rate: _______
  - [ ] Compare to baseline (should be significantly lower)
- [ ] **Operability**: Filter intervention rates
  - [ ] L1 intervention rate: _______
  - [ ] L2 intervention rate: _______
  - [ ] Alert burden analysis

#### 3. Auditability (Interpretability) Evaluation (H2)
- [ ] Compute LEG/SHAP top-k feature stability
  - [ ] Features analyzed: MAP, lactate, _______
  - [ ] Jaccard similarity: _______
  - [ ] Rank correlation: _______
- [ ] Perform monotonicity verification
- [ ] Conduct clinical consistency assessment
- [ ] Generate auditable "policy card"
- [ ] Execute critical case reviews
  - [ ] Number of cases reviewed: _______
  - [ ] Document key findings

#### 4. Fairness Evaluation (H4)
- [ ] Evaluate across key stratifications
  - [ ] Stratification variables: Age, Gender, Baseline SOFA, _______
  - [ ] Performance disparities: _______
  - [ ] Safety metric disparities: _______
  - [ ] Filter intervention rate disparities: _______
- [ ] Assess systematic unfairness
- [ ] Document fairness findings

#### 5. Pseudo-Dynamic / Shadow Approximation Evaluation
- [ ] **Tier A (Required)**: Streaming replay on test set
  - [ ] Record policy-clinician disagreement rate: _______
  - [ ] Record temporal stability
  - [ ] Record OOD frequency: _______
  - [ ] Record filter intervention rate: _______

### Deliverables
- [ ] Final comprehensive evaluation report
- [ ] Statistical analysis results
- [ ] Policy card
- [ ] Case review documentation

### Checkpoint 4 (Project Acceptance)
- [ ] Does policy meet all acceptance criteria?
- [ ] **CRITICAL**: Hard contraindication violations ≈ 0?
- [ ] Are fairness thresholds met?
- [ ] Decision: Accept policy / Reject and iterate

### Notes
_Record evaluation results and acceptance decision_

---

## Phase 5 & 6: Robustness & Ablation Studies (3-4 weeks)

**Status**: 🔵 Not Started
**Start Date**: TBD
**Completion Date**: TBD

### Task Checklist

#### 1. Ablation Studies

##### Algorithm Layer
- [ ] Compare RF-CQL vs. Standard CQL
- [ ] Compare RF-CQL vs. Post-hoc filtering only
- [ ] Compare RF-CQL vs. Adaptive conservatism only
- [ ] Compare RF-CQL vs. BC

##### Safety Layer
- [ ] Ablate L1 filtering
- [ ] Ablate L2 gating
- [ ] Ablate confounding pessimization U_del

##### Reward Layer
- [ ] Ablate smoothness penalty
- [ ] Ablate risk penalty

#### 2. Robustness & Sensitivity Analysis
- [ ] Sweep key hyperparameters
  - [ ] α_0 sensitivity
  - [ ] γ sensitivity
  - [ ] δ sensitivity
  - [ ] κ sensitivity
  - [ ] β sensitivity
  - [ ] w_i (reward weights) sensitivity
- [ ] Analyze impact on Pareto frontier
- [ ] Test robustness to feature noise
- [ ] Test robustness to missing data
- [ ] Test robustness to temporal drift
- [ ] Report failure cases

### Deliverables
- [ ] Ablation study report
- [ ] Robustness & sensitivity analysis report

### Notes
_Record component contributions and robustness findings_

---

## Phase 7-9: Acceptance, Reporting & Deployment Planning (2-3 weeks)

**Status**: 🔵 Not Started
**Start Date**: TBD
**Completion Date**: TBD

### Task Checklist

#### 1. Acceptance (Phase 7)
- [ ] Review against acceptance criteria checklist
- [ ] Confirm all safety/fairness thresholds met
- [ ] Exclude any policies failing safety/fairness gates from main conclusions
- [ ] Document acceptance decision

#### 2. Reporting & Summarization (Phase 8)
- [ ] Draft final research paper
  - [ ] Introduction
  - [ ] Methods
  - [ ] Results
  - [ ] Discussion
  - [ ] Conclusion
- [ ] Write technical report
- [ ] Clarify evidence contribution of this study
- [ ] Prepare presentation materials

#### 3. Deployment Path Planning (Phase 9)
- [ ] Design shadow mode implementation plan
  - [ ] Define shadow deployment protocol
  - [ ] Define monitoring metrics
- [ ] Define pre-deployment shutdown thresholds
- [ ] Design prospective trial framework
  - [ ] HITL (Human-in-the-Loop) protocol
  - [ ] PCCP (Prospective Clinical Comparison Protocol)
- [ ] Document deployment prerequisites

### Deliverables
- [ ] Final research paper
- [ ] Technical report
- [ ] Code repository (with documentation)
- [ ] Deployment planning document

### Important Note
**Do NOT initiate any clinical trial-like deployment before completing shadow evaluation and shutdown strategy**

### Notes
_Record final conclusions and deployment planning_

---

## Overall Project Status Summary

### Phase Completion Status
- [ ] Phase 0: Preparation (1-2 weeks)
- [ ] Phase -1: Data Audit (3-4 weeks)
- [ ] Phase R: Reward Design (2-3 weeks)
- [ ] Phase 0&1: Baseline + Safety Layers (3-4 weeks)
- [ ] Phase 2&3: Core Algorithm (5-7 weeks)
- [ ] Phase 4: Main Evaluation (4-5 weeks)
- [ ] Phase 5&6: Robustness (3-4 weeks)
- [ ] Phase 7-9: Acceptance & Reporting (2-3 weeks)

### Critical Red Lines (Must Pass)
- [ ] Test set remained sealed until Phase 4
- [ ] Hard contraindication violations ≈ 0
- [ ] Filtered-MDP implemented correctly (mask in Bellman)
- [ ] Train-inference isomorphism maintained
- [ ] System A-B consistency validated
- [ ] Fairness criteria met across all key groups

### Key Decisions Log
| Date | Phase | Decision | Rationale |
|------|-------|----------|-----------|
|      |       |          |           |

### Risk Register
| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
|      |        |            |        |

---

**Last Updated**: 2025-11-11
**Updated By**: Initial Setup
**Next Review Date**: _______
