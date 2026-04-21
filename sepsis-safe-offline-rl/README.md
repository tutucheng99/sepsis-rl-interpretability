# ICU Sepsis Safe Offline Reinforcement Learning

## Project Overview

This project implements a safe, auditable offline reinforcement learning system for dynamic treatment regimens in ICU sepsis management. The research emphasizes multi-layered safety architecture, causal auditing, and rigorous evaluation protocols.

## Research Objectives

1. **Safety-First Design**: Implement a 3-layer safety architecture (L1: Semantic, L2: Cognitive, L3: RL Optimization)
2. **Causal Auditing**: Address confounding and data quality issues before training
3. **Dual Reward System**: Separate training rewards (System A) from evaluation metrics (System B)
4. **Comprehensive Evaluation**: Validate performance, safety, interpretability, and fairness

## Project Structure

```
sepsis-safe-offline-rl/
├── README.md                          # This file
├── PROJECT_PLAN.md                    # Detailed phase-by-phase execution plan
├── EXECUTION_CHECKLIST.md             # Progress tracking checklist
├── requirements.txt                   # Python dependencies
├── config/                            # Configuration files
├── data/                              # Data storage (train/val/test splits)
├── phase_-1_data_audit/               # Phase -1: Causal & fairness data audit
├── phase_0_baseline/                  # Phase 0: Baseline models & OPE suite
├── phase_1_safety_layers/             # Phase 1: L1 & L2 safety layer implementation
├── phase_2_3_core_algorithm/          # Phase 2-3: RF-CQL + Filtered-MDP
├── phase_4_evaluation/                # Phase 4: Comprehensive evaluation
├── phase_5_6_robustness/              # Phase 5-6: Ablation & robustness tests
├── phase_r_reward_design/             # Phase R: Dual reward system design
├── utils/                             # Shared utility functions
├── notebooks/                         # Jupyter notebooks for exploration
├── experiments/                       # Experiment logs and results
├── logs/                              # Runtime logs
└── docs/                              # Additional documentation
```

## Quick Start Guide

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Gym-Sepsis environment
# (See config/environment.yaml for details)
```

### 2. Data Preparation (Phase 0)

```bash
# Run data preparation script
python phase_0_baseline/prepare_data.py

# This will:
# - Load ~10k patient trajectories
# - Split into train (9k) / val (0.5k) / test (0.5k)
# - Apply standardized preprocessing
# - Seal test set until Phase 4
```

### 3. Execute Research Phases Sequentially

Follow the execution order defined in `PROJECT_PLAN.md`:
- **Phase -1**: Data audit (3-4 weeks)
- **Phase R**: Reward design (2-3 weeks)
- **Phase 0 & 1**: Baseline + safety layers (3-4 weeks)
- **Phase 2 & 3**: Core algorithm (5-7 weeks)
- **Phase 4**: Main evaluation (4-5 weeks)
- **Phase 5 & 6**: Robustness tests (3-4 weeks)

## Critical Safety Protocols

### Test Set Isolation
- **NEVER** use test set before Phase 4 main evaluation
- No hyperparameter tuning on test set
- No threshold setting on test set

### Safety Red Lines
- **Hard constraint violations must ≈ 0** (acceptance criterion)
- All policies must pass L1 semantic safety checks
- Document any safety violations immediately

### Reproducibility Requirements
- Fix random seeds for all experiments
- Use unified precision (FP32)
- Version control all code and configs
- Track experiments with MLflow or similar

## Key Files and Entry Points

| File | Purpose |
|------|---------|
| `EXECUTION_CHECKLIST.md` | Track progress through research phases |
| `config/experiment_config.yaml` | Central experiment configuration |
| `phase_*/README.md` | Phase-specific documentation and instructions |
| `utils/logging.py` | Standardized logging and experiment tracking |
| `utils/metrics.py` | Evaluation metrics (System A & B) |

## Development Workflow

### For Researchers

1. **Check current phase** in `EXECUTION_CHECKLIST.md`
2. **Read phase README** for specific tasks and deliverables
3. **Implement components** following safety protocols
4. **Log experiments** with detailed metadata
5. **Update checklist** upon phase completion
6. **Pass checkpoint** before advancing to next phase

### For AI Agents

1. **Consult this README** for project context and structure
2. **Review `PROJECT_PLAN.md`** for detailed task specifications
3. **Check `EXECUTION_CHECKLIST.md`** for current status
4. **Read phase-specific README** before implementing
5. **Follow naming conventions** and code organization patterns
6. **Update documentation** as code evolves

## Agent-Friendly Guidelines

### File Organization
- **One phase = one directory**: All code for a phase lives in its directory
- **Shared utilities**: Common functions go in `utils/`
- **Configuration over hardcoding**: Use `config/` files for parameters
- **Self-documenting code**: Clear variable names, comprehensive docstrings

### Naming Conventions
- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_CASE`
- **Modules**: Descriptive names (e.g., `behavior_policy.py`, not `bp.py`)

### Documentation Standards
- **Module-level docstring**: Purpose, key classes/functions
- **Function docstring**: Args, returns, raises, examples
- **Class docstring**: Purpose, attributes, usage example
- **Inline comments**: For complex logic only

### Code Quality
- **Type hints**: Use for function signatures
- **Error handling**: Explicit exception handling with logging
- **Testing**: Write unit tests for critical components
- **Linting**: Follow PEP 8 style guidelines

## Checkpoints and Deliverables

### Checkpoint 1 (After Phase -1)
- [ ] Data audit report completed
- [ ] Behavioral policy π_beh fitted
- [ ] Confounding uncertainty U_del quantified
- [ ] RL/BC domain boundaries defined

### Checkpoint 2 (After Phase R)
- [ ] System A (training reward) implemented
- [ ] System B (evaluation metrics) defined
- [ ] Reward weight calibration completed
- [ ] Consistency validation passed

### Checkpoint 3 (After Phase 2-3)
- [ ] Filtered-MDP correctly implemented
- [ ] RF-CQL algorithm validated
- [ ] Inference arbiter functional
- [ ] OPE evaluation stable (sufficient ESS)

### Checkpoint 4 (After Phase 4 - Final Acceptance)
- [ ] Hard safety violations ≈ 0
- [ ] Performance targets met on test set
- [ ] Interpretability validated
- [ ] Fairness criteria satisfied

## Related Resources

- **Previous Research**: See `../paper/` for prior work
- **Gym-Sepsis Environment**: [Documentation link]
- **Clinical Guidelines**: See `docs/clinical_references.md`
- **Safety Protocols**: See `docs/safety_guidelines.md`

## Contact and Support

For questions about:
- **Research methodology**: Consult `PROJECT_PLAN.md`
- **Implementation details**: Check phase-specific READMEs
- **Code issues**: Review `utils/` documentation
- **Project status**: See `EXECUTION_CHECKLIST.md`

## License and Ethics

This project involves medical decision support systems. All implementations must:
- Prioritize patient safety above performance metrics
- Undergo rigorous validation before any clinical consideration
- Maintain transparency and auditability
- Follow established medical ethics guidelines

**IMPORTANT**: This is a research prototype. It is NOT approved for clinical use and requires extensive validation, shadow deployment, and regulatory approval before any patient-facing application.

---

**Last Updated**: 2025-11-11
**Project Status**: Initialization Phase
**Next Milestone**: Complete Phase 0 (Data Preparation)
