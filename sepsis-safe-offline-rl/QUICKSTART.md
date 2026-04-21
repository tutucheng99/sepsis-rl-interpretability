# Quick Start Guide

## Welcome to the Sepsis Safe Offline RL Project!

This guide will help you get started quickly, whether you're a researcher or an AI agent assisting with the project.

---

## For Researchers: Getting Started

### 1. Environment Setup (5-10 minutes)

```bash
# Navigate to project directory
cd sepsis-safe-offline-rl

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import numpy; print('Setup successful!')"
```

### 2. Configure Experiment Tracking

**Option A: MLflow (Recommended)**
```bash
# Start MLflow server
mlflow server --host 127.0.0.1 --port 5000

# In another terminal, set tracking URI
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

**Option B: Weights & Biases**
```bash
# Login to W&B
wandb login

# Update config/environment.yaml to use wandb
```

### 3. Prepare Your Data

```bash
# Place your data in data/raw/
# The expected format is documented in docs/data_format.md

# Run data preparation (once you have the script)
python phase_0_baseline/prepare_data.py
```

### 4. Start with Phase -1 (Data Audit)

```bash
# Navigate to Phase -1
cd phase_-1_data_audit

# Read the README
cat README.md  # or open in your editor

# Begin with behavioral policy estimation
python behavior_policy.py
```

### 5. Track Your Progress

Open `EXECUTION_CHECKLIST.md` and check off tasks as you complete them. This is your roadmap through the project!

---

## For AI Agents: Navigation Guide

### Understanding the Project Structure

This project is organized by research phases. Each phase has:
- **README.md**: Detailed instructions for that phase
- **Python files**: Implementation code
- **config/**: Configuration files
- **results/**: Output directory

### Where to Start

1. **Read the main README**: `README.md` provides project context
2. **Review the execution plan**: `PROJECT_PLAN.md` contains detailed methodology
3. **Check current status**: `EXECUTION_CHECKLIST.md` shows progress
4. **Consult phase README**: Each `phase_*/README.md` has specific instructions

### Common Tasks and Where to Find Them

| Task | Location |
|------|----------|
| Data audit implementation | `phase_-1_data_audit/` |
| Reward design | `phase_r_reward_design/` |
| Safety layer implementation | `phase_1_safety_layers/` |
| Core RL algorithm | `phase_2_3_core_algorithm/` |
| Evaluation scripts | `phase_4_evaluation/` |
| Utility functions | `utils/` |
| Configuration | `config/` |

### Key Files for Context

When asked to implement or modify code, always consult:
1. **Phase-specific README** for requirements
2. **`config/experiment_config.yaml`** for hyperparameters
3. **`docs/safety_guidelines.md`** for clinical constraints
4. **`utils/`** for reusable components

### Code Implementation Guidelines

**When implementing a new component**:
1. Check if similar code exists in `utils/`
2. Follow the structure shown in existing phase files
3. Use type hints and docstrings
4. Log important events using `utils/logging.py`
5. Update the execution checklist when complete

**Example workflow**:
```python
# 1. Import utilities
from utils.logging import get_logger
from utils.metrics import compute_system_a_reward

# 2. Set up logging
logger = get_logger(__name__)

# 3. Implement your function
def my_function(data, config):
    logger.info("Starting my_function")
    # ... implementation ...
    logger.info("Completed my_function")
    return result
```

### Critical Safety Reminders

- **NEVER** use test set before Phase 4
- **ALWAYS** apply safety masks during training (not just inference)
- **VERIFY** hard constraint violations ≈ 0 before proceeding
- **DOCUMENT** all design decisions in code comments

---

## Project Phases at a Glance

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| **Phase 0** | 1-2 weeks | Data preparation, environment setup |
| **Phase -1** | 3-4 weeks | Causal-fairness audit report |
| **Phase R** | 2-3 weeks | Dual reward system |
| **Phase 0&1** | 3-4 weeks | Baseline models + safety layers |
| **Phase 2&3** | 5-7 weeks | RF-CQL + Filtered-MDP |
| **Phase 4** | 4-5 weeks | Comprehensive evaluation |
| **Phase 5&6** | 3-4 weeks | Robustness analysis |
| **Phase 7-9** | 2-3 weeks | Final report + deployment plan |

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'mlflow'`
- **Solution**: Activate your virtual environment and run `pip install -r requirements.txt`

**Issue**: Gym-Sepsis environment not found
- **Solution**: Install from source or check installation instructions in the environment's repository

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in `config/experiment_config.yaml`

**Issue**: OPE estimates are unstable (high variance)
- **Solution**: Check ESS (effective sample size). If ESS < 50, you may have data quality issues (revisit Phase -1)

### Getting Help

1. **Check phase-specific README** for detailed instructions
2. **Review PROJECT_PLAN.md** for methodology clarification
3. **Consult `docs/safety_guidelines.md`** for clinical questions
4. **Look at `utils/`** for code examples

---

## Next Steps

### For Researchers

1. Complete environment setup
2. Familiarize yourself with the project structure
3. Begin Phase -1 (Data Audit)
4. Regularly update `EXECUTION_CHECKLIST.md`

### For AI Agents

1. Read `README.md` for full context
2. Check `EXECUTION_CHECKLIST.md` for current phase
3. Consult phase-specific README before implementing
4. Update checklist after completing tasks

---

## Important Reminders

### Safety First
This project involves medical decision support. Patient safety must ALWAYS be the top priority.

### Reproducibility
- Fix all random seeds
- Document all hyperparameter choices
- Use version control (Git)
- Track experiments with MLflow/W&B

### Communication
- Update `EXECUTION_CHECKLIST.md` regularly
- Document design decisions in code comments
- Maintain clear commit messages
- Log all safety interventions

---

## Resources

- **Main README**: Complete project overview
- **PROJECT_PLAN**: Detailed execution plan
- **EXECUTION_CHECKLIST**: Progress tracking
- **Safety Guidelines**: Clinical constraints
- **Config Files**: Experiment parameters

---

**Ready to Begin?**

Start by reading `README.md` for the big picture, then dive into Phase -1!

Good luck, and remember: Safety, Reproducibility, Transparency.

---

**Last Updated**: 2025-11-11
