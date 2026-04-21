# Performance and Interpretability Trade-offs in Reinforcement Learning for Sepsis Treatment

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.pdf)
[![License](https://img.shields.io/badge/License-Academic-blue)]()
[![Python](https://img.shields.io/badge/Python-3.10-green)]()

**Authors:** Zhiyu Cheng, Yalun Ding, Chuanhui Peng (Equal contribution)
**Course:** STAT 8289 - Reinforcement Learning
**Institution:** George Washington University
**Date:** October 2025

---

## 🎯 Project Overview

This project investigates the **performance-interpretability trade-off** in reinforcement learning for sepsis treatment, challenging two prevailing assumptions:
1. **Interpretability inevitably compromises performance**
2. **Online RL methods necessarily outperform offline approaches**

### Key Findings

- **Performance**: Online RL achieves marginally higher survival (95.4% vs 94.2%), with only 1.9 percentage point advantage on high-severity patients
- **Interpretability**: Offline CQL exhibits **600-fold stronger** feature importance signals compared to DQN (40.06 vs 0.069)
- **Clinical Deployment**: CQL delivers comparable survival rates with transparent decision logic and no patient risk during training

### Novel Contributions

1. **First quantitative interpretability comparison** using Linearly Estimated Gradients (LEG) across offline and online RL algorithms
2. **Demonstration that the performance-interpretability trade-off is not inevitable** - CQL achieves both high performance and superior interpretability
3. **Comprehensive evaluation** of 8 methods (2 baselines, 3 offline RL, 3 online RL) with SOFA-stratified analysis

---

## 📄 Paper

**Full Paper:** [paper/main.pdf](paper/main.pdf) (29 pages, JASA format)

### Abstract

Sepsis remains a leading cause of mortality in critical care, and reinforcement learning (RL) offers a promising route to data-driven treatment policies. Yet clinical adoption is impeded by the prevailing assumption that interpretability inevitably compromises performance, and that online RL methods necessarily outperform offline approaches. We interrogate these trade-offs by comparing three offline RL methods (Behavior Cloning, Conservative Q-Learning, and Deep Q-Network trained on static datasets) with three online RL methods (Double DQN with Attention, Double DQN with Residual connections, and Soft Actor-Critic with environment interaction) using the gym-sepsis simulator.

Across 500 evaluation episodes, online RL achieves marginally higher overall survival (95.4% for DDQN-Attention vs. 94.2% for BC), with a 1.9 percentage point advantage on high-severity patients (90.5% vs. 88.6%). However, this modest performance gain comes at the cost of requiring extensive environment interaction during training—infeasible in clinical settings. Interpretability analysis reveals that offline methods, particularly CQL, produce LEG saliency peaks of 40.06—roughly 600-fold larger than DQN's 0.069—highlighting clinically coherent emphasis on blood pressure and lactate levels.

**Keywords:** Reinforcement Learning, Sepsis Treatment, Interpretability, Conservative Q-Learning, LEG Analysis, Offline RL, MIMIC-III

---

## 🚀 Quick Start

### View Results

```bash
# View the complete paper
open paper/main.pdf

# View key figures
open results/figures/algorithm_comparison.png
open results/figures/leg_interpretability_comparison.png
```

### Reproduce Experiments

```bash
# 1. Setup environment
conda create -n sepsis_rl python=3.10
conda activate sepsis_rl
pip install -r requirements.txt

# 2. Evaluate all models
python scripts/re_evaluate_all.py

# 3. Run LEG interpretability analysis
python scripts/Interpret_LEG/leg_analysis_offline.py
python scripts/Interpret_LEG/leg_analysis_online.py

# 4. Generate figures
python scripts/create_leg_comparison_figure.py
python scripts/06_visualization.py
```

---

## 📊 Key Results

### Overall Performance (500 Episodes)

| Algorithm | Type | Survival (%) | Avg Return | Avg Length |
|-----------|------|--------------|------------|------------|
| **DDQN-Attention** | Online | **95.4** | 13.62 ± 6.28 | 7.9 ± 1.0 |
| SAC | Online | 94.8 | 13.44 ± 6.66 | 7.7 ± 1.2 |
| BC | Offline | 94.2 | 13.26 ± 7.01 | 9.5 ± 0.6 |
| CQL | Offline | 94.0 | 13.20 ± 7.12 | 9.5 ± 0.5 |
| DQN | Offline | 94.0 | 13.20 ± 7.12 | 7.8 ± 1.2 |

### High-Severity Patients (SOFA ≥ 11)

| Algorithm | Type | Survival (%) | Avg Return |
|-----------|------|--------------|------------|
| **DDQN-Attention** | Online | **90.5** | 12.16 ± 8.79 |
| SAC | Online | 88.7 | 11.62 ± 9.49 |
| BC | Offline | 88.6 | 11.63 ± 9.82 |
| CQL | Offline | 88.5 | 11.55 ± 9.95 |
| DQN | Offline | 84.3 | 10.29 ± 11.46 |

### Interpretability (LEG Analysis)

| Algorithm | Max Saliency | Interpretability | Clinical Deployment |
|-----------|--------------|------------------|---------------------|
| **CQL** | **40.06** | Excellent | Suitable |
| BC | 0.78 | Mixed | Requires validation |
| DQN | 0.069 | Poor | Not suitable |

**600-fold difference** between CQL and DQN in maximum saliency magnitude!

---

## 🗂️ Project Structure

```
project_1/
├── paper/                          # Complete 29-page paper (JASA format)
│   ├── main.pdf                    # Final PDF
│   ├── main.tex                    # LaTeX source
│   ├── sections/                   # Paper sections
│   │   ├── 01_introduction.tex
│   │   ├── 02_related.tex
│   │   ├── 03_problem.tex
│   │   ├── 04_methods.tex
│   │   ├── 05_results.tex
│   │   ├── 06_discussion.tex
│   │   ├── 07_conclusion.tex
│   │   ├── 08_contributions.tex
│   │   └── 09_appendix.tex
│   └── references.bib              # Bibliography (35+ references)
│
├── src/                            # Source code
│   ├── envs/                       # Environment wrappers
│   │   ├── reward_functions.py    # Reward function implementations
│   │   └── sepsis_wrapper.py      # Gym-sepsis wrapper
│   ├── evaluation/                 # Evaluation utilities
│   │   └── metrics.py             # Performance metrics
│   ├── data/                       # Data collection
│   │   └── collect_data.py        # Dataset generation
│   └── visualization/              # Visualization tools
│       ├── interpretability.py    # LEG analysis
│       └── policy_viz.py          # Policy visualization
│
├── scripts/                        # Experiment scripts
│   ├── 01_baseline_evaluation.py  # Baseline policies
│   ├── 02_train_bc.py             # Behavior Cloning
│   ├── 03_train_cql.py            # Conservative Q-Learning
│   ├── 04_train_dqn.py            # Deep Q-Network
│   ├── 06_visualization.py        # Generate figures
│   ├── 07_final_analysis.py       # Final analysis
│   ├── evaluate_yalun_models.py   # Online RL evaluation
│   ├── re_evaluate_all.py         # Re-evaluate all models
│   └── Interpret_LEG/             # LEG interpretability analysis
│       ├── leg_analysis_offline.py
│       └── leg_analysis_online.py
│
├── results/                        # All experimental results
│   ├── figures/                    # Visualization outputs
│   │   ├── algorithm_comparison.png              # Main performance figure
│   │   ├── leg_interpretability_comparison.png   # Main interpretability figure
│   │   └── leg/                   # Detailed LEG analysis (90+ figures)
│   ├── models/                     # Trained models
│   │   ├── bc_simple_reward.d3
│   │   └── cql_simple_reward.d3
│   ├── baseline_results.pkl        # Baseline evaluation
│   ├── bc_results.pkl             # BC evaluation
│   ├── cql_results.pkl            # CQL evaluation
│   ├── dqn_results.pkl            # DQN evaluation
│   └── yalun_models_evaluation.pkl # Online RL evaluation
│
├── data/                           # Datasets
│   ├── offline_dataset.pkl        # ~20K transitions for offline RL
│   └── checkpoint_ep*.pkl         # Training checkpoints
│
├── notebooks/                      # Jupyter notebooks
│   └── LEG_interplate_v2.ipynb    # LEG analysis notebook
│
├── gym-sepsis/                     # Sepsis simulation environment (submodule)
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── PAPER_UPDATE_GUIDE.md          # Paper update instructions
```

---

## 🔬 Methodology

### Algorithms Evaluated

#### Offline RL (No environment interaction during training)
1. **Behavior Cloning (BC)** - Supervised imitation learning baseline
2. **Conservative Q-Learning (CQL)** - Offline RL with conservatism penalty
3. **Deep Q-Network (DQN)** - Q-learning with deep networks (offline mode)

#### Online RL (Environment interaction during training)
4. **Double DQN with Attention** - Attention mechanism for feature selection
5. **Double DQN with Residual** - Residual connections for gradient flow
6. **Soft Actor-Critic (SAC)** - Maximum entropy RL

#### Baselines
7. **Random Policy** - Uniform random action selection
8. **Heuristic Policy** - Clinical guideline-based rules

### Interpretability Analysis

**Method:** Linearly Estimated Gradients (LEG) - A perturbation-based method that approximates Q-function gradients via local linear regression

**Metrics:**
- Maximum saliency magnitude (strength of strongest feature signal)
- Saliency range (differentiation between important/unimportant features)
- Clinical coherence (alignment with medical knowledge)

**Implementation:**
- 1,000 perturbation samples per state
- Standard deviation σ = 0.1
- Ridge regularization λ = 10⁻⁶
- 10 representative states per algorithm

---

## 📈 Figures

### Main Figures in Paper

1. **Figure 1: Algorithm Comparison** (`results/figures/algorithm_comparison.png`)
   - Overall survival rates
   - Average returns
   - SOFA-stratified survival
   - Episode lengths

2. **Figure 2: LEG Interpretability Comparison** (`results/figures/leg_interpretability_comparison.png`)
   - Maximum saliency magnitude comparison (600-fold difference!)
   - Feature importance patterns for CQL, BC, DQN
   - Clinical deployment suitability summary

### Additional Visualizations

- 90+ detailed LEG analysis figures in `results/figures/leg/`
- Individual state analysis for each algorithm
- Feature importance heatmaps
- Q-value landscapes

---

## 🛠️ Technical Details

### Environment
- **Simulator:** gym-sepsis (based on MIMIC-III database)
- **State Space:** 46-dimensional (vital signs, lab values, SOFA scores)
- **Action Space:** 25 actions (5×5 grid: IV fluids × vasopressor dosing)
- **Episodes:** 500 evaluation episodes per algorithm

### Training
- **Offline Dataset:** ~20,000 transitions from heuristic policy
- **Offline Training:** BC (30 min), CQL (1-2 hours), DQN (1-2 hours)
- **Online Training:** 1M timesteps with environment interaction
- **Compute:** NVIDIA A100 GPU (Google Colab)

### Dependencies
```
Python 3.10
d3rlpy (offline RL)
stable-baselines3 (online RL)
gym==0.21.0
torch
numpy, pandas, matplotlib, seaborn
```

---

## 📚 References

### Key Papers

1. **Raghu et al. (2017)** - "Deep Reinforcement Learning for Sepsis Treatment" - NeurIPS Workshop on ML for Health
2. **Kumar et al. (2020)** - "Conservative Q-Learning for Offline Reinforcement Learning" - NeurIPS
3. **Greydanus et al. (2018)** - "Visualizing and Understanding Atari Agents" - ICML (LEG method)

### Data and Code

- **MIMIC-III Database:** https://mimic.mit.edu/
- **gym-sepsis Environment:** https://github.com/gefeilin/gym-sepsis/tree/main/gym_sepsis/envs
- **This Project:** https://github.com/tutucheng99/STAT_8289_Project_1_Zhiyu_Cheng

---

## 👥 Author Contributions

- **Zhiyu Cheng**: Designed and implemented offline RL experiments (BC, CQL, DQN); conducted baseline evaluations; performed LEG interpretability analysis for offline methods; wrote Methods and Results sections; contributed to Discussion and Conclusion.

- **Yalun Ding**: Designed and implemented online RL experiments (DDQN-Attention, DDQN-Residual, SAC); conducted SOFA-stratified analysis; performed LEG interpretability analysis for online methods; wrote Introduction and Related Work sections; contributed to Discussion.

- **Chuanhui Peng**: Managed offline dataset generation and quality control; coordinated cross-algorithm evaluation; created main comparison figures; wrote Abstract and Problem Formulation sections; managed bibliography and LaTeX formatting.

---

## 📧 Contact

**Zhiyu Cheng** - George Washington University
**Course:** STAT 8289 - Reinforcement Learning
**Instructor:** [Instructor Name]
**Semester:** Fall 2025

---

## 📜 License

This project is for **academic use only**.

- Based on MIMIC-III database (PhysioNet Credentialed Health Data License)
- Uses gym-sepsis environment (MIT License)
- Code available for educational and research purposes

---

## 🙏 Acknowledgments

- STAT 8289 course staff at George Washington University
- Authors of gym-sepsis simulation environment
- MIMIC-III database contributors
- d3rlpy and Stable-Baselines3 library developers

---

**Project Status:** ✅ Complete
**Paper Status:** ✅ Final (29 pages)
**Last Updated:** October 28, 2025
