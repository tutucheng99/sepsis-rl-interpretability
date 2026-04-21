# Code Repository - Sepsis RL Interpretability Study

**Manuscript:** Performance and Interpretability Trade-offs in Reinforcement Learning for Sepsis Treatment: Comparing Offline and Online Approaches

---

## 📦 Requirements

### Python Environment
```bash
Python 3.10+
```

### Core Dependencies
```bash
# Offline RL Library
d3rlpy >= 2.0.0

# Online RL Library
stable-baselines3 >= 2.0.0

# Environment
gym >= 0.26.0
gym-sepsis  # Custom environment (see Installation)

# Deep Learning
torch >= 2.0.0
tensorflow >= 2.15.0

# Data & Visualization
numpy < 2.0
pandas
matplotlib
seaborn
plotly

# Utilities
jupyter
tqdm
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏗️ Project Structure

```
code/
├── src/
│   ├── envs/
│   │   ├── sepsis_wrapper.py       # Gym-sepsis environment wrapper
│   │   └── reward_functions.py     # Reward shaping utilities
│   ├── evaluation/
│   │   └── metrics.py              # Performance metrics (survival, return, length)
│   ├── visualization/
│   │   ├── leg_analysis.py         # LEG interpretability analysis
│   │   └── plotting.py             # Result visualization
│   └── data/
│       └── dataset_utils.py        # Offline dataset collection
│
└── scripts/
    ├── 01_baseline_evaluation.py   # Evaluate random & heuristic policies
    ├── 02_train_bc.py              # Train Behavior Cloning
    ├── 03_train_cql.py             # Train Conservative Q-Learning
    ├── 04_train_dqn.py             # Train Deep Q-Network
    ├── 06_visualization.py         # Generate result figures
    └── 07_final_analysis.py        # Comprehensive LEG + performance analysis
```

---

## 🚀 Quick Start

### 1. **Install Gym-Sepsis Environment**

The gym-sepsis environment is a MIMIC-III-derived sepsis simulator.

```bash
# Clone gym-sepsis repository
git clone https://github.com/akiani/gym-sepsis.git
cd gym-sepsis

# Install in development mode
pip install -e .
```

**Important:** The environment uses pre-trained transition dynamics and reward models learned from MIMIC-III data.

### 2. **Generate Offline Dataset**

Collect 10,000 episodes using heuristic policy:

```bash
python scripts/00_generate_dataset.py \
    --n_episodes 10000 \
    --output data/offline_dataset.pkl
```

**Dataset Statistics:**
- ~100,000 transitions
- 94.6% survival rate
- Split: 9,000 train / 500 val / 500 test

### 3. **Evaluate Baselines**

```bash
python scripts/01_baseline_evaluation.py
```

**Output:** `results/baseline_results.pkl`

### 4. **Train Offline RL Algorithms**

```bash
# Behavior Cloning
python scripts/02_train_bc.py \
    --dataset data/offline_dataset.pkl \
    --n_epochs 10 \
    --batch_size 1024

# Conservative Q-Learning
python scripts/03_train_cql.py \
    --dataset data/offline_dataset.pkl \
    --n_steps 200000 \
    --alpha 1.0

# Deep Q-Network (online training)
python scripts/04_train_dqn.py \
    --n_timesteps 100000 \
    --buffer_size 100000
```

**Output:**
- Trained models: `results/models/bc_simple_reward.d3`, `cql_simple_reward.d3`, `dqn_simple_reward.zip`
- Evaluation results: `results/bc_results.pkl`, `cql_results.pkl`, `dqn_results.pkl`

### 5. **Run LEG Interpretability Analysis**

```bash
python scripts/07_final_analysis.py \
    --models results/models/ \
    --n_states 10 \
    --n_samples 1000 \
    --sigma 0.1
```

**Output:**
- `results/figures/leg_6model_comparison.png` - Main interpretability figure
- `results/figures/leg/[model]/` - Detailed per-state LEG visualizations

### 6. **Generate Publication Figures**

```bash
python scripts/06_visualization.py
```

**Output:**
- `results/figures/algorithm_comparison.png` - Performance comparison
- `results/figures/leg_interpretability_comparison.png` - Interpretability summary

---

## 🧪 Reproducibility

### Random Seeds
All experiments use fixed random seeds for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Gym: `env.seed(42)`

### Computational Resources
- **Training Time:** ~2 hours per algorithm (on RTX 3090)
- **Memory:** ~8GB RAM, ~4GB VRAM
- **Storage:** ~500MB (datasets + models)

### Hyperparameters

**Behavior Cloning (BC):**
```python
batch_size = 1024
learning_rate = 1e-3
n_epochs = 10
encoder = [256, 256, 128]
```

**Conservative Q-Learning (CQL):**
```python
batch_size = 1024
learning_rate = 3e-4
alpha = 1.0  # Conservatism penalty
n_steps = 200000
target_update_interval = 2000
encoder = [256, 256, 128]
```

**Deep Q-Network (DQN):**
```python
batch_size = 256
learning_rate = 1e-4
buffer_size = 100000
n_timesteps = 100000
exploration = epsilon_greedy(1.0 → 0.05)
target_update_interval = 1000
network = [256, 256, 128]
```

---

## 📊 Expected Results

### Performance Metrics (500 episodes):

| Algorithm | Survival (%) | Avg Return | High-SOFA Survival (%) |
|-----------|-------------|------------|------------------------|
| Random | 95.0 | 13.50 | ~88% |
| Heuristic | 94.6 | 13.38 | ~89% |
| BC | 94.2 | 13.26 | 88.6 |
| CQL | 94.0 | 13.20 | 88.5 |
| DQN | 94.0 | 13.20 | 84.3 |
| DDQN-Attention | **95.4** | 13.62 | **90.5** |
| DDQN-Residual | 94.2 | 13.26 | 87.0 |
| SAC | 94.8 | 13.44 | 88.7 |

### Interpretability Metrics (LEG Analysis):

| Algorithm | Max Saliency | Top Feature | Interpretability |
|-----------|-------------|-------------|------------------|
| **CQL** | **40.06** | SysBP | ⭐⭐⭐⭐⭐ Excellent |
| DDQN-Attention | 3.57 | qSOFA | ⭐⭐⭐ Moderate |
| DDQN-Residual | 2.93 | INR | ⭐⭐⭐ Moderate |
| SAC | 1.17 | INR | ⭐⭐ Mixed |
| BC | 0.78 | Various | ⭐⭐ Mixed |
| DQN | 0.069 | Various | ⭐ Poor |

**Key Finding:** 600-fold interpretability difference (CQL vs DQN)

---

## 🔬 LEG Analysis Details

### Algorithm Overview
LEG (Linearly Estimated Gradients) approximates policy feature importance via local linear regression:

1. Sample perturbations around state: $s' \sim \mathcal{N}(s, \sigma^2 I)$
2. Compute Q-values: $Q(s', \pi(s'))$
3. Fit ridge regression: $Q \approx \beta_0 + \sum_j \gamma_j \cdot s'_j$
4. Saliency scores: $\gamma_j$ indicate feature importance

### Parameters Used
- **n_samples:** 1,000 perturbations per state
- **sigma:** 0.1 (perturbation noise)
- **n_states:** 10 representative states (uniform across SOFA levels)
- **ridge_alpha:** 0.01 (regularization)

### Clinical Interpretation

**CQL Top Features (High Saliency):**
- Systolic BP: -40.06 (decreasing BP → aggressive treatment)
- Lactate: -37.75 (increasing lactate → fluid resuscitation)
- Mean BP: -24.50 (MAP <65 mmHg → vasopressor escalation)

These align perfectly with Surviving Sepsis Campaign guidelines!

**DQN Top Features (Weak Saliency):**
- All features <0.07 (no coherent clinical pattern)
- Cannot validate policy logic

---

## 📁 Output Files

### Trained Models
```
results/models/
├── bc_simple_reward.d3          # Behavior Cloning (d3rlpy format)
├── cql_simple_reward.d3         # Conservative Q-Learning
└── dqn_simple_reward.zip        # DQN (Stable-Baselines3 format)
```

### Evaluation Results
```
results/
├── baseline_results.pkl         # Random & heuristic performance
├── bc_results.pkl               # BC evaluation
├── cql_results.pkl              # CQL evaluation
├── dqn_results.pkl              # DQN evaluation
└── yalun_models_evaluation.pkl  # Online RL results (DDQN, SAC)
```

### Figures
```
results/figures/
├── algorithm_comparison.png                    # Main performance figure
├── leg_6model_comparison.png                   # Main interpretability figure
├── leg_interpretability_comparison.png         # Offline algorithms only
└── leg/[model]/
    ├── analysis_state_1.png                    # Per-state LEG analysis
    ├── saliency_state_1.png                    # Feature importance bars
    └── ...
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Gym-sepsis import error:**
```bash
ModuleNotFoundError: No module named 'gym_sepsis'
```
**Solution:** Install gym-sepsis from source (see Quick Start Step 1)

**2. d3rlpy version mismatch:**
```bash
AttributeError: 'DiscreteCQL' object has no attribute 'fit_online'
```
**Solution:** Upgrade d3rlpy: `pip install d3rlpy>=2.0.0`

**3. NumPy 2.0 compatibility:**
```bash
AttributeError: module 'numpy' has no attribute 'float'
```
**Solution:** Downgrade NumPy: `pip install "numpy<2.0"`

**4. CUDA out of memory:**
```bash
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or use CPU: `device='cpu'`

---

## 📧 Contact & Support

**Authors:**
- Zhiyu Cheng (zhiyu.cheng@email.gwu.edu)
- Yalun Ding
- Chuanhui Peng

**Institution:** George Washington University, Department of Statistics

**Course:** STAT 8289 - Reinforcement Learning (Fall 2025)

**Issues:** Please open a GitHub issue or contact the corresponding author.

---

## 📄 License & Citation

### Data
- **MIMIC-III:** Publicly available with PhysioNet credentialed access
- **gym-sepsis:** MIT License

### Code
- **Our Implementation:** MIT License (free for academic/research use)

### Citation
If you use this code, please cite:

```bibtex
@article{cheng2025sepsis_interpretability,
  title={Performance and Interpretability Trade-offs in Reinforcement Learning for Sepsis Treatment: Comparing Offline and Online Approaches},
  author={Cheng, Zhiyu and Ding, Yalun and Peng, Chuanhui},
  journal={AI in Neurology},
  year={2025},
  note={Inaugural Issue}
}
```

---

## 🙏 Acknowledgments

- **MIMIC-III Database:** Johnson et al. (2016)
- **gym-sepsis Environment:** Raghu et al. (2017)
- **d3rlpy Library:** Seno & Imai
- **Stable-Baselines3:** Raffin et al.
- **GWU STAT 8289 Course:** Reinforcement Learning instruction

---

**Ready for reproducible research!** 🚀
