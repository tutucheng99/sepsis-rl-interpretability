# Saliency Visualization Guide for Reviewer 2 Comment 2

本文档说明如何使用现有的LEG saliency可视化图表。

---

## Available Figures

### 1. 6-Model LEG Comparison (RECOMMENDED)
**File**: `results/figures/leg_6model_comparison.png`
**Size**: 620 KB
**Content**:
- Top: LEG saliency scores bar chart for all 6 algorithms
- Middle: Performance comparison (Overall & High-Severity)
- Bottom: Performance-Interpretability trade-off scatter plot

**Recommendation**: 可以直接作为论文Figure使用

---

### 2. Overall Feature Importance Bar Chart
**File**: `github_models/sepsis_leg_analysis/overall_feature_importance.png`
**Content**: Top 20 features ranked by average LEG saliency across all models
**Key features identified**:
- INR (2.55)
- qSOFA_sysbp (2.24)
- qSOFA (2.12)
- elixhauser (2.01)
- SOFA (1.53)
- lactate (1.41)

**Recommendation**: 可作为补充Figure展示特征重要性细节

---

### 3. Model Comparison Heatmap
**File**: `github_models/sepsis_leg_analysis/model_comparison_heatmap.png`
**Content**: Heatmap comparing feature importance between models
**Note**: Currently only shows 2 models (DDQN-Attention, SAC)

**Recommendation**: 可能需要重新生成包含全部6模型的版本

---

### 4. Individual LEG Analysis (60 files)
**Location**: `results/figures/leg/`
- `bc_simple_reward/`: 10 saliency + 10 analysis images
- `cql_simple_reward/`: 10 saliency + 10 analysis images
- `dqn_simple_reward/`: 10 saliency + 10 analysis images

**Recommendation**: 选择代表性样本放入Supplementary Materials

---

## Figure Usage Recommendations

### Main Paper Figures

| Figure # | Content | Source File |
|----------|---------|-------------|
| Figure 3 | Training Convergence Curves | `revisions/figures/Figure_training_convergence.png` (NEW) |
| Figure 4 | 6-Model LEG Comparison | `results/figures/leg_6model_comparison.png` (EXISTING) |
| Figure 5 | Top 20 Feature Importance | `github_models/sepsis_leg_analysis/overall_feature_importance.png` (EXISTING) |

### Supplementary Materials

| Figure # | Content | Source File |
|----------|---------|-------------|
| S1 | BC Individual Saliency | `results/figures/leg/bc_simple_reward/saliency_state_1.png` |
| S2 | CQL Individual Saliency | `results/figures/leg/cql_simple_reward/saliency_state_1.png` |
| S3 | DQN Individual Saliency | `results/figures/leg/dqn_simple_reward/saliency_state_1.png` |

---

## Optional: Generate Enhanced 6-Model Heatmap

If you want to create a new comprehensive heatmap with all 6 models, the data exists in:
`github_models/sepsis_leg_analysis/multi_model_feature_importance.json`

A script to generate this can be created upon request.

---

## Figure Captions (Draft)

### Figure 3: Training Convergence Curves
> **Figure 3.** Training convergence curves for the six reinforcement learning algorithms. (a-d) Individual loss trajectories for BC, CQL, DQN, and SAC, with smoothed curves overlaid on raw data. Labels indicate offline vs. hybrid training paradigms. (e) Normalized loss comparison showing relative convergence rates across all algorithms. CQL demonstrates rapid initial convergence followed by stable learning, while BC exhibits gradual optimization of its imitation objective.

### Figure 4: LEG Interpretability Comparison
> **Figure 4.** Linearly Estimated Gradient (LEG) interpretability analysis across six RL algorithms for sepsis treatment. Top panel: Maximum LEG saliency scores indicating feature sensitivity, with CQL achieving the highest interpretability (40.06). Middle panels: Survival rate comparison for overall population and high-severity patients (SOFA ≥ 11). Bottom panel: Performance-interpretability trade-off, demonstrating that CQL achieves competitive survival rates while maintaining superior interpretability compared to hybrid approaches.

### Figure 5: Feature Importance Rankings
> **Figure 5.** Top 20 most important clinical features across all models as determined by average absolute LEG saliency scores. Key features include INR (international normalized ratio), qSOFA components (systolic blood pressure, respiratory rate, GCS), disease severity scores (SOFA, LODS, Elixhauser), and vital signs (lactate, temperature, ventilation status).

