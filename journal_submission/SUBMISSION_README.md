# Journal Submission Package

**Manuscript Title:** Performance and Interpretability Trade-offs in Reinforcement Learning for Sepsis Treatment: Comparing Offline and Online Approaches

**Target Journal:** AI in Neurology (Inaugural Issue)

**Submission Date:** January 2025

---

## 📁 Package Contents

### 1. **manuscript/** - Complete Manuscript Files
- `main.pdf` - Final compiled manuscript (27 pages)
- `main.tex` - LaTeX main file
- `references.bib` - Bibliography with 42 references (includes 7 new neurology references)
- `sections/` - All manuscript sections:
  - `01_introduction.tex` - Introduction with sepsis-neurology connection
  - `02_related.tex` - Related Work
  - `03_problem.tex` - Problem Formulation
  - `04_methods.tex` - Methods (3 offline + 3 online RL algorithms)
  - `05_results.tex` - Results (LEG interpretability analysis)
  - `06_discussion.tex` - Discussion (includes neurological impact section)
  - `07_conclusion.tex` - Conclusion
  - `08_contributions.tex` - Author Contributions
  - `09_appendix.tex` - Appendix

### 2. **figures/** - Main Figures
- `Fig1_algorithm_comparison.png` - Overall performance comparison (8 algorithms)
- `Fig2_leg_6model_comparison.png` - Comprehensive 6-model LEG interpretability analysis

### 3. **code/** - Reproducible Code
- `src/` - Source code library
  - `envs/` - Environment wrappers (gym-sepsis)
  - `evaluation/` - Evaluation metrics
  - `visualization/` - LEG analysis and plotting tools
  - `data/` - Data collection utilities
- `scripts/` - Experiment scripts
  - `01_baseline_evaluation.py` - Baseline policy evaluation
  - `02_train_bc.py` - Behavior Cloning training
  - `03_train_cql.py` - Conservative Q-Learning training
  - `04_train_dqn.py` - Deep Q-Network training
  - `06_visualization.py` - Results visualization
  - `07_final_analysis.py` - Comprehensive analysis

### 4. **supplementary/** - Supplementary Materials
- (Empty - can add detailed LEG visualizations if requested by reviewers)

---

## 🎯 Key Revisions for "AI in Neurology"

Based on editor feedback, we have made the following revisions to align with the journal's focus on AI and neurological disorders:

### Added Content:
1. **Introduction (Section 1, Page 3):**
   - New paragraph on sepsis-associated neurological consequences
   - Mechanisms: Systemic inflammation → BBB disruption → Neurodegeneration
   - Epidemiology: 30-70% SAE incidence, 62% increased dementia risk
   - Alzheimer's disease connection: Accelerated amyloid-β accumulation

2. **Discussion (Section 6.3, Page 19):**
   - New subsection: "Neurological Impact and Treatment Optimization"
   - Explains why interpretable RL policies are medically essential
   - Links acute sepsis treatment to long-term cognitive preservation

### New References (7 added):
- Barichello et al. 2021 (Tissue Barriers) - BBB dysfunction in sepsis
- Gofton & Young 2012 (Nat Rev Neurol) - Sepsis-associated encephalopathy
- Wang et al. 2024 (Brain-X) - Recent SAE advances
- Yang et al. 2022 (Front Aging Neurosci) - Dementia risk meta-analysis
- Yang et al. 2023 (Mol Psychiatry) - Sepsis & Alzheimer's pathophysiology
- Iwashyna et al. 2010 (JAMA) - Long-term cognitive impairment
- Annane & Sharshar 2015 (Intensive Care Med) - Sepsis-induced delirium

### Updated Keywords:
Added: Sepsis-Associated Encephalopathy, Neurological Outcomes, Blood-Brain Barrier, AI in Neurology

---

## 📊 Manuscript Statistics

- **Pages:** 27
- **Figures:** 2 main figures (+ 90+ supplementary LEG visualizations available)
- **Tables:** 3 (overall performance, SOFA-stratified, interpretability metrics)
- **References:** 42 (interdisciplinary: RL, Clinical Medicine, Neurology)
- **Word Count:** ~9,500 words

---

## 🔬 Technical Specifications

### Algorithms Evaluated:
- **Offline RL:** Behavior Cloning (BC), Conservative Q-Learning (CQL), Deep Q-Network (DQN)
- **Online RL:** Double DQN with Attention, Double DQN with Residual, Soft Actor-Critic (SAC)
- **Baselines:** Random Policy, Heuristic Policy

### Environment:
- **Simulator:** gym-sepsis (MIMIC-III-derived)
- **State Space:** 46-dimensional (vital signs, lab values, SOFA scores)
- **Action Space:** 25 discrete actions (IV fluid × vasopressor dosing)
- **Evaluation:** 500 episodes per algorithm

### Key Findings:
- CQL achieves 600-fold stronger interpretability than DQN (max saliency: 40.06 vs 0.069)
- Online RL gains 1.4pp survival advantage but loses 11-fold interpretability
- CQL identifies clinically coherent features (blood pressure, lactate)

---

## 🚀 Reproducibility

All experiments can be reproduced using the provided code:

```bash
# Install dependencies
pip install -r code/requirements.txt

# Run full pipeline
python code/scripts/01_baseline_evaluation.py
python code/scripts/02_train_bc.py
python code/scripts/03_train_cql.py
python code/scripts/04_train_dqn.py
python code/scripts/07_final_analysis.py
```

---

## 📧 Contact Information

**Corresponding Author:** Zhiyu Cheng
**Email:** zhiyu.cheng@email.gwu.edu
**Institution:** Department of Statistics, George Washington University

**Co-authors:**
- Yalun Ding (GWU Statistics)
- Chuanhui Peng (GWU Statistics)

**Equal Contribution:** All authors contributed equally to this work.

---

## ✅ Submission Checklist

- [x] Manuscript PDF compiled successfully
- [x] All figures in high resolution (>300 DPI)
- [x] References formatted correctly (AGSM style)
- [x] Author contributions statement included
- [x] Conflict of interest statement included
- [x] Data availability statement included
- [x] Code provided for reproducibility
- [x] Sepsis-neurology connection emphasized per editor request
- [x] Keywords updated to match journal scope

---

**Ready for submission to AI in Neurology (Inaugural Issue)** ✅
