# Response to Reviewer 2

---

Dear Editor and Reviewer,

We sincerely thank Reviewer 2 for the thorough and constructive evaluation of our manuscript. The detailed comments have helped us significantly improve the clarity and completeness of our work. Below we address each comment point by point.

**Note**: The revised manuscript is submitted as a Word document (.docx) as required by the journal.

---

## Comment 1: DQN Classification Inconsistency

**Response**: We thank the reviewer for identifying this concern. Upon review, Section 2.3.3 of our manuscript already contains a detailed explanation of DQN's hybrid methodological status. To further clarify, we have added a footnote to Table 1 emphasizing that DQN is trained via online simulator interaction (hybrid paradigm) but lacks both CQL's offline conservatism constraints and the architectural innovations of DDQN-Attention/Residual/SAC.

---

## Comment 2: Feature Saliency Visualizations

**Response**: We appreciate this suggestion. Our manuscript already includes Figure 2, which presents a comprehensive 6-model LEG interpretability comparison with maximum LEG saliency scores, performance comparisons, and a performance-saliency scatter plot. Table 3 also provides detailed top feature rankings for each algorithm. We believe these existing visualizations adequately address the reviewer's concern.

---

## Comment 3: Training Convergence Curves

**Response**: We agree that training dynamics provide important insight into algorithm behavior. We have added a new **Figure 3** showing training convergence curves for all algorithms, including individual loss trajectories, smoothed curves, and normalized comparison panel. Corresponding text describing the convergence patterns has been added to Section 3.1.

---

## Comment 4: Residual Connections Rationale

**Response**: Section 2.3.4.2 already explains that residual connections in our three-layer network serve to preserve input feature information, improve sample efficiency, and enable feature reuse. We have added a new reference (Khan et al., 2024) on residual networks in medical AI to support this architectural choice.

---

## Comment 5: Formula (10) Clarification

**Response**: Section 2.4.1 already contains the complete derivation of the LEG saliency formula, showing how it is derived from the closed-form ridge regression solution. No additional changes were necessary.

---

## Comment 6: Indicator Function Definition

**Response**: The indicator function is already defined in Section 2.5.1, where we specify that survival is determined by the sign of the terminal reward. For additional clarity, we have added an explicit mathematical definition of the indicator function following Equation (11).

---

## Comment 7: Recent References

**Response**: We have added 8 new references from 2023-2024:

| Ref # | Authors | Topic |
|-------|---------|-------|
| 36 | Zhou et al. (2024) | Offline safe RL for healthcare |
| 37 | Bock et al. (2024) | Medical Decision Transformer for sepsis |
| 38 | Zhang et al. (2023) | Continuous-Time Decision Transformer |
| 39 | Tang et al. (2023) | RL with human expertise for sepsis |
| 40 | Nerella et al. (2024) | Transformer models in healthcare survey |
| 41 | Chen et al. (2024) | AI in sepsis management review |
| 42 | Chakraborty et al. (2024) | RL primer for clinicians |
| 43 | Khan et al. (2024) | Residual networks in medical AI |

Section 4.5 (Future Directions) has been expanded to discuss recent Transformer-based RL developments and safe offline RL frameworks.

---

## Summary of Revisions

| Comment | Action Taken |
|---------|--------------|
| 1. DQN classification | Added Table 1 footnote clarifying hybrid status |
| 2. Saliency visualization | Already addressed by existing Figure 2 & Table 3 |
| 3. Convergence curves | Added new Figure 3 with training dynamics |
| 4. Residual rationale | Added new reference (Khan 2024) |
| 5. Formula (10) | Already complete in Section 2.4.1 |
| 6. Indicator function | Added explicit definition |
| 7. Recent references | Added 8 references (2023-2024) |

---

We believe these revisions have addressed all of the reviewer's concerns. We are grateful for the opportunity to strengthen our work and remain open to any further suggestions.

Sincerely,

The Authors
