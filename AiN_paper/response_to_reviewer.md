# Response to Reviewer

---

Dear Editor and Reviewer,

We sincerely thank the reviewer for the positive assessment of our manuscript and the constructive suggestion regarding generative artificial intelligence. We appreciate the opportunity to clarify our research scope and address this comment.

---

## Reviewer's Comment

*"The recent rise of generative artificial intelligence requests its due attention. Hence, I would like to ask the authors to conduct additional analysis based on generative artificial intelligence..."*

---

## Our Response

We thank the reviewer for this thoughtful suggestion and the comprehensive introduction to generative AI methods including GANs and Transformers. We appreciate the reviewer's perspective on the importance of these emerging technologies.

**Clarification of Research Scope**

Our study focuses specifically on **reinforcement learning (RL) for sequential clinical decision-making** in sepsis treatment, comparing traditional offline RL methods (Behavior Cloning, Conservative Q-Learning) with online RL methods (DQN, DDQN-Attention, DDQN-Residual, SAC). The core contribution is a systematic **interpretability benchmark** using Linearly Estimated Gradients (LEG) analysis, demonstrating that algorithm selection profoundly impacts policy transparency—a critical requirement for regulatory approval and clinical trust.

**Distinction from Generative AI**

While generative AI methods (GANs and Transformers) have achieved remarkable success in image generation and natural language processing, they address fundamentally different problems from our sequential decision-making framework:

| Aspect | Our Study (RL for Sepsis) | Generative AI (GAN/Transformer) |
|--------|---------------------------|--------------------------------|
| **Task** | Sequential treatment decisions over time | Data generation or classification |
| **Output** | Action policy (which treatment to administer) | Generated images/text or class labels |
| **Objective** | Maximize long-term patient survival | Generate realistic samples or predict categories |
| **Key Challenge** | Distributional shift, interpretability | Mode collapse, training stability |

**Relevant Transformer-Based RL Methods**

We acknowledge that Transformer architectures have been adapted for reinforcement learning. Notably, **Decision Transformer** (Chen et al., 2021) reformulates RL as a sequence modeling problem, treating states, actions, and returns as tokens. We have cited this work in our manuscript (Reference 34) and discussed it in Section 4 (Discussion) as a promising future direction:

> *"Our study focuses on 3 specific offline RL algorithms including BC, CQL, and DQN and does not explore other promising methods such as Implicit Q-Learning (Kostrikov et al. 2022), Decision Transformer (Chen et al. 2021), or model-based offline RL. These methods may offer different performance-interpretability trade-offs, and future work should extend our LEG analysis framework to a broader set of algorithms."*

**Proposed Revision**

To address the reviewer's suggestion, we propose adding the following paragraph to Section 4.5 (Future Directions) to more explicitly acknowledge the potential of Transformer-based methods:

> *"Additionally, Transformer-based reinforcement learning methods such as Decision Transformer represent an emerging paradigm that reformulates sequential decision-making as sequence modeling. By treating trajectories as sequences of states, actions, and returns, these methods leverage the powerful attention mechanisms originally developed for natural language processing. Future work should investigate whether Transformer-based RL can achieve both strong performance and interpretability in clinical decision support, potentially combining the sequence modeling capabilities of Transformers with the conservative value estimation principles that make CQL interpretable."*

**Why GANs Are Less Applicable**

Generative Adversarial Networks, while powerful for image synthesis, are less directly applicable to our clinical decision-making problem because:

1. **Our task is policy learning, not data generation**: We aim to learn *which treatment actions to take*, not to generate synthetic patient data.

2. **Interpretability requirements differ**: GANs optimize for realistic sample generation, whereas clinical AI requires feature-level explanations of *why* specific treatments are recommended.

3. **Sequential nature of treatment**: Sepsis management involves multi-step decision sequences over ICU stays, which is naturally modeled by RL's Markov Decision Process framework rather than GAN's single-shot generation paradigm.

---

## Summary

We respectfully submit that incorporating full GAN or standard Transformer analyses would extend beyond the scope of our current study, which provides the first systematic LEG-based interpretability benchmark for offline vs. online RL in sepsis treatment. However, we have:

1. **Already cited** Decision Transformer (Chen et al., 2021) as a relevant Transformer-based RL method
2. **Proposed adding** an expanded discussion of Transformer-based RL in Future Directions
3. **Clarified** the distinction between generative AI tasks and our sequential decision-making framework

We hope this response adequately addresses the reviewer's concern. We remain open to further suggestions and are committed to improving the manuscript.

---

Sincerely,

The Authors
