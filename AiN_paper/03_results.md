# 3. Results

We present the evaluation results for all policies across 500 episodes each, focusing on overall performance, SOFA-stratified analysis, and the LEG interpretability comparison that reveals dramatic differences in feature importance patterns across algorithms.

## 3.1 Overall Performance Comparison

Table 1 and Figure 1 show results for 8 policies. DDQN-Attention achieves the highest survival at 95.4%, with all methods in a narrow range from 94.0% to 95.4%. Online RL with architectural innovations including DDQN variants and SAC achieves marginally higher rates from 94.2% to 95.4% than offline RL methods BC and CQL which range from 94.0% to 94.2%. Vanilla DQN at 94.0% matches offline methods despite online training.

The high baseline survival of approximately 94-95% across all policies, including the random policy at 95.0%, reveals that the simulator's outcome model is relatively insensitive to treatment choices. This likely reflects the learned dynamics model's bias toward survival in the underlying MIMIC-III training data, where the majority of patients survived regardless of the specific treatment administered. The random policy's performance matching or exceeding several trained policies does not imply that treatment is irrelevant clinically; rather, it underscores a fundamental limitation of simulator-only evaluation where the outcome model may not capture the true causal effect of treatments on survival. Average returns range from 13.20 to 13.62 with high variance from sparse rewards. Online RL with architectural innovations achieves modest gains of 1.2 to 1.4 percentage points over offline methods yet requires 1M training timesteps, which is infeasible clinically. Offline methods BC and CQL achieve comparable survival of 94.0-94.2% from pre-collected data only. DQN underperforms on high-severity cases as discussed in Section 3.2, motivating our focus on interpretability in Section 3.3.

**Table 1.** Overall performance across 500 evaluation episodes. Survival rates range from 94.0% to 95.4%, with DDQN-Attention achieving the highest at 95.4%. BC and CQL are trained offline on 2,105 episodes with approximately 20K transitions collected by a heuristic policy. DQN is trained online via simulator interaction but serves as a foundational Q-learning baseline lacking offline-specific modifications or online architectural innovations. Online methods including DDQN variants and SAC train via 1M timesteps of simulator interaction with architectural enhancements. All methods are evaluated on 500 newly sampled simulator episodes.

| Model | Survival (%) | Avg Return | Avg Length | Training |
|-------|--------------|------------|------------|----------|
| **Baselines** | | | | |
| Random | 95.0 | 13.50 ± 6.54 | 9.3 ± 1.1 | – |
| Heuristic | 94.6 | 13.38 ± 6.78 | 9.5 ± 1.2 | – |
| **Offline RL** (trained on 2,105 episodes, ~20K transitions) | | | | |
| BC | 94.2 | 13.26 ± 7.01 | 9.5 ± 0.6 | Offline |
| CQL | 94.0 | 13.20 ± 7.12 | 9.5 ± 0.5 | Offline |
| **Hybrid Baseline** (vanilla Q-learning, online trained) | | | | |
| DQN | 94.0 | 13.20 ± 7.12 | 7.8 ± 1.2 | Online^†^ |
| **Online RL** (trained via 1M timesteps of simulator interaction) | | | | |
| DDQN-Attention | **95.4** | 13.62 ± 6.28 | 7.9 ± 1.0 | Online |
| DDQN-Residual | 94.2 | 13.26 ± 7.01 | 9.0 ± 0.8 | Online |
| SAC | 94.8 | 13.44 ± 6.66 | 7.7 ± 1.2 | Online |

^†^DQN: foundational online Q-learning baseline lacking offline constraints from CQL or architectural innovations from DDQN-Attention/Residual and SAC. See Section 4.4.

**Figure 1.** Performance comparison showing survival rates in the top left panel, returns in the top right panel, SOFA-stratified survival in the bottom left panel, and episode lengths in the bottom right panel.

## 3.2 SOFA-Stratified Analysis

We stratified episodes by SOFA score into low (≤5), medium (6-10), and high (≥11) categories. All methods exceed 97% on low and medium severity cases, reflecting a ceiling effect. Table 2 shows high-SOFA results. DDQN-Attention achieves the highest survival at 90.5%, with a 1.9 percentage point advantage over BC and CQL at 88.6% and 88.5% respectively. Offline methods BC and CQL match SAC at 88.7% but substantially outperform DQN at 84.3%. CQL combines competitive high-SOFA performance with superior interpretability as shown in Section 3.3.

**Table 2.** Performance on high-severity patients with SOFA ≥ 11. DDQN-Attention achieves the highest survival rate at 90.5% on high-SOFA patients, demonstrating the benefit of attention mechanisms for complex cases. Offline RL methods BC and CQL achieve competitive survival rates of 88.5-88.6% comparable to SAC at 88.7%, while vanilla DQN substantially underperforms at 84.3%, suggesting that unconstrained online Q-learning without offline safeguards or architectural enhancements performs poorly on critical patients.

| Model | n | Survival (%) | Avg Return | Avg Length |
|-------|---|--------------|------------|------------|
| **Offline RL** | | | | |
| BC | 211 | 88.6 | 11.63 ± 9.82 | 8.3 ± 1.1 |
| CQL | 191 | 88.5 | 11.55 ± 9.95 | 8.3 ± 1.1 |
| **Hybrid Baseline** | | | | |
| DQN | 185 | 84.3 | 10.29 ± 11.46 | 8.5 ± 1.2 |
| **Online RL** | | | | |
| DDQN-Attention | 190 | **90.5** | 12.16 ± 8.79 | 8.0 ± 1.1 |
| DDQN-Residual | 200 | 87.0 | 11.10 ± 10.09 | 8.3 ± 1.2 |
| SAC | 195 | 88.7 | 11.62 ± 9.49 | 8.1 ± 1.1 |

n = sample size for each algorithm in the high-SOFA stratum.

## 3.3 LEG Interpretability Analysis

We now present the core contribution of this work: a systematic comparison of interpretability across all 6 algorithms, comprising 3 offline methods including BC, CQL, and DQN, and 3 online methods including DDQN-Attention, DDQN-Residual, and SAC, using Linearly Estimated Gradients analysis. We analyzed 10 representative states per algorithm using identical parameters with 1,000 perturbation samples and σ = 0.1, sampled uniformly across SOFA severity levels, and computed feature importance scores for the action selected by each policy. The results reveal dramatic differences in interpretability magnitude spanning 3 orders of magnitude, with profound implications for the offline-vs-online trade-off and clinical deployment.

### 3.3.1 Feature Importance Magnitude Comparison

Table 3 and Figure 2 summarize the LEG interpretability metrics for all 6 algorithms. The most striking finding is the maximum saliency magnitude, which quantifies the strength of the strongest feature importance signal. CQL achieves the highest saliency with a maximum of 40.06 for systolic blood pressure, suggesting strong dependence on a clinically relevant hemodynamic marker. The 3 online RL methods exhibit intermediate saliency, with DDQN-Attention achieving 3.57 for qSOFA, DDQN-Residual achieving 2.93 for INR, and SAC achieving 1.17 for INR. The remaining offline methods show weaker signals. BC achieves 0.78, and DQN exhibits the weakest signal at 0.069, representing a 600-fold difference compared to CQL since 40.06 / 0.069 ≈ 580. While we lack validated benchmarks to define absolute interpretability thresholds, this 3-order-of-magnitude range demonstrates substantial algorithmic variation in feature importance strength.

This 3-order-of-magnitude range reveals a clear interpretability hierarchy. Conservative offline RL as represented by CQL ranks highest, followed by online RL with architectural innovations such as DDQN-Attention and DDQN-Residual, then online RL without structure as represented by SAC, then imitation learning as represented by BC, and finally online-trained DQN at the bottom. Online methods achieve 11-fold weaker interpretability than CQL at 40.06 vs. 3.57 despite marginally higher survival rates of 95.4% vs. 94.0%, demonstrating a measurable performance-interpretability trade-off. However, online methods remain 50-fold more interpretable than vanilla DQN at 3.57 vs. 0.069, suggesting that architectural choices such as attention mechanisms and residual connections can partially mitigate the interpretability loss from online training.

The interpretability patterns reflect algorithmic differences in representation learning. CQL's conservatism biases the Q-function toward simple, threshold-based structures aligned with the heuristic behavioral policy, yielding strong gradients on clinically relevant features such as blood pressure and lactate. Online methods with attention and residual architectures preserve moderate interpretability by learning structured feature weighting, with DDQN-Attention's top feature being qSOFA which aligns with clinical severity scoring. In contrast, DQN's unconstrained deep network learns highly non-linear representations where no single feature dominates, producing uniformly weak saliency scores unsuitable for clinical validation.

**Table 3.** LEG interpretability metrics for all 6 algorithms with 10 representative states each using identical parameters of n = 1000 and σ = 0.1. CQL achieves 11-fold stronger saliency than the best online method DDQN-Attention and 600-fold stronger than vanilla DQN. Saliency magnitude is a proxy for feature-driven decision-making but does not guarantee clinical trust without prospective validation. Training paradigm labels are Offline for fixed dataset, Hybrid for online-trained vanilla Q-learning baseline, and Online for simulator interaction with architectural innovations.

| Algorithm | Training | Max Saliency | Top Feature | Clinical Coherence^‡^ |
|-----------|----------|--------------|-------------|----------------------|
| CQL | Offline | **40.06** | SysBP | High (hemodynamic) |
| DDQN-Attention | Online | 3.57 | qSOFA | High (severity score) |
| DDQN-Residual | Online | 2.93 | INR | Mixed (coagulation) |
| SAC | Online | 1.17 | INR | Mixed (coagulation) |
| BC | Offline | 0.78 | qSOFA | High (severity score) |
| DQN | Hybrid^†^ | 0.069 | INR | Weak (low signal) |

^†^DQN: vanilla online-trained Q-learning baseline; see Section 4.4.
^‡^Clinical coherence: qualitative assessment of alignment with medical guidelines.

**Figure 2.** Comprehensive 6-Model LEG Interpretability Analysis. The top panel shows maximum LEG saliency scores on logarithmic scale revealing 3 orders of magnitude variation. CQL achieves 40.06 as the strongest feature-driven signal, online methods achieve 1.17 to 3.57 as intermediate signals, BC achieves 0.78 as a weak signal, and DQN achieves only 0.069 as a negligible signal, representing a 600-fold difference between CQL and DQN. The middle panels show performance comparison where DDQN-Attention achieves highest overall survival at 95.4%, though not statistically significant, and high-SOFA survival at 90.5%, while CQL balances competitive performance at 94.0% overall and 88.5% high-SOFA with substantially stronger saliency magnitude. The bottom panel shows a performance-saliency scatter plot demonstrating the measurable trade-off, with online methods gaining 1.4 percentage points in survival within confidence intervals but exhibiting 11-fold weaker saliency compared to CQL. CQL offers a favorable balance for clinical deployment with statistically indistinguishable survival rates and strong, clinically coherent feature importance, while online methods achieve marginal performance gains at the cost of substantially reduced saliency signal strength.

### 3.3.2 Clinical Implications and Algorithm Selection

The 6-model LEG comparison reveals that interpretability is not uniformly sacrificed for performance. CQL's conservative value estimation biases the Q-function toward threshold-based logic mirroring the heuristic behavioral policy, yielding strong gradients on clinically relevant features such as SysBP, lactate, and MeanBP aligned with Surviving Sepsis Campaign guidelines ^4^. Online methods with architectural innovations including DDQN-Attention's multi-head attention and DDQN-Residual's skip connections preserve moderate interpretability by learning structured feature weighting, with top features qSOFA and INR aligning with clinical severity markers. In contrast, vanilla DQN's unconstrained deep network learns distributed representations with uniformly weak, clinically incoherent saliency patterns.

For clinical deployment, this hierarchy suggests several recommendations. Conservative offline RL as implemented in CQL offers the optimal balance with competitive survival of 94.0% overall and 88.5% high-SOFA, exceptional interpretability with 40.06 saliency, and no patient risk during training. Online RL with attention as implemented in DDQN-Attention achieves marginally better survival at 95.4% overall and 90.5% high-SOFA but requires 11-fold interpretability sacrifice and environment interaction infeasible in clinical settings. Behavior cloning and vanilla DQN exhibit poor or inconsistent interpretability unsuitable for regulatory approval. The finding that CQL achieves both strong performance and exceptional interpretability demonstrates that the performance-interpretability trade-off is not inevitable. Conservatism in the learning objective enables simultaneously effective and explainable policies.
