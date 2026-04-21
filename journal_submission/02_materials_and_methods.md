# 2. Materials and Methods

We formulate sepsis treatment as a finite-horizon Markov Decision Process (MDP) in the offline reinforcement learning setting, where the goal is to learn an optimal policy from a fixed dataset without further environment interaction. We describe the experimental setup including the simulation environment, reinforcement learning algorithms, interpretability analysis, and evaluation protocol.

---

## 2.1 Materials

### Databases
- **MIMIC-III Database**: Medical Information Mart for Intensive Care III, Beth Israel Deaconess Medical Center, Boston, MA, USA.^4^

### Simulation Environment
- **Gym-Sepsis Simulator**: Version 1.0, available at https://github.com/gefeilin/gym-sepsis, originally developed by Raghu et al.^1^

### Software and Libraries
- **Python**: Version 3.9, Python Software Foundation, Wilmington, DE, USA.
- **PyTorch**: Version 1.12, Facebook AI Research (Meta), Menlo Park, CA, USA.
- **d3rlpy**: Version 1.1.1, Takuma Seno, GitHub, Japan.
- **Stable-Baselines3**: Version 1.6.0, DLR-RM, German Aerospace Center, Cologne, Germany.
- **NumPy**: Version 1.21, NumFOCUS, Austin, TX, USA.
- **Pandas**: Version 1.4, NumFOCUS, Austin, TX, USA.

### Hardware
- All experiments were conducted on a workstation with NVIDIA RTX 3080 GPU (NVIDIA Corporation, Santa Clara, CA, USA) and Intel Core i9-12900K CPU (Intel Corporation, Santa Clara, CA, USA).

---

## 2.2 MDP Formulation

The sepsis treatment MDP is defined by the tuple M = (S, A, P, R, γ). The state space S ⊂ ℝ^46 captures patient physiological condition through laboratory values, vital signs, and clinical severity scores. The action space A contains 24 discrete actions indexed from 0 to 23, representing a 5×5 grid of IV fluid and vasopressor dosing levels. The gym-sepsis environment implements the mapping:

> a = min(5 × IV_bin + VP_bin, 23)     (1)

effectively merging the highest dosage combination into action 23. The transition dynamics P(s_{t+1}|s_t, a_t) are learned from MIMIC-III data via the gym-sepsis simulator.^1^ The reward function uses sparse terminal rewards: R(s_T, a_T) = +15 for survival, −15 for death, and 0 for intermediate steps. We use discount factor γ = 0.99.

A policy π: S → Δ(A) maps states to action distributions. The goal is to find π* = arg max_π V^π(s) where the value function is:

> V^π(s) = E_{τ~π,P} [Σ_{t=0}^{H-1} γ^t R(s_t, a_t) | s_0 = s]     (2)

The action-value function Q^π(s, a) represents expected return when taking action a in state s and following π thereafter. The optimal policy is derived via π*(s) = arg max_a Q*(s, a).

In offline RL, the agent learns exclusively from a fixed dataset D = {(s_i, a_i, r_i, s'_i)}_{i=1}^N collected under a behavioral policy, without environment interaction during training.^2^ The central challenge is distributional shift, where the learned policy π may select out-of-distribution actions with unreliable Q-value estimates due to extrapolation error.^3^

---

## 2.3 Environment and Data

### 2.3.1 Gym-Sepsis Simulator

We use Gym-Sepsis,^1^ an RL simulator for ICU sepsis treatment trained on MIMIC-III data.^4^

At each timestep, the state is a 46-dimensional vector spanning:
- **Laboratory values**: lactate, creatinine, platelet count, bilirubin, INR
- **Vital signs**: systolic blood pressure, mean arterial pressure, heart rate, SpO₂
- **Demographics**: age, gender, race
- **Clinical severity scores**: SOFA, LODS, SIRS, qSOFA, Elixhauser
- **Treatment status**: mechanical ventilation, blood culture

The SOFA score^5^ ranges from 0 to 24, with higher values indicating greater organ dysfunction.

The action space is discrete with 24 actions indexed from 0 to 23, derived from a 5×5 grid over IV fluid and vasopressor dosage bins. Episodes span ICU stays with 4-hour timesteps until discharge or death. We use sparse rewards: r_t = +15 for survival, −15 for death, and 0 for intermediate steps.

### 2.3.2 Offline Training Dataset

We generated an offline dataset of 2,105 episodes with approximately 20K transitions using a heuristic policy based on clinical guidelines,^6,7^ achieving 94.6% survival. All episodes were used for training without held-out validation or test sets from the offline data. Performance evaluation was conducted by running trained policies on 500 newly sampled episodes in the gym-sepsis simulator.

---

## 2.4 Algorithms

We compare 3 offline RL algorithms representing different learning paradigms: Behavior Cloning for supervised learning, Conservative Q-Learning for offline Q-learning, and Deep Q-Network as online RL adapted for offline evaluation. We also evaluate 3 online RL algorithms with architectural innovations. All algorithms use the same neural network architecture for fair comparison: a 3-layer multilayer perceptron with hidden dimensions of 256, 256, and 128 with ReLU activations.

### 2.4.1 Offline RL Algorithms

**Behavior Cloning (BC).** Behavior Cloning treats offline RL as a supervised learning problem, training a policy to imitate the behavioral policy by minimizing the negative log-likelihood of observed actions.^8^ Given a dataset D = {(s_i, a_i)}_{i=1}^N of state-action pairs, BC learns a policy π_θ(a|s) by solving:

> θ* = arg min_θ −(1/N) Σ_{i=1}^N log π_θ(a_i|s_i)     (3)

We use d3rlpy's DiscreteBCConfig with batch size 1,024, learning rate 1×10⁻³ with Adam optimizer, training for 50,000 gradient steps over 10 epochs.

**Conservative Q-Learning (CQL).** Conservative Q-Learning^9^ is an offline RL algorithm that learns a conservative Q-function to avoid overestimation on out-of-distribution actions. CQL augments the standard Bellman error with a conservatism penalty:

> min_Q α·E_{s~D}[log Σ_a exp Q(s,a) − E_{a~π_β} Q(s,a)] + (1/2)E_{(s,a,r,s')~D}[(Q(s,a) − T^π Q(s,a))²]     (4)

where α controls the strength of the conservatism penalty, π_β is the behavioral policy, and T^π is the Bellman operator. We use d3rlpy's DiscreteCQLConfig with batch size 1,024, learning rate 3×10⁻⁴ with Adam optimizer, α = 1.0, target network updates every 2,000 steps, training for 200,000 gradient steps.

**Deep Q-Network (DQN).** Deep Q-Network^10^ combines Q-learning with deep neural networks using experience replay and target networks for stability. The Q-function is updated to minimize the temporal difference error:

> L(θ) = E_{(s,a,r,s')~D}[(Q_θ(s,a) − (r + γ max_{a'} Q_{θ⁻}(s',a')))²]     (5)

We use Stable-Baselines3 with batch size 256, learning rate 1×10⁻⁴, target network updates every 1,000 steps, ε-greedy exploration from 1.0 to 0.05, training for 100,000 timesteps.

### 2.4.2 Online RL Algorithms

To provide a comprehensive comparison between offline and online RL paradigms, we also evaluate 3 state-of-the-art online RL algorithms with architectural innovations. These algorithms train by interacting with the Gym-Sepsis simulator, collecting 1 million timesteps of experience through exploration.

**Double DQN with Attention (DDQN-Attention).** This algorithm extends Double DQN^11^ with a multi-head self-attention mechanism in the encoder network:

> h_t = MultiHeadAttention(s_t, s_t, s_t) + s_t     (6)

The attention mechanism computes scaled dot-product attention across 4 parallel heads, each learning different feature correlations.

**Double DQN with Residual Connections (DDQN-Residual).** This variant incorporates deep residual networks^12^ with skip connections between layers:

> h_{l+1} = σ(LayerNorm(W_l h_l + b_l + h_l))     (7)

where σ is the ReLU activation, W_l and b_l are learnable weights and biases.

**Soft Actor-Critic (SAC).** SAC^13^ is a maximum entropy RL algorithm that optimizes both expected return and policy entropy:

> J(π) = E_{τ~π}[Σ_t r(s_t, a_t) + α H(π(·|s_t))]     (8)

where H(π(·|s)) is the entropy of the policy at state s, and α is a temperature parameter. We use the discrete action space variant with a residual encoder architecture.

**Training Details.** All 3 online RL algorithms were trained with 1,000,000 environment interaction steps using experience replay buffers of size 100,000. Training used batch size 256, learning rate 3×10⁻⁴ with Adam optimizer, and target network soft updates with τ = 0.005.

---

## 2.5 LEG Interpretability Analysis

To assess interpretability, we employ Linearly Estimated Gradients (LEG),^14^ a model-agnostic perturbation-based method for computing feature importance in RL policies. LEG approximates the policy gradient ∇_s Q(s_0, π(s_0)) with respect to state features by locally linearizing the Q-function around a reference state s_0.

### 2.5.1 LEG Algorithm

Given a state s_0 ∈ ℝ^46 and a learned policy π derived from Q-function Q(s,a) as π(s) = arg max_a Q(s,a), LEG estimates the saliency vector γ̂ ∈ ℝ^46 through the following procedure.

For perturbation sampling, we generate n = 1,000 random perturbations Z_i ~ N(0, σ²I) with σ = 0.1, producing perturbed states:

> s_i = s_0 + Z_i,  i = 1, ..., n     (9)

For Q-value evaluation, we compute the Q-value change for each perturbed state s_i and the selected action a_0 = π(s_0):

> ŷ_i = Q(s_i, a_0) − Q(s_0, a_0)     (10)

For ridge regression, we compute the sample covariance matrix of perturbations:

> Σ = (1/n) Z^T Z + λI,  λ = 10⁻⁶     (11)

For gradient estimation, we estimate the saliency vector via the closed-form ridge regression solution:

> γ̂ = Σ⁻¹((1/n) Z^T ŷ)     (12)

where ŷ = [ŷ_1, ..., ŷ_n]^T. Each element γ̂_j quantifies the marginal contribution of feature j to the Q-value.

### 2.5.2 State Selection and Analysis Protocol

We apply LEG to all 6 algorithms using identical parameters for fair comparison. For each algorithm, we analyze 10 representative states sampled from the gym-sepsis environment, ensuring coverage across SOFA severity levels: low (SOFA ≤ 5), medium (6-10), and high (≥ 11).

### 2.5.3 Interpretability Metrics

We quantify interpretability using 3 complementary metrics:
- **Maximum saliency magnitude**: max_j |γ̂_j|, measuring whether any single feature exerts dominant influence on the Q-function.
- **Saliency range**: difference between maximum and minimum saliency scores, indicating how concentrated importance is across features.
- **Clinical coherence**: qualitative assessment of whether top-ranked features align with established medical knowledge, comparing LEG-identified important features against Surviving Sepsis Campaign guidelines.^6^

---

## 2.6 Evaluation Metrics

We evaluate algorithm performance using outcome-based and severity-stratified metrics.

### 2.6.1 Primary Outcome Metrics

**Survival rate** is the proportion of evaluation episodes ending in patient discharge rather than mortality:

> Survival Rate = (1/N) Σ_{i=1}^N 𝟙[R_i > 0]     (13)

where N is the number of evaluation episodes and R_i = Σ_t r_{i,t} is the cumulative return for episode i.

**Average return** is the mean cumulative reward across all episodes:

> R̄ = (1/N) Σ_{i=1}^N Σ_{t=1}^{T_i} r_{i,t}     (14)

**Average episode length** is the mean number of timesteps per episode:

> T̄ = (1/N) Σ_{i=1}^N T_i     (15)

### 2.6.2 SOFA-Stratified Analysis

To assess algorithm performance across patient severity levels, we stratify evaluation episodes by initial Sequential Organ Failure Assessment score.^5^ We define 3 severity strata:
- **Low SOFA**: s_0^SOFA ≤ 5
- **Medium SOFA**: 6 ≤ s_0^SOFA ≤ 10
- **High SOFA**: s_0^SOFA ≥ 11

---

## 2.7 Baseline Policies

To contextualize RL algorithm performance, we evaluate 2 baseline policies:
- **Random policy**: selects actions uniformly at random from the 24-action space at each timestep, with π(a|s) = 1/24.
- **Heuristic policy**: implements threshold-based rules from sepsis guidelines^6^: escalates IV fluids when SysBP < 100 mmHg or lactate > 2.0 mmol/L, and escalates vasopressors when MeanBP < 65 mmHg.

---

## 2.8 Statistical Analysis

All policies were evaluated on N = 500 episodes in the Gym-Sepsis simulator using a standardized protocol. Each episode begins with stochastic initialization sampling an initial patient state from the MIMIC-III-derived distribution. Policies are evaluated deterministically without exploration noise. For discrete policies, we select a_t = arg max_a Q(s_t, a). For SAC, we use the mean action.

Episodes terminate when the simulator predicts patient discharge or death, or after 50 timesteps (approximately 8.3 days), whichever occurs first. All results are reported as mean ± standard deviation. Given the sample size of N = 500, performance differences exceeding approximately ±2% for survival rates likely reflect genuine algorithmic differences rather than sampling variability.

---

## References (for this section)

1. Raghu A, Komorowski M, Ahmed I, et al. Deep reinforcement learning for sepsis treatment. NeurIPS Workshop on Machine Learning for Health. 2017.
2. Levine S, Kumar A, Tucker G, et al. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint. 2020.
3. Fujimoto S, Meger D, Precup D. Off-policy deep reinforcement learning without exploration. ICML. 2019:2052-2062.
4. Johnson AE, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. Sci Data. 2016;3(1):1-9.
5. Vincent JL, Moreno R, Takala J, et al. The SOFA score to describe organ dysfunction/failure. Intensive Care Med. 1996;22(7):707-710.
6. Rhodes A, Evans LE, Alhazzani W, et al. Surviving sepsis campaign: international guidelines. Intensive Care Med. 2017;43(3):304-377.
7. Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for sepsis. JAMA. 2016;315(8):762-774.
8. Pomerleau DA. Efficient training of artificial neural networks for autonomous navigation. Neural Comput. 1991;3(1):88-97.
9. Kumar A, Zhou A, Tucker G, et al. Conservative Q-learning for offline reinforcement learning. NeurIPS. 2020;33:1179-1191.
10. Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning. Nature. 2015;518(7540):529-533.
11. Van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double Q-learning. AAAI. 2016;30.
12. He K, Zhang X, Ren S, et al. Deep residual learning for image recognition. CVPR. 2016:770-778.
13. Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning. ICML. 2018:1861-1870.
14. Greydanus S, Koul A, Dodge J, et al. Visualizing and understanding atari agents. ICML. 2018:1792-1801.
