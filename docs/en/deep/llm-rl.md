# RL Methods for LLM Training

## Interview Summary

RL methods for LLM training align language models with human preferences. **RLHF** uses a reward model + PPO to optimize responses. **TRPO** provides theoretical guarantees via constrained optimization but is complex. **PPO** simplifies TRPO with clipping. **DPO** eliminates the reward model by directly optimizing on preference pairs. **GRPO** uses group-based advantage estimation without a critic. Know the tradeoffs: PPO is flexible but requires a reward model; DPO is simpler but less flexible; GRPO balances both.

---

## Why RL for LLM Training?

### The Alignment Problem

Pre-trained LLMs learn to predict the next token, but this doesn't guarantee:
- Helpful, harmless, honest responses
- Following instructions accurately
- Avoiding harmful content

**Supervised Fine-Tuning (SFT)** alone is limited because:
1. Collecting expert demonstrations is expensive
2. Models can memorize without understanding intent
3. Hard to specify "good" responses for all situations

### The RLHF Pipeline

Reinforcement Learning from Human Feedback (RLHF) addresses this:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RLHF PIPELINE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Supervised Fine-Tuning (SFT)
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Base LLM     │ →  │ Fine-tune on    │ →  │ SFT Model    │
│ (pretrained) │    │ demonstrations  │    │ π_SFT        │
└──────────────┘    └─────────────────┘    └──────────────┘

Step 2: Reward Model Training
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Collect      │ →  │ Train to predict│ →  │ Reward Model │
│ preferences  │    │ human preference│    │ r_φ(x, y)    │
│ (y_w > y_l)  │    │                 │    │              │
└──────────────┘    └─────────────────┘    └──────────────┘

Step 3: RL Optimization
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ SFT Model    │ →  │ Optimize with   │ →  │ Aligned LLM  │
│ π_SFT        │    │ PPO/TRPO/etc    │    │ π_θ          │
└──────────────┘    └─────────────────┘    └──────────────┘
```

### The Core Objective

All methods aim to maximize expected reward while staying close to the reference policy:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{KL}\left(\pi_\theta \| \pi_{ref}\right)$$

Where:
- $\pi_\theta$: Policy being optimized (the LLM)
- $\pi_{ref}$: Reference policy (usually SFT model)
- $r_\phi(x, y)$: Reward model score for response $y$ to prompt $x$
- $\beta$: KL penalty coefficient (prevents reward hacking)

**Why the KL penalty?** Without it, the model can exploit reward model weaknesses, producing high-reward but nonsensical outputs (reward hacking).

---

## TRPO: Trust Region Policy Optimization

### The Foundation

TRPO (Schulman et al., 2015) provides the theoretical foundation. It asks: *How can we improve the policy while guaranteeing we don't make it worse?*

### Motivation: The Policy Improvement Problem

In standard policy gradient, large updates can catastrophically degrade performance:

$$\theta_{new} = \theta_{old} + \alpha \nabla_\theta J(\theta)$$

The problem: How do we choose $\alpha$? Too large → policy collapse. Too small → slow learning.

### The Surrogate Objective

TRPO optimizes a **surrogate objective** that lower-bounds the true improvement:

$$L^{CPI}(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t \right] = \mathbb{E}_t \left[ r_t(\theta) A_t \right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

### The Trust Region Constraint

TRPO constrains the KL divergence to ensure monotonic improvement:

$$\max_\theta \quad L^{CPI}(\theta)$$
$$\text{s.t.} \quad \mathbb{E}_t \left[ D_{KL}\left(\pi_{\theta_{old}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t)\right) \right] \leq \delta$$

**Key Insight**: Within this "trust region" (where $D_{KL} \leq \delta$), the surrogate objective accurately predicts the true objective.

### Mathematical Derivation

**Step 1: Performance Difference Lemma**

The difference between two policies can be expressed as:

$$J(\pi_{new}) - J(\pi_{old}) = \mathbb{E}_{s \sim d^{\pi_{new}}, a \sim \pi_{new}} \left[ A^{\pi_{old}}(s, a) \right]$$

Where $d^{\pi_{new}}$ is the state distribution under the new policy.

**Step 2: The Problem**

We can't sample from $d^{\pi_{new}}$ because we don't have the new policy yet!

**Step 3: The Approximation**

TRPO uses the old state distribution with importance sampling:

$$L^{CPI}(\theta) = \mathbb{E}_{s \sim d^{\pi_{old}}, a \sim \pi_{old}} \left[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A^{\pi_{old}}(s, a) \right]$$

**Step 4: Bounding the Error**

The error introduced by using the wrong state distribution is bounded when $D_{KL}$ is small:

$$|J(\pi_{new}) - J(\pi_{old}) - L^{CPI}(\theta)| \leq C \cdot \sqrt{\mathbb{E}_s[D_{KL}(\pi_{old} \| \pi_{new})]}$$

### TRPO Algorithm

```
TRPO Algorithm:
─────────────────────────────────────────────────────────
Input: Initial policy π_θ, constraint δ

for iteration = 1, 2, ... do:
    1. Collect trajectories using current policy π_θ

    2. Compute advantages A_t for all timesteps

    3. Compute policy gradient:
       g = ∇_θ L^CPI(θ)|_{θ=θ_old}

    4. Compute Fisher Information Matrix:
       F = ∇²_θ D_KL(π_θ_old || π_θ)|_{θ=θ_old}

    5. Compute natural gradient direction:
       d = F^{-1} g

    6. Compute step size via line search:
       Find largest β such that:
       - D_KL(π_θ_old || π_{θ_old + βd}) ≤ δ
       - L^CPI(θ_old + βd) > L^CPI(θ_old)

    7. Update: θ ← θ_old + βd
─────────────────────────────────────────────────────────
```

### Solving the Constrained Optimization

Using Lagrangian methods:

$$\mathcal{L}(\theta, \lambda) = L^{CPI}(\theta) - \lambda \left( D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) - \delta \right)$$

The optimal step uses the **natural gradient**:

$$\theta_{new} = \theta_{old} + \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g$$

Where $F$ is the Fisher Information Matrix (approximates the Hessian of KL).

### Complexity Issues

TRPO requires:
1. Computing the Fisher matrix $F$ (expensive for neural networks)
2. Computing $F^{-1}g$ (use conjugate gradient, but still costly)
3. Line search to satisfy constraint (multiple forward passes)

**This motivates PPO** — can we get similar guarantees without the complexity?

---

## PPO: Proximal Policy Optimization

### From Constraint to Penalty

PPO (Schulman et al., 2017) simplifies TRPO by replacing the hard constraint with clipping.

### The Clipped Surrogate Objective

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- $\epsilon$ is the clipping parameter (typically 0.1-0.2)

### How Clipping Works

**Case 1: $A_t > 0$ (good action)**

We want to increase $\pi_\theta(a|s)$, so $r_t$ increases.

$$L^{CLIP} = \min(r_t A_t, (1+\epsilon) A_t)$$

When $r_t > 1 + \epsilon$, the objective is clipped to $(1+\epsilon)A_t$. No incentive to increase probability further.

**Case 2: $A_t < 0$ (bad action)**

We want to decrease $\pi_\theta(a|s)$, so $r_t$ decreases.

$$L^{CLIP} = \min(r_t A_t, (1-\epsilon) A_t) = \max(r_t |A_t|, (1-\epsilon) |A_t|) \cdot (-1)$$

When $r_t < 1 - \epsilon$, the objective is clipped. No incentive to decrease probability further.

### Visual Understanding

```
                    L^CLIP when A > 0
           │
    (1+ε)A ┼─────────────────────────────
           │                    ╱
           │                  ╱
         A ┼────────────────╱
           │              ╱
           │            ╱
           │          ╱
           │        ╱
         0 ┼──────╱─────────────────────── r_t
           │    ╱
           │  ╱
           │╱
           0    1-ε    1    1+ε    2

           Clipped: no gradient beyond 1+ε

                    L^CLIP when A < 0
           │
         0 ┼──────────────────────────────
           │╲
           │  ╲
           │    ╲
           │      ╲
    (1-ε)A ┼────────╲─────────────────────
           │          ╲
           │            ╲
         A ┼──────────────╲───────────────
           │                ╲
           0    1-ε    1    1+ε    2    r_t

           Clipped: no gradient beyond 1-ε
```

### PPO for LLM (RLHF)

In the LLM setting, PPO optimizes:

$$\mathcal{L}_{PPO-LLM} = \mathbb{E}_{x, y} \left[ \min\left( r(\theta) A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

Where:
- $x$ is the prompt
- $y$ is the generated response
- $A = r_\phi(x, y) - b(x)$ is the advantage (reward minus baseline)
- The KL term prevents deviation from the reference policy

### PPO-RLHF Algorithm

```
PPO for RLHF:
─────────────────────────────────────────────────────────
Input: SFT model π_ref, Reward model r_φ, Clip ε, KL coef β

Initialize: π_θ ← π_ref

for iteration = 1, 2, ... do:
    1. Sample prompts x ~ D

    2. Generate responses: y ~ π_θ(·|x)

    3. Compute rewards: R = r_φ(x, y)

    4. Compute advantages: A = R - baseline
       (baseline can be running mean or value function)

    5. For each minibatch epoch:

       Compute ratio: r(θ) = π_θ(y|x) / π_θ_old(y|x)

       Compute clipped objective:
       L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)

       Compute KL penalty:
       L^KL = β · D_KL(π_θ || π_ref)

       Update: θ ← θ + α∇_θ(L^CLIP - L^KL)
─────────────────────────────────────────────────────────
```

### Token-Level vs Response-Level

PPO for LLM can operate at different granularities:

**Response-Level**: Treat entire response as one action
- Simpler, but high variance
- Reward assigned to full sequence

**Token-Level**: Each token is an action
- Lower variance with proper credit assignment
- More expensive (one "step" per token)

$$L^{token}(\theta) = \sum_{t=1}^{T} \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)$$

Where $r_t(\theta) = \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{old}(y_t|x, y_{<t})}$.

---

## DPO: Direct Preference Optimization

### The Key Insight

DPO (Rafailov et al., 2023) asks: *Can we skip the reward model entirely?*

**Key Observation**: The optimal policy for the RLHF objective has a closed form!

### Mathematical Derivation

**Step 1: The RLHF Objective**

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r(x, y) \right] - \beta D_{KL}(\pi \| \pi_{ref})$$

Expanding the KL divergence:

$$= \mathbb{E}_{x, y \sim \pi} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right]$$

**Step 2: The Optimal Solution**

Taking the derivative and setting to zero, the optimal policy is:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Where $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function.

**Step 3: Rearranging for Reward**

Solving for $r(x, y)$:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Key insight**: The reward can be expressed in terms of policy ratios!

**Step 4: The Bradley-Terry Model**

Human preferences follow the Bradley-Terry model:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Where $\sigma$ is the sigmoid function and $y_w \succ y_l$ means "$y_w$ is preferred over $y_l$".

**Step 5: Substituting the Reward**

Plugging in our expression for $r$:

$$P(y_w \succ y_l | x) = \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

Note: The $Z(x)$ terms cancel!

$$= \sigma\left( \beta \log \frac{\pi^*(y_w|x) / \pi_{ref}(y_w|x)}{\pi^*(y_l|x) / \pi_{ref}(y_l|x)} \right)$$

### The DPO Loss

Since we want our policy $\pi_\theta$ to match the optimal policy $\pi^*$, we directly optimize:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

Simplifying notation:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E} \left[ \log \sigma\left( \beta (r_\theta(x, y_w) - r_\theta(x, y_l)) \right) \right]$$

Where $r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ is the **implicit reward**.

### Understanding DPO Gradients

Taking the gradient:

$$\nabla_\theta \mathcal{L}_{DPO} = -\beta \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_l - \hat{r}_w)}_{\text{weight}} \left( \underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{increase } y_w} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{decrease } y_l} \right) \right]$$

Where $\hat{r}_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$ and $\hat{r}_l = \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$.

**Interpretation**:
- Weight $\sigma(\hat{r}_l - \hat{r}_w)$: Higher when model incorrectly ranks the pair
- Effect: Increase probability of preferred $y_w$, decrease probability of rejected $y_l$

### DPO Algorithm

```
DPO Algorithm:
─────────────────────────────────────────────────────────
Input: Reference policy π_ref, Preference dataset D, β

Initialize: π_θ ← π_ref

for iteration = 1, 2, ... do:
    1. Sample batch: (x, y_w, y_l) ~ D

    2. Compute log probabilities:
       log π_θ(y_w|x), log π_θ(y_l|x)
       log π_ref(y_w|x), log π_ref(y_l|x)

    3. Compute implicit rewards:
       r_w = β(log π_θ(y_w|x) - log π_ref(y_w|x))
       r_l = β(log π_θ(y_l|x) - log π_ref(y_l|x))

    4. Compute loss:
       L = -log σ(r_w - r_l)

    5. Update: θ ← θ - α∇_θL
─────────────────────────────────────────────────────────
```

### Why DPO Works

**Mathematical Equivalence**: DPO is solving the same optimization as RLHF:
- RLHF: Train reward model → Use RL to optimize
- DPO: Directly optimize using the closed-form solution

**The Implicit Reward**: $r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ measures how much more likely $y$ is under $\pi_\theta$ versus $\pi_{ref}$.

---

## GRPO: Group Relative Policy Optimization

### Motivation

GRPO (Shao et al., 2024) addresses limitations of both PPO and DPO:
- **PPO**: Requires a separate critic/value network
- **DPO**: Limited to pairwise preferences

### The Core Idea

Instead of using a learned value function (critic), GRPO estimates advantages from a **group of responses** to the same prompt.

### Group-Based Advantage Estimation

For each prompt $x$, sample a group of $G$ responses: $\{y_1, y_2, ..., y_G\} \sim \pi_{\theta_{old}}(y|x)$

Compute rewards for each: $\{r_1, r_2, ..., r_G\}$ where $r_i = r_\phi(x, y_i)$

**Advantage Estimation**:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

This is a **relative ranking** within the group — no absolute value function needed!

### Why Group Normalization?

1. **Removes baseline bias**: Centering by mean removes prompt-specific difficulty
2. **Normalizes scale**: Dividing by std handles reward magnitude differences
3. **No critic needed**: Baseline comes from same-batch responses

### GRPO Objective

$$\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{x} \mathbb{E}_{\{y_i\}_{i=1}^G \sim \pi_{old}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( r_i(\theta) \hat{A}_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]$$

With KL regularization:

$$\mathcal{L}_{GRPO}(\theta) = \mathcal{L}_{clip}(\theta) - \beta \cdot \mathbb{E}_{x, y \sim \pi_\theta} \left[ D_{KL}(\pi_\theta(y|x) \| \pi_{ref}(y|x)) \right]$$

### GRPO Algorithm

```
GRPO Algorithm:
─────────────────────────────────────────────────────────
Input: Reference policy π_ref, Reward model r_φ, Group size G

Initialize: π_θ ← π_ref

for iteration = 1, 2, ... do:
    1. Sample prompts: {x_1, ..., x_B} ~ D

    2. For each prompt x_b:
       Generate G responses: {y_b,1, ..., y_b,G} ~ π_θ_old(·|x_b)
       Compute rewards: {r_b,1, ..., r_b,G}

       Normalize advantages:
       μ_b = mean({r_b,i})
       σ_b = std({r_b,i})
       A_b,i = (r_b,i - μ_b) / σ_b

    3. Compute probability ratios:
       r_θ(y_b,i) = π_θ(y_b,i|x_b) / π_θ_old(y_b,i|x_b)

    4. Compute clipped objective:
       L = (1/BG) Σ_b Σ_i min(r_θ A_b,i, clip(r_θ, 1-ε, 1+ε) A_b,i)

    5. Add KL penalty:
       L_total = L - β · D_KL(π_θ || π_ref)

    6. Update: θ ← θ + α∇_θL_total
─────────────────────────────────────────────────────────
```

### Comparison with PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Baseline | Learned value function $V_\phi(s)$ | Group mean of rewards |
| Additional networks | Actor + Critic | Actor only |
| Sample efficiency | Can reuse samples | Needs fresh groups |
| Variance | Lower with good critic | Depends on group size |

### Mathematical Connection to Ranking

GRPO's advantage can be seen as a soft ranking:

If $G=2$ and we have $(y_w, y_l)$ with $r_w > r_l$:

$$\hat{A}_w = \frac{r_w - (r_w + r_l)/2}{\sqrt{(r_w - r_l)^2/2}} = \frac{r_w - r_l}{|r_w - r_l|} \cdot \frac{1}{\sqrt{2}} = \frac{1}{\sqrt{2}}$$

$$\hat{A}_l = -\frac{1}{\sqrt{2}}$$

This shows GRPO with $G=2$ reduces to a normalized version of pairwise ranking!

---

## Method Comparison

### At a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        METHOD COMPARISON                                     │
└─────────────────────────────────────────────────────────────────────────────┘

                    Reward Model?    Critic?    Reference Policy?
                         │             │              │
TRPO                    YES           YES            Optional
                         │             │              │
PPO                     YES           YES            YES (for KL)
                         │             │              │
DPO                     NO            NO             YES (required)
                         │             │              │
GRPO                    YES           NO             YES (for KL)

─────────────────────────────────────────────────────────────────────────────

                    Preference      Optimization     Complexity
                    Format          Type
                         │              │               │
TRPO                Reward model   Constrained        HIGH
                                   (trust region)
                         │              │               │
PPO                 Reward model   Clipped            MEDIUM
                                   surrogate
                         │              │               │
DPO                 Pairwise       Contrastive        LOW
                    (y_w, y_l)     (classification)
                         │              │               │
GRPO                Reward model   Clipped +          MEDIUM
                                   group baseline
```

### Detailed Comparison Table

| Method | Requires Reward Model | Requires Critic | Preference Format | Training Stability | Sample Efficiency | Implementation Complexity |
|--------|----------------------|-----------------|-------------------|-------------------|-------------------|--------------------------|
| **TRPO** | Yes | Yes | Scalar reward | Very High (guaranteed improvement) | Medium | High (Fisher matrix, line search) |
| **PPO** | Yes | Yes | Scalar reward | High | Medium | Medium |
| **DPO** | No | No | Pairwise $(y_w, y_l)$ | Medium | High (offline) | Low |
| **GRPO** | Yes | No | Scalar reward | High | Medium | Medium |

### Pros and Cons

#### TRPO

**Pros:**
- Theoretical guarantee of monotonic improvement
- Very stable training
- Works well with any advantage estimator

**Cons:**
- Computationally expensive (Fisher matrix inversion)
- Line search adds overhead
- Complex implementation
- Rarely used for LLMs due to cost

#### PPO

**Pros:**
- Simple to implement
- Good balance of stability and performance
- Flexible — works with any reward signal
- Can do online learning (reward model updates with policy)
- Widely tested and understood

**Cons:**
- Requires reward model (expensive to train)
- Requires critic network (more parameters)
- Reward hacking possible if KL not tuned well
- Sample inefficient (on-policy)

#### DPO

**Pros:**
- No reward model needed — simpler pipeline
- No critic needed — fewer parameters
- Stable training — just classification loss
- Sample efficient — offline on preference data
- Mathematically elegant

**Cons:**
- Limited to pairwise preferences
- Can't incorporate arbitrary reward signals
- Reference policy must be fixed (can't update)
- May underperform PPO on out-of-distribution prompts
- Less flexible for online/iterative improvement

#### GRPO

**Pros:**
- No critic network needed
- Works with reward models (flexible signals)
- Natural handling of multiple responses
- Good for outcome-based rewards (math, code)
- Simpler than PPO (no value function training)

**Cons:**
- Requires generating multiple responses per prompt
- Variance depends on group size $G$
- Less sample efficient than DPO
- Still needs reward model

### When to Use Each Method

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| Maximum theoretical guarantees | TRPO | Guaranteed improvement (if you can afford it) |
| General-purpose alignment | PPO | Flexible, well-understood, good performance |
| Offline training with preference data | DPO | Simple, efficient, no reward model |
| Math/code tasks with verifiable answers | GRPO | Can use outcome-based rewards |
| Limited compute | DPO | No reward model, no critic |
| Continuous improvement pipeline | PPO | Can update reward model over time |
| Cold start with little preference data | PPO | Can bootstrap with simple reward proxy |

### The Tradeoff Triangle

```
                    Simplicity
                       /\
                      /  \
                     /    \
                    / DPO  \
                   /________\
                  /          \
                 /   GRPO    \
                /____________ \
               /              \
              /      PPO       \
             /_________________ \
            /                   \
           /       TRPO          \
          /_______________________\
         Flexibility ←────────→ Stability
```

- **TRPO**: Maximum stability, moderate flexibility, low simplicity
- **PPO**: Good balance of all three
- **DPO**: Maximum simplicity, limited flexibility, moderate stability
- **GRPO**: Moderate simplicity, good flexibility, good stability

---

## Implementation Example: DPO Loss

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # log π_θ(y_w|x)
    policy_rejected_logps: torch.Tensor,  # log π_θ(y_l|x)
    ref_chosen_logps: torch.Tensor,       # log π_ref(y_w|x)
    ref_rejected_logps: torch.Tensor,     # log π_ref(y_l|x)
    beta: float = 0.1
) -> torch.Tensor:
    """
    Compute DPO loss.

    Args:
        policy_chosen_logps: Log probs of chosen response under π_θ
        policy_rejected_logps: Log probs of rejected response under π_θ
        ref_chosen_logps: Log probs of chosen response under π_ref
        ref_rejected_logps: Log probs of rejected response under π_ref
        beta: Temperature parameter

    Returns:
        DPO loss (scalar)
    """
    # Compute implicit rewards
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO loss: -log σ(r_w - r_l)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss


def compute_sequence_logprob(logits, labels, mask):
    """Compute log probability of a sequence."""
    # logits: (batch, seq_len, vocab_size)
    # labels: (batch, seq_len)
    # mask: (batch, seq_len) - 1 for valid tokens, 0 for padding

    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Sum over sequence, masking out padding
    sequence_log_probs = (token_log_probs * mask).sum(dim=-1)

    return sequence_log_probs
```

---

## Quiz

<details>
<summary><strong>Q1: Why does RLHF include a KL penalty term?</strong></summary>

**Answer**: The KL penalty prevents **reward hacking**. Without it, the policy can exploit weaknesses in the reward model to generate high-reward but nonsensical outputs. The KL term $D_{KL}(\pi_\theta \| \pi_{ref})$ keeps the optimized policy close to the reference (SFT) policy, ensuring the model retains its language capabilities.

**Key equation**:
$$\max_\pi \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi \| \pi_{ref})$$

**Common pitfall**: Setting $\beta$ too low leads to reward hacking; too high prevents learning.

</details>

<details>
<summary><strong>Q2: Derive why DPO's implicit reward is $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$.</strong></summary>

**Answer**: Starting from the optimal RLHF policy:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

Taking log of both sides:
$$\log \pi^*(y|x) = \log \pi_{ref}(y|x) + \frac{r(x,y)}{\beta} - \log Z(x)$$

Rearranging for $r$:
$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

In DPO, we parameterize $\pi^* \approx \pi_\theta$ and the partition function $Z(x)$ cancels when comparing pairs.

**Common pitfall**: Forgetting that $Z(x)$ only cancels in the pairwise comparison setting.

</details>

<details>
<summary><strong>Q3: What is the key difference between how PPO and GRPO estimate advantages?</strong></summary>

**Answer**:

**PPO** uses a learned **value function** (critic):
$$A_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{(TD error)}$$
or GAE for multi-step returns.

**GRPO** uses **group statistics**:
$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

The key differences:
- PPO requires training an additional network (critic)
- GRPO needs multiple samples per prompt (group)
- PPO's baseline is state-dependent; GRPO's is prompt-dependent
- GRPO baseline is unbiased but may have higher variance with small groups

</details>

<details>
<summary><strong>Q4: Why can't DPO easily incorporate arbitrary reward signals like PPO can?</strong></summary>

**Answer**: DPO's derivation assumes the reward is implicitly defined through the optimal policy's relationship with the reference policy:

$$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

This derivation requires:
1. Preference data $(y_w, y_l)$ pairs
2. The Bradley-Terry assumption about how rewards relate to preferences

For arbitrary reward signals (e.g., code execution results, mathematical correctness), DPO would require:
- Converting scalar rewards to pairwise comparisons
- Assuming rewards follow Bradley-Terry model
- Potentially losing information in the conversion

PPO directly optimizes any scalar reward through the policy gradient.

</details>

<details>
<summary><strong>Q5: What makes TRPO computationally expensive compared to PPO?</strong></summary>

**Answer**: TRPO requires:

1. **Fisher Information Matrix**: $F = \mathbb{E}[\nabla_\theta \log \pi \cdot \nabla_\theta \log \pi^T]$
   - Size: $|\theta| \times |\theta|$ — huge for large networks!

2. **Natural Gradient**: $d = F^{-1}g$
   - Direct inversion is $O(|\theta|^3)$
   - Use conjugate gradient: $O(k \cdot |\theta|^2)$ where $k$ is CG iterations

3. **Line Search**: Multiple forward passes to find valid step size satisfying KL constraint

**PPO's simplification**: Replace constraint with clipping
- No Fisher matrix computation
- No matrix inversion
- No line search
- Just gradient descent on clipped objective

</details>

<details>
<summary><strong>Q6: In GRPO, what happens if the group size G is too small?</strong></summary>

**Answer**: With small $G$:

1. **High variance in baseline**: Mean of few samples is noisy
2. **Biased std estimation**: With $G=2$, std is just $|r_1 - r_2|/\sqrt{2}$
3. **Binary advantage**: With $G=2$, advantages are essentially $\pm$ constant

**Mathematical example** with $G=2$:
$$\hat{A}_1 = \frac{r_1 - (r_1+r_2)/2}{\sigma} = \frac{r_1 - r_2}{2\sigma} = \pm\frac{1}{\sqrt{2}}$$

Regardless of the actual reward difference magnitude!

**Typical recommendation**: $G \geq 4$ for stable training; $G = 8$-$16$ common in practice.

</details>

---

## References

1. **TRPO**: Schulman et al. (2015). "Trust Region Policy Optimization." ICML.
2. **PPO**: Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv.
3. **RLHF**: Ouyang et al. (2022). "Training language models to follow instructions with human feedback." NeurIPS.
4. **DPO**: Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS.
5. **GRPO**: Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv.

---

## What to Memorize for Interviews

| Method | Key Equation | One-Line Summary |
|--------|-------------|------------------|
| **TRPO** | $\max L^{CPI}$ s.t. $D_{KL} \leq \delta$ | Constrained optimization with trust region |
| **PPO** | $\min(r_t A_t, \text{clip}(r_t) A_t)$ | Clipped surrogate replaces constraint |
| **DPO** | $-\log\sigma(\beta(r_w - r_l))$ where $r = \log\frac{\pi_\theta}{\pi_{ref}}$ | Implicit reward from policy ratio |
| **GRPO** | $A_i = (r_i - \mu) / \sigma$ | Group-normalized advantage, no critic |
