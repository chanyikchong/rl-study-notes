# RLHF: Reinforcement Learning from Human Feedback

## Interview Summary

**RLHF** aligns language models with human preferences through a 3-step pipeline: (1) Supervised Fine-Tuning, (2) Reward Model training, (3) RL optimization. The core challenge is maximizing reward while preventing reward hacking via KL regularization. Multiple algorithms exist: **TRPO** (2015) provides theoretical guarantees, **PPO** (2017) simplifies with clipping, **DPO** (2023) eliminates the reward model, **GRPO** (2024) uses group-based advantages.

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
│ π_SFT        │    │ PPO/DPO/GRPO    │    │ π_θ          │
└──────────────┘    └─────────────────┘    └──────────────┘
```

---

## The Core RLHF Objective

All methods aim to maximize expected reward while staying close to the reference policy:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{KL}\left(\pi_\theta \| \pi_{ref}\right)$$

Where:
- $\pi_\theta$: Policy being optimized (the LLM)
- $\pi_{ref}$: Reference policy (usually SFT model)
- $r_\phi(x, y)$: Reward model score for response $y$ to prompt $x$
- $\beta$: KL penalty coefficient (prevents reward hacking)

### Why the KL Penalty?

Without it, the model can exploit reward model weaknesses:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REWARD HACKING                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Without KL penalty:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Policy finds "exploits" in reward model:                                   │
│  - Repetitive phrases that score high                                       │
│  - Sycophantic responses ("Great question!")                               │
│  - Technically high-reward but nonsensical outputs                          │
│                                                                             │
│  Result: High reward, but useless or harmful responses                      │
└─────────────────────────────────────────────────────────────────────────────┘

With KL penalty:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Policy must stay close to SFT model:                                       │
│  - Can't deviate too far from sensible language                            │
│  - Balanced improvement in reward                                           │
│  - Maintains linguistic coherence                                           │
│                                                                             │
│  Result: Genuine improvement in helpfulness/safety                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Evolution of RLHF Methods

The field has evolved from complex constrained optimization to simpler, more practical methods:

```
Timeline of RLHF Methods:
═══════════════════════════════════════════════════════════════════════════════

2015 ─── TRPO (Trust Region Policy Optimization)
         │
         │  Theoretical foundation: monotonic improvement guarantee
         │  Problem: Complex (Fisher matrix, conjugate gradient, line search)
         │
         ▼
2017 ─── PPO (Proximal Policy Optimization)
         │
         │  Simplified TRPO: clipping instead of constraint
         │  Became the default for RLHF (used by InstructGPT, ChatGPT)
         │
         ▼
2023 ─── DPO (Direct Preference Optimization)
         │
         │  Key insight: Skip reward model entirely!
         │  Directly optimize on preference pairs
         │  Simpler pipeline, but less flexible
         │
         ▼
2024 ─── GRPO (Group Relative Policy Optimization)
         │
         │  No critic network needed
         │  Group-based advantage estimation
         │  Used in DeepSeekMath
         │
         ▼
      ... More methods emerging (KTO, IPO, ORPO, etc.)

═══════════════════════════════════════════════════════════════════════════════
```

---

## Method Comparison At a Glance

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
```

### Detailed Comparison

| Method | Year | Key Innovation | Pros | Cons |
|--------|------|----------------|------|------|
| [TRPO](trpo.md) | 2015 | Trust region constraint | Theoretical guarantees, stable | Complex, expensive |
| [PPO](ppo.md) | 2017 | Clipped surrogate | Simple, effective, widely used | Needs reward model + critic |
| [DPO](dpo.md) | 2023 | Implicit reward | No reward model, simple | Less flexible, pairwise only |
| [GRPO](grpo.md) | 2024 | Group normalization | No critic, flexible | Multiple samples needed |

### When to Use Each

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| General RLHF | PPO | Well-tested, flexible |
| Limited compute | DPO | No reward model training |
| Verifiable tasks (math/code) | GRPO | Outcome-based rewards |
| Maximum stability | TRPO | Guaranteed improvement |
| Offline preference data | DPO | No online sampling needed |

---

## Individual Method Pages

Each method has its own detailed page:

1. **[TRPO](trpo.md)** — Trust Region Policy Optimization (2015)
   - Theoretical foundation of policy optimization
   - Monotonic improvement guarantees
   - Natural gradient and Fisher information matrix

2. **[PPO](ppo.md)** — Proximal Policy Optimization (2017)
   - Clipped surrogate objective
   - Complete Actor-Critic implementation
   - PPO for RLHF specifics

3. **[DPO](dpo.md)** — Direct Preference Optimization (2023)
   - Bypassing the reward model
   - Implicit reward derivation
   - Bradley-Terry model connection

4. **[GRPO](grpo.md)** — Group Relative Policy Optimization (2024)
   - Group-based advantage estimation
   - No critic required
   - Connection to ranking losses

---

## Key Equations Summary

| Method | Core Equation |
|--------|---------------|
| **RLHF Objective** | $\max_\pi \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi \|\| \pi_{ref})$ |
| **TRPO** | $\max L^{CPI}$ s.t. $D_{KL} \leq \delta$ |
| **PPO** | $L = \min(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t)$ |
| **DPO** | $L = -\log\sigma(\beta(r_w - r_l))$ where $r = \log\frac{\pi_\theta}{\pi_{ref}}$ |
| **GRPO** | $A_i = (r_i - \mu) / \sigma$ (group-normalized) |

---

## Quiz

<details>
<summary><strong>Q1: What are the three steps of the RLHF pipeline?</strong></summary>

**Answer**:
1. **Supervised Fine-Tuning (SFT)**: Fine-tune base LLM on demonstration data
2. **Reward Model Training**: Train a model to predict human preferences from comparison data
3. **RL Optimization**: Use PPO/DPO/etc. to maximize reward while staying close to SFT model

</details>

<details>
<summary><strong>Q2: Why is the KL penalty necessary in RLHF?</strong></summary>

**Answer**: The KL penalty prevents **reward hacking** — where the model exploits weaknesses in the reward model to produce high-reward but nonsensical or harmful outputs. By keeping the policy close to the SFT model, we ensure the model maintains linguistic coherence and only makes genuine improvements.

</details>

<details>
<summary><strong>Q3: What's the main advantage of DPO over PPO?</strong></summary>

**Answer**: DPO **eliminates the need for a reward model**. Instead of training a separate reward model and then doing RL, DPO directly optimizes the policy on preference pairs. This simplifies the pipeline and avoids potential issues with reward model quality.

</details>

---

## References

1. **TRPO**: Schulman et al. (2015). "Trust Region Policy Optimization."
2. **PPO**: Schulman et al. (2017). "Proximal Policy Optimization Algorithms."
3. **InstructGPT/RLHF**: Ouyang et al. (2022). "Training language models to follow instructions with human feedback."
4. **DPO**: Rafailov et al. (2023). "Direct Preference Optimization."
5. **GRPO**: Shao et al. (2024). "DeepSeekMath."
