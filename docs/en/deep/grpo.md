# GRPO: Group Relative Policy Optimization (2024)

## Interview Summary

**GRPO** eliminates the need for a critic network by estimating advantages from a **group of responses** to the same prompt. For each prompt, sample $G$ responses, compute rewards, then normalize: $\hat{A}_i = (r_i - \mu) / \sigma$. This is a **relative ranking** within the group — no learned value function needed. GRPO uses PPO's clipped objective but with group-normalized advantages. Particularly effective for tasks with **verifiable rewards** (math, code) where ground-truth correctness provides the reward signal.

**What to memorize**: Group advantage formula $\hat{A}_i = (r_i - \mu) / \sigma$, why it eliminates the critic, connection to ranking losses.

---

## Motivation

GRPO (Shao et al., 2024) addresses key limitations of existing methods:

| Method | Limitation GRPO Addresses |
|--------|--------------------------|
| **PPO** | Requires a separate critic/value network (doubles parameters) |
| **DPO** | Limited to pairwise preferences (only 2 responses compared) |

**GRPO's solution**: Use a *group* of responses as a self-contained baseline — no critic, and can handle any number of responses.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE CRITIC PROBLEM IN LLM RLHF                           │
└─────────────────────────────────────────────────────────────────────────────┘

In classic RL (Atari, robotics):
┌─────────────────────────────────────────────────────────────────────────────┐
│  Critic V(s) predicts future rewards from state s                           │
│  • States are Markovian (contain all relevant info)                         │
│  • Critic learns a meaningful value function                                │
│  • Reduces variance in advantage estimates                                   │
└─────────────────────────────────────────────────────────────────────────────┘

In LLM RLHF:
┌─────────────────────────────────────────────────────────────────────────────┐
│  "State" = prompt + partial response                                         │
│  • What's the "value" of a partial response?                                │
│  • Reward only at end (non-Markovian structure)                             │
│  • Critic often struggles to learn meaningful values                         │
│  • Doubles memory requirements (critic = another LLM-sized network)         │
└─────────────────────────────────────────────────────────────────────────────┘

GRPO insight: Skip the critic entirely!
```

---

## The Core Idea

Instead of using a learned value function (critic), GRPO estimates advantages from a **group of responses** to the same prompt.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GRPO SAMPLING STRATEGY                               │
└─────────────────────────────────────────────────────────────────────────────┘

For each prompt x, sample G responses:

                              Prompt: "What is 2 + 2?"
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
               Response 1          Response 2          Response G
               "2 + 2 = 4"        "It's 4!"          "The answer..."
                    │                   │                   │
                    ▼                   ▼                   ▼
                 r₁ = 1.0           r₂ = 0.8           rG = 0.9
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
                                        ▼
                              Compute group stats:
                              μ = mean(r₁, r₂, ..., rG)
                              σ = std(r₁, r₂, ..., rG)
                                        │
                                        ▼
                           Advantages = (rᵢ - μ) / σ
```

---

## Group-Based Advantage Estimation

### The Formula

For each prompt $x$, sample a group of $G$ responses: $\{y_1, y_2, ..., y_G\} \sim \pi_{\theta_{old}}(y|x)$

Compute rewards for each: $\{r_1, r_2, ..., r_G\}$ where $r_i = r_\phi(x, y_i)$

**Advantage Estimation**:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)} = \frac{r_i - \mu_G}{\sigma_G}$$

This is a **relative ranking** within the group — no absolute value function needed!

### Why This Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY GROUP NORMALIZATION WORKS                            │
└─────────────────────────────────────────────────────────────────────────────┘

1. REMOVES BASELINE BIAS
   ──────────────────────────────────────────────────────────────────────────
   Problem: Different prompts have different inherent difficulty

   Prompt A: "What is 1+1?"      → Most responses get high reward
   Prompt B: "Prove P≠NP"        → Most responses get low reward

   Without normalization: Prompt A responses always look "better"
   With normalization: Only relative quality within prompt matters

2. NORMALIZES SCALE
   ──────────────────────────────────────────────────────────────────────────
   Problem: Reward magnitudes can vary across prompts/domains

   Dividing by σ ensures:
   • Best response in group ≈ +1 to +2
   • Worst response in group ≈ -1 to -2
   • Consistent gradient magnitudes

3. NO CRITIC NEEDED
   ──────────────────────────────────────────────────────────────────────────
   Traditional: A(s,a) = R - V(s), where V(s) is learned
   GRPO:        A(x,y) = (r - μ) / σ, where μ,σ from same batch

   The group mean serves as an empirical baseline!
```

---

## GRPO Objective

GRPO combines PPO's clipped objective with group-normalized advantages:

$$\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{x} \mathbb{E}_{\{y_i\}_{i=1}^G \sim \pi_{old}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( r_i(\theta) \hat{A}_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]$$

Where:
- $r_i(\theta) = \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}$ is the probability ratio (same as PPO)
- $\hat{A}_i = (r_i - \mu_G) / \sigma_G$ is the group-normalized advantage
- $\epsilon$ is the clipping parameter (typically 0.2)

### With KL Regularization

To prevent reward hacking:

$$\mathcal{L}_{GRPO}(\theta) = \mathcal{L}_{clip}(\theta) - \beta \cdot \mathbb{E}_{x, y \sim \pi_\theta} \left[ D_{KL}(\pi_\theta(y|x) \| \pi_{ref}(y|x)) \right]$$

---

## GRPO Algorithm

```
GRPO Algorithm:
═══════════════════════════════════════════════════════════════════════════════
Input: Reference policy π_ref, Reward model/function r_φ, Group size G, β

Initialize: π_θ ← π_ref

for iteration = 1, 2, ... do:

    ┌─ SAMPLE PROMPTS ────────────────────────────────────────────────────────┐
    │  1. Sample batch of prompts: {x_1, ..., x_B} ~ D                        │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ GENERATE RESPONSE GROUPS ──────────────────────────────────────────────┐
    │  2. For each prompt x_b:                                                 │
    │     Generate G responses: {y_b,1, ..., y_b,G} ~ π_θ_old(·|x_b)          │
    │     Compute rewards: {r_b,1, ..., r_b,G} via reward model               │
    │                                                                          │
    │     (For verifiable tasks like math, rewards can be binary 0/1)         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ NORMALIZE ADVANTAGES ──────────────────────────────────────────────────┐
    │  3. For each prompt group:                                               │
    │     μ_b = mean({r_b,i}_{i=1}^G)                                         │
    │     σ_b = std({r_b,i}_{i=1}^G) + ε  (add small ε for stability)         │
    │     A_b,i = (r_b,i - μ_b) / σ_b                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ COMPUTE POLICY RATIOS ─────────────────────────────────────────────────┐
    │  4. r_θ(y_b,i) = π_θ(y_b,i|x_b) / π_θ_old(y_b,i|x_b)                   │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ COMPUTE CLIPPED OBJECTIVE ─────────────────────────────────────────────┐
    │  5. L = (1/BG) Σ_b Σ_i min(r_θ A_b,i, clip(r_θ, 1-ε, 1+ε) A_b,i)       │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ ADD KL PENALTY ────────────────────────────────────────────────────────┐
    │  6. L_total = L - β · D_KL(π_θ || π_ref)                                │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ UPDATE ────────────────────────────────────────────────────────────────┐
    │  7. θ ← θ + α∇_θL_total                                                 │
    └─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
```

---

## GRPO vs PPO Comparison

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Baseline** | Learned value function $V_\phi(s)$ | Group mean of rewards $\mu_G$ |
| **Networks** | Actor + Critic (2x parameters) | Actor only |
| **Variance source** | Critic estimation error | Group size (larger G → lower variance) |
| **Sample efficiency** | Can reuse samples (with care) | Needs fresh groups per iteration |
| **Memory** | Policy + Critic + Reward Model | Policy + Reward Model |
| **Best for** | General RLHF | Verifiable reward tasks |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MEMORY COMPARISON                                       │
└─────────────────────────────────────────────────────────────────────────────┘

PPO for RLHF:
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Policy  │ │ Critic  │ │ Reward  │ │  Ref    │
│  π_θ    │ │  V_φ    │ │  Model  │ │  π_ref  │
│ (7B)    │ │ (7B)    │ │ (7B)    │ │ (7B)    │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
Total: ~28B parameters to store

GRPO:
┌─────────┐             ┌─────────┐ ┌─────────┐
│ Policy  │             │ Reward  │ │  Ref    │
│  π_θ    │             │  Model  │ │  π_ref  │
│ (7B)    │             │ (7B)    │ │ (7B)    │
└─────────┘             └─────────┘ └─────────┘
Total: ~21B parameters (saves 25%)
```

---

## Mathematical Connection to Ranking

GRPO's advantage can be seen as a **soft ranking loss**. Consider the special case $G=2$:

If we have $(y_w, y_l)$ with $r_w > r_l$:

$$\mu = \frac{r_w + r_l}{2}$$

$$\sigma = \sqrt{\frac{(r_w - \mu)^2 + (r_l - \mu)^2}{2}} = \frac{|r_w - r_l|}{\sqrt{2}}$$

Therefore:

$$\hat{A}_w = \frac{r_w - (r_w + r_l)/2}{|r_w - r_l|/\sqrt{2}} = \frac{(r_w - r_l)/2}{(r_w - r_l)/\sqrt{2}} = \frac{1}{\sqrt{2}} \approx 0.707$$

$$\hat{A}_l = -\frac{1}{\sqrt{2}} \approx -0.707$$

**Key insight**: With $G=2$, GRPO reduces to a normalized pairwise ranking! The winner always gets $+\frac{1}{\sqrt{2}}$ and loser gets $-\frac{1}{\sqrt{2}}$ regardless of reward magnitudes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRPO AS GENERALIZED RANKING                              │
└─────────────────────────────────────────────────────────────────────────────┘

G=2:   Equivalent to pairwise preference (like DPO)
       Winner: A = +1/√2, Loser: A = -1/√2

G=4:   Soft ranking over 4 responses
       Best:  A ≈ +1.5
       Good:  A ≈ +0.5
       Poor:  A ≈ -0.5
       Worst: A ≈ -1.5

G=16:  Fine-grained ranking signal
       More variance reduction
       Better advantage estimates
       But: More generation cost

Trade-off: Larger G → better estimates, higher compute cost
```

---

## When to Use GRPO

### Ideal Use Cases

1. **Mathematical Reasoning** (DeepSeekMath)
   - Binary correctness rewards (right/wrong)
   - No subjective judgment needed
   - Easy to verify

2. **Code Generation**
   - Test-based rewards (pass/fail)
   - Deterministic evaluation
   - Clear success criteria

3. **Memory-Constrained Settings**
   - No critic network needed
   - ~25% memory savings vs PPO

### Less Ideal Use Cases

1. **Subjective Quality** (helpfulness, creativity)
   - Requires learned reward model
   - Reward model may be noisy
   - Group normalization may amplify noise

2. **Low-Diversity Responses**
   - If all G responses are similar → small σ → unstable
   - Need sufficient exploration

---

## Practical Considerations

### Choosing Group Size G

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GROUP SIZE TRADE-OFFS                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Small G (e.g., 2-4):
├── Pros: Lower generation cost
├── Cons: Higher variance in advantage estimates
└── Best for: Quick iteration, limited compute

Medium G (e.g., 8-16):
├── Pros: Good balance of variance and cost
├── Cons: Moderate generation overhead
└── Best for: Most practical applications

Large G (e.g., 32-64):
├── Pros: Low variance, stable training
├── Cons: High generation cost
└── Best for: Final training runs, critical applications
```

### Handling Edge Cases

```python
# Pseudocode for robust advantage estimation
def compute_advantages(rewards):
    mu = mean(rewards)
    sigma = std(rewards)

    # Handle low-variance groups
    if sigma < epsilon:
        # All rewards similar → no meaningful ranking
        # Option 1: Skip this group
        # Option 2: Use raw centered rewards
        return (rewards - mu)  # unnormalized

    return (rewards - mu) / sigma
```

---

## Quiz

<details>
<summary><strong>Q1: What problem does GRPO solve that PPO doesn't handle well in LLM settings?</strong></summary>

**Answer**: GRPO eliminates the need for a **critic (value) network**. In LLM RLHF, the critic is problematic because:

1. **Memory**: Doubles parameters (critic is another LLM-sized model)
2. **Semantics**: "Value of a partial response" is ill-defined
3. **Training**: Critic often struggles to learn meaningful values for language

GRPO replaces the learned baseline $V(s)$ with the empirical group mean $\mu_G$, which is computed directly from same-prompt responses.

**Key equation**: $\hat{A}_i = (r_i - \mu_G) / \sigma_G$ vs PPO's $\hat{A}_t = R_t - V(s_t)$

</details>

<details>
<summary><strong>Q2: Write out the GRPO advantage formula and explain why each term is needed.</strong></summary>

**Answer**:

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}$$

**Terms**:
- $r_i$: Reward for response $y_i$ (from reward model or verifier)
- $\mu_G = \frac{1}{G}\sum_{j=1}^G r_j$: Group mean — serves as baseline, removes prompt-specific difficulty
- $\sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_G)^2}$: Group std — normalizes scale, ensures consistent gradient magnitudes

**Why subtract mean?** Different prompts have different inherent difficulties. Centering ensures we only learn relative quality.

**Why divide by std?** Reward scales can vary. Normalization prevents some prompts from dominating gradients.

</details>

<details>
<summary><strong>Q3: Show that GRPO with G=2 is equivalent to pairwise ranking.</strong></summary>

**Answer**: For $G=2$ with rewards $r_w > r_l$:

$$\mu = \frac{r_w + r_l}{2}$$

$$\sigma = \sqrt{\frac{(r_w - \mu)^2 + (r_l - \mu)^2}{2}} = \frac{r_w - r_l}{\sqrt{2}}$$

Therefore:
$$\hat{A}_w = \frac{r_w - \mu}{\sigma} = \frac{(r_w - r_l)/2}{(r_w - r_l)/\sqrt{2}} = \frac{1}{\sqrt{2}}$$

$$\hat{A}_l = \frac{r_l - \mu}{\sigma} = -\frac{1}{\sqrt{2}}$$

**Key insight**: The magnitude of reward difference cancels out! Winner always gets $+\frac{1}{\sqrt{2}}$, loser always gets $-\frac{1}{\sqrt{2}}$. This is pure pairwise ranking — exactly what DPO does with its implicit rewards.

</details>

<details>
<summary><strong>Q4: Why is GRPO particularly well-suited for math reasoning tasks?</strong></summary>

**Answer**: Math reasoning tasks have **verifiable rewards**:

1. **Binary correctness**: Answer is right or wrong (no subjective judgment)
2. **No reward model bias**: Ground truth provides reward, not a learned model
3. **Clear signal**: When G responses are sampled, correct ones get r=1, incorrect get r=0

The group normalization naturally creates a ranking:
- If 3/8 responses are correct: correct ones get positive advantages, incorrect get negative
- The proportion of correct responses doesn't bias the learning (thanks to normalization)

**Why PPO is worse here**: The critic struggles to predict "value of a partial math solution" — the value depends on correctness of the final answer, which is hard to predict from intermediate steps.

</details>

<details>
<summary><strong>Q5: You're training with GRPO and notice that loss is unstable with frequent NaN values. What's likely wrong and how do you fix it?</strong></summary>

**Answer**: Most likely cause: **low variance in reward groups**.

When all G responses get similar rewards (e.g., all correct or all wrong):
- $\sigma_G \approx 0$
- Division by $\sigma_G$ → explosion or NaN

**Fixes**:

1. **Add epsilon to denominator**: $\hat{A}_i = (r_i - \mu) / (\sigma + \epsilon)$ where $\epsilon = 10^{-6}$

2. **Skip low-variance groups**: If $\sigma < \text{threshold}$, don't update on this batch

3. **Increase temperature**: Sample with higher temperature to increase response diversity

4. **Larger G**: More responses → more likely to have variance

5. **Check reward function**: If rewards are clipped too aggressively, variance disappears

**Debugging**: Log $\sigma_G$ values per batch. If many are near zero, increase diversity or adjust reward scaling.

</details>

---

## References

1. Shao, Z., et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models."
2. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms."
3. Rafailov, R., et al. (2023). "Direct Preference Optimization."

---

## What to Memorize for Interviews

| Concept | Key Point |
|---------|-----------|
| **Main idea** | Eliminate critic using group-based advantage estimation |
| **Advantage formula** | $\hat{A}_i = (r_i - \mu_G) / \sigma_G$ |
| **Why normalize** | Removes prompt difficulty bias, normalizes scale |
| **G=2 equivalence** | Reduces to pairwise ranking (like DPO) |
| **Memory savings** | No critic → ~25% less memory vs PPO |
| **Best use case** | Verifiable rewards (math, code) |
| **Connection to PPO** | Same clipped objective, different baseline |
