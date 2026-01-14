# DPO: Direct Preference Optimization (2023)

## Interview Summary

**DPO** eliminates the reward model by deriving a **closed-form optimal policy** for the RLHF objective. The key insight: $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$ — rewards can be expressed as policy ratios! Using the Bradley-Terry preference model, DPO directly optimizes $\mathcal{L} = -\log \sigma(\beta(r_w - r_l))$ where $r = \log \frac{\pi_\theta}{\pi_{ref}}$ is the **implicit reward**. Simpler pipeline (no RL loop), but requires paired preference data and keeps reference model in memory.

**What to memorize**: Implicit reward $r = \log \frac{\pi_\theta}{\pi_{ref}}$, DPO loss formula, why it's mathematically equivalent to RLHF.

---

## The Key Insight

DPO (Rafailov et al., 2023) asks: *Can we skip the reward model entirely?*

The answer is yes — by exploiting the mathematical structure of the RLHF objective.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RLHF vs DPO PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Traditional RLHF (PPO):
┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│ Collect  │ → │ Train Reward │ → │ RL Training │ → │ Aligned      │
│ Prefs    │   │ Model r_φ    │   │ (PPO loop)  │   │ Model        │
└──────────┘   └──────────────┘   └─────────────┘   └──────────────┘
     ↓              ↓                   ↓                 ↓
  (y_w, y_l)    Separate NN      Actor + Critic      π_θ

DPO:
┌──────────┐   ┌─────────────────────────────────┐   ┌──────────────┐
│ Collect  │ → │ Direct Optimization on Prefs   │ → │ Aligned      │
│ Prefs    │   │ (single supervised loss)        │   │ Model        │
└──────────┘   └─────────────────────────────────┘   └──────────────┘
     ↓                        ↓                           ↓
  (y_w, y_l)        No reward model needed!            π_θ
```

**Why this matters**: DPO eliminates the entire RL training loop and reward model training, making alignment much simpler and more stable.

---

## Mathematical Derivation

### Step 1: The RLHF Objective

Recall the standard RLHF objective:

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r(x, y) \right] - \beta D_{KL}(\pi \| \pi_{ref})$$

Expanding the KL divergence:

$$= \mathbb{E}_{x, y \sim \pi} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right]$$

### Step 2: The Optimal Solution

This is a constrained optimization problem. Taking the derivative and setting to zero, the optimal policy has a closed form:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Where $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the **partition function** (normalizing constant).

**Intuition**: The optimal policy is the reference policy reweighted by exponentiated reward. High reward → higher probability.

### Step 3: Rearranging for Reward

Here's the key trick — we solve for $r(x, y)$:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Key insight**: The reward can be expressed entirely in terms of policy ratios! We don't need to learn a separate reward function.

### Step 4: The Bradley-Terry Model

Human preferences follow the Bradley-Terry model:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Where:
- $\sigma$ is the sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- $y_w \succ y_l$ means "$y_w$ is preferred over $y_l$"

**Why sigmoid?** The Bradley-Terry model says the probability of preferring one option over another depends on the difference in their "strengths" (rewards) through a logistic function.

### Step 5: Substituting the Reward

Plugging in our expression for $r$:

$$P(y_w \succ y_l | x) = \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

**Critical observation**: The $Z(x)$ terms cancel!

$$= \sigma\left( \beta \log \frac{\pi^*(y_w|x) / \pi_{ref}(y_w|x)}{\pi^*(y_l|x) / \pi_{ref}(y_l|x)} \right)$$

This is why DPO works — the intractable partition function disappears.

---

## The DPO Loss

Since we want our policy $\pi_\theta$ to match the optimal policy $\pi^*$, we maximize the likelihood of observed preferences:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

Using simplified notation with the **implicit reward** $r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E} \left[ \log \sigma\left( \beta (r_\theta(x, y_w) - r_\theta(x, y_l)) \right) \right]$$

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DPO LOSS INTUITION                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                          r_w - r_l
              ◄───────────────┼────────────────►
                              │
   Model prefers y_l          │          Model prefers y_w
   (WRONG - high loss)        │          (CORRECT - low loss)
                              │
         ┌────────────────────┼────────────────────┐
   Loss  │                    │                    │
         │    ╲               │               ╱    │
         │     ╲              │              ╱     │
         │      ╲             │             ╱      │
         │       ╲            │            ╱       │
         │        ╲___________│___________╱        │
         └────────────────────┴────────────────────┘
              -∞              0              +∞

    When r_w - r_l is large positive → low loss (model correctly ranks)
    When r_w - r_l is negative → high loss (model incorrectly ranks)
```

---

## Understanding DPO Gradients

Taking the gradient of the DPO loss:

$$\nabla_\theta \mathcal{L}_{DPO} = -\beta \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_l - \hat{r}_w)}_{\text{weight}} \left( \underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{increase } y_w} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{decrease } y_l} \right) \right]$$

Where:
- $\hat{r}_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$
- $\hat{r}_l = \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$

### Gradient Interpretation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DPO GRADIENT COMPONENTS                                │
└─────────────────────────────────────────────────────────────────────────────┘

The gradient has two parts:

1. WEIGHT: σ(r̂_l - r̂_w)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  • High when model incorrectly ranks (r̂_l > r̂_w)                       │
   │  • Low when model correctly ranks (r̂_w > r̂_l)                          │
   │  → Focus learning on examples the model gets wrong                      │
   └─────────────────────────────────────────────────────────────────────────┘

2. DIRECTION: ∇log π(y_w) - ∇log π(y_l)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  • Increase probability of preferred response y_w                       │
   │  • Decrease probability of rejected response y_l                        │
   │  → Contrastive learning on the preference pair                          │
   └─────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: DPO naturally focuses on hard examples where the model disagrees with human preferences.

---

## The Implicit Reward

The **implicit reward** is central to understanding DPO:

$$r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

**What it measures**: How much more likely response $y$ is under the trained policy $\pi_\theta$ compared to the reference policy $\pi_{ref}$.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      IMPLICIT REWARD INTERPRETATION                         │
└─────────────────────────────────────────────────────────────────────────────┘

r_θ(x, y) = log π_θ(y|x) - log π_ref(y|x)

┌─────────────────────────────────────────────────────────────────────────────┐
│  r_θ > 0:  Training has made y MORE likely (policy favors this response)   │
│  r_θ = 0:  Same probability as reference (no change)                       │
│  r_θ < 0:  Training has made y LESS likely (policy disfavors this)         │
└─────────────────────────────────────────────────────────────────────────────┘

Why "implicit"?
─────────────────────────────────────────────────────────────────────────────
• We never train a separate reward model
• The reward is implicitly defined by how the policy changes from reference
• Same mathematical role as explicit reward in RLHF objective
```

---

## DPO Algorithm

```
DPO Algorithm:
═══════════════════════════════════════════════════════════════════════════════
Input: Reference policy π_ref, Preference dataset D = {(x, y_w, y_l)}, β

Initialize: π_θ ← π_ref (copy reference policy)

for iteration = 1, 2, ... do:

    ┌─ SAMPLE BATCH ──────────────────────────────────────────────────────────┐
    │  1. Sample preference pairs: (x, y_w, y_l) ~ D                          │
    │     - x: prompt                                                          │
    │     - y_w: preferred (winning) response                                  │
    │     - y_l: rejected (losing) response                                    │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ COMPUTE LOG PROBABILITIES ─────────────────────────────────────────────┐
    │  2. Forward pass through both models:                                    │
    │     log π_θ(y_w|x), log π_θ(y_l|x)    (current policy)                  │
    │     log π_ref(y_w|x), log π_ref(y_l|x) (reference policy, frozen)       │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ COMPUTE IMPLICIT REWARDS ──────────────────────────────────────────────┐
    │  3. Calculate reward differences:                                        │
    │     r_w = β(log π_θ(y_w|x) - log π_ref(y_w|x))                          │
    │     r_l = β(log π_θ(y_l|x) - log π_ref(y_l|x))                          │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ COMPUTE LOSS ──────────────────────────────────────────────────────────┐
    │  4. DPO loss (binary cross-entropy style):                               │
    │     L = -log σ(r_w - r_l)                                               │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ UPDATE ────────────────────────────────────────────────────────────────┐
    │  5. Gradient descent: θ ← θ - α∇_θL                                     │
    └─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
```

---

## DPO vs PPO: Detailed Comparison

| Aspect | PPO | DPO |
|--------|-----|-----|
| **Reward Model** | Required (separate training) | Not needed |
| **Training Loop** | RL loop (generate → reward → update) | Supervised learning |
| **Reference Policy** | For KL penalty | Required (kept in memory) |
| **Critic Network** | Required | Not needed |
| **Memory** | Policy + Critic + Reward Model | Policy + Reference |
| **Stability** | Can be unstable | More stable |
| **Data** | Can sample online | Requires offline preferences |
| **Flexibility** | Any reward signal | Only pairwise preferences |

### When to Use Each

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHOOSE YOUR METHOD                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Use DPO when:
├── You have a static dataset of preference pairs
├── You want simpler implementation and training
├── Memory is limited (no reward model needed)
├── You want more stable training
└── Pairwise preferences are sufficient

Use PPO when:
├── You need online data generation
├── You have non-pairwise reward signals (e.g., from code execution)
├── You want to iterate on reward model separately
├── You need maximum flexibility in reward design
└── You have compute for the full RLHF pipeline
```

---

## Why DPO Works

### Mathematical Equivalence

DPO solves the same optimization problem as RLHF:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EQUIVALENCE OF RLHF AND DPO                              │
└─────────────────────────────────────────────────────────────────────────────┘

RLHF Objective:
    max_π E[r(x,y)] - β D_KL(π || π_ref)
          ↓
    Optimal policy: π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)
          ↓
    Implies: r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)

DPO Approach:
    Instead of learning r explicitly, use the implicit form
          ↓
    Optimize π_θ to match π* using preference likelihood
          ↓
    Z(x) cancels when comparing two responses!
          ↓
    Result: Same optimal policy, simpler path to get there
```

### The Reference Policy Role

The reference policy $\pi_{ref}$ plays two crucial roles:

1. **Regularization**: Prevents the policy from deviating too far (implicit KL constraint)
2. **Baseline**: Defines the "zero point" for implicit rewards

```
Without reference policy constraint:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Policy could maximize preference likelihood by:                            │
│  • Assigning near-zero probability to y_l                                   │
│  • Making y_w arbitrarily likely                                            │
│  → Degenerate policy, reward hacking                                        │
└─────────────────────────────────────────────────────────────────────────────┘

With reference policy (via implicit reward):
┌─────────────────────────────────────────────────────────────────────────────┐
│  r_θ = log(π_θ/π_ref) penalizes deviation from reference                   │
│  • Can't make y_l arbitrarily unlikely without penalty                      │
│  • Changes relative to π_ref, not absolute                                  │
│  → Stable, well-behaved optimization                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Pitfalls

### 1. Reference Policy Drift

**Problem**: If $\pi_\theta$ drifts too far from $\pi_{ref}$, the implicit reward becomes meaningless.

**Solution**: Monitor $D_{KL}(\pi_\theta \| \pi_{ref})$ during training; consider lower $\beta$.

### 2. Length Bias

**Problem**: Longer responses naturally have lower log probabilities.

**Solution**: Normalize by sequence length or use length-controlled training data.

### 3. Chosen/Rejected Imbalance

**Problem**: If $y_w$ is always much longer/shorter than $y_l$, model learns length instead of quality.

**Solution**: Ensure preference pairs have similar lengths.

### 4. Forgetting

**Problem**: Supervised training on preferences can cause catastrophic forgetting.

**Solution**: Mix in SFT data, use replay buffers.

---

## Quiz

<details>
<summary><strong>Q1: What is the key mathematical insight that enables DPO?</strong></summary>

**Answer**: The optimal policy for the RLHF objective has a closed form: $\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(r(x,y)/\beta)$. Rearranging gives us $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$. When we compute preference probabilities under the Bradley-Terry model, the partition function $Z(x)$ cancels, allowing us to express everything in terms of policy ratios.

**Key equation**: $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$

**Common misconception**: DPO is just "supervised learning on preferences" — it's actually solving the same constrained optimization as RLHF, just through a different path.

</details>

<details>
<summary><strong>Q2: Write out the DPO loss function and explain each term.</strong></summary>

**Answer**:
$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

Terms:
- $\sigma$: Sigmoid function, outputs preference probability
- $\beta$: Temperature parameter controlling preference sharpness
- $\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$: Implicit reward for preferred response
- $\log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$: Implicit reward for rejected response
- The difference gives relative preference under current policy

</details>

<details>
<summary><strong>Q3: Why does DPO need to keep the reference policy in memory?</strong></summary>

**Answer**: The reference policy serves two critical purposes:

1. **Defines implicit rewards**: $r_\theta(x,y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ requires computing log probabilities under $\pi_{ref}$

2. **Provides regularization**: Without the reference, the model could assign arbitrary probabilities. The ratio to reference implicitly penalizes deviation.

This is a key trade-off: DPO eliminates the reward model but requires keeping $\pi_{ref}$ in memory (doubling memory for the policy).

**Common misconception**: Some think DPO is "pure supervised learning" — but the reference policy connection to $\pi_{ref}$ is what makes it mathematically equivalent to KL-regularized RLHF.

</details>

<details>
<summary><strong>Q4: What does the gradient weight $\sigma(\hat{r}_l - \hat{r}_w)$ tell us about DPO's learning dynamics?</strong></summary>

**Answer**: The weight $\sigma(\hat{r}_l - \hat{r}_w)$ is high when the model incorrectly ranks the pair (assigns higher implicit reward to $y_l$ than $y_w$) and low when correctly ranked.

**Interpretation**:
- When $\hat{r}_w > \hat{r}_l$ (correct): $\sigma(\hat{r}_l - \hat{r}_w) \approx 0$ → small gradient
- When $\hat{r}_l > \hat{r}_w$ (incorrect): $\sigma(\hat{r}_l - \hat{r}_w) \approx 1$ → large gradient

This is **adaptive weighting** — DPO automatically focuses learning on examples where the model disagrees with human preferences, similar to hard negative mining.

**Pitfall**: If the model becomes overconfident on some pairs, it may stop learning from them entirely.

</details>

<details>
<summary><strong>Q5: How would you debug a DPO training run where the loss is decreasing but model quality is getting worse?</strong></summary>

**Answer**: This typically indicates one of several issues:

1. **Check for length gaming**: Is the model learning to prefer longer/shorter responses regardless of quality? Compare length distributions of chosen vs rejected.

2. **Monitor KL divergence**: If $D_{KL}(\pi_\theta \| \pi_{ref})$ is very large, the model may be overfitting to the preference data. Try lower $\beta$.

3. **Inspect implicit rewards**: Plot $r_\theta(y_w) - r_\theta(y_l)$ distribution. If it's saturating (all values very large positive), gradients vanish.

4. **Check data quality**: Preference labels may be noisy or contradictory.

5. **Evaluate on held-out data**: Training loss can decrease while test performance degrades (overfitting).

**Debugging strategy**: Start with β=0.1, monitor KL, use length normalization, maintain a validation set with human evaluation.

</details>

---

## References

1. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS.
2. Bradley, R. A., & Terry, M. E. (1952). "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons." Biometrika.
3. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." NeurIPS.

---

## What to Memorize for Interviews

| Concept | Key Point |
|---------|-----------|
| **Main idea** | Eliminate reward model using closed-form optimal policy |
| **Implicit reward** | $r_\theta(x,y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ |
| **DPO loss** | $\mathcal{L} = -\log \sigma(\beta(r_w - r_l))$ |
| **Why Z cancels** | Partition function cancels in Bradley-Terry preference comparison |
| **Gradient weight** | $\sigma(\hat{r}_l - \hat{r}_w)$ — focuses on misranked pairs |
| **vs PPO** | Simpler (no RL loop/reward model), but needs pairwise data |
| **Memory trade-off** | No reward model, but keep $\pi_{ref}$ in memory |
