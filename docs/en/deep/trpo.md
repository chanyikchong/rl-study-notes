# TRPO: Trust Region Policy Optimization (2015)

## Interview Summary

**TRPO** provides the theoretical foundation for stable policy optimization. It guarantees **monotonic improvement** by constraining updates to a "trust region" where $D_{KL} \leq \delta$. The algorithm optimizes a surrogate objective using the **natural gradient** (Fisher matrix inverse). While theoretically sound, TRPO is computationally expensive due to Fisher matrix computation and line search. PPO was created to simplify TRPO while retaining its stability.

**What to memorize**: Trust region constraint $D_{KL} \leq \delta$, surrogate objective $L^{CPI}$, why it motivated PPO.

---

## The Foundation

TRPO (Schulman et al., 2015) asks: *How can we improve the policy while guaranteeing we don't make it worse?*

This was a breakthrough because vanilla policy gradient has no such guarantee — a single bad update can permanently destroy a good policy.

---

## Motivation: The Policy Improvement Problem

In standard policy gradient, large updates can catastrophically degrade performance:

$$\theta_{new} = \theta_{old} + \alpha \nabla_\theta J(\theta)$$

**The problem**: How do we choose $\alpha$?
- Too large → policy collapse
- Too small → slow learning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE CATASTROPHIC UPDATE PROBLEM                          │
└─────────────────────────────────────────────────────────────────────────────┘

In supervised learning:
  Bad gradient step → next batch still valid → can recover

In reinforcement learning:
  Bad gradient step → policy collects bad data → bad gradients → worse policy
                                      ↓
                            VICIOUS CYCLE → COLLAPSE

┌─────────────────────────────────────────────────────────────────────────────┐
│  Policy Quality                                                             │
│       │                                                                     │
│   100 ┼───────────╮                                                        │
│       │            ╲                                                        │
│    80 ┼             ╲                                                       │
│       │              ╲  ← One bad update                                    │
│    60 ┼               ╲                                                     │
│       │                ╲                                                    │
│    40 ┼                 ╲____                                               │
│       │                      ╲____                                          │
│    20 ┼                           ╲_____                                    │
│       │                                  ╲_______                           │
│     0 ┼─────────────────────────────────────────→ Time                     │
│       │                                                                     │
│       │  Once collapsed, may NEVER recover                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Surrogate Objective

TRPO optimizes a **surrogate objective** that lower-bounds the true improvement:

$$L^{CPI}(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t \right] = \mathbb{E}_t \left[ r_t(\theta) A_t \right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the **probability ratio**.

**Intuition**:
- If $A_t > 0$ (good action): we want to increase $\pi_\theta(a_t|s_t)$, making $r_t > 1$
- If $A_t < 0$ (bad action): we want to decrease $\pi_\theta(a_t|s_t)$, making $r_t < 1$

---

## The Trust Region Constraint

TRPO constrains the KL divergence to ensure monotonic improvement:

$$\max_\theta \quad L^{CPI}(\theta)$$
$$\text{s.t.} \quad \mathbb{E}_t \left[ D_{KL}\left(\pi_{\theta_{old}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t)\right) \right] \leq \delta$$

**Key Insight**: Within this "trust region" (where $D_{KL} \leq \delta$), the surrogate objective accurately predicts the true objective.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRUST REGION VISUALIZATION                          │
└─────────────────────────────────────────────────────────────────────────────┘

                    Policy Space
                         │
                         │
                    ┌────┴────┐
               ╭────│ θ_old   │────╮
              ╱     └─────────┘     ╲
             ╱                       ╲
            ╱    Trust Region         ╲
           │     (D_KL ≤ δ)           │
           │                          │
           │  ✓ Surrogate objective   │
           │    accurately predicts   │
           │    true improvement      │
            ╲                        ╱
             ╲                      ╱
              ╲                    ╱
               ╰─────────────────╯
                        │
                        │
            Outside: Surrogate may be wrong!
```

---

## Mathematical Derivation

### Step 1: Performance Difference Lemma

The difference between two policies can be expressed as:

$$J(\pi_{new}) - J(\pi_{old}) = \mathbb{E}_{s \sim d^{\pi_{new}}, a \sim \pi_{new}} \left[ A^{\pi_{old}}(s, a) \right]$$

Where $d^{\pi_{new}}$ is the state distribution under the new policy.

### Step 2: The Problem

We can't sample from $d^{\pi_{new}}$ because we don't have the new policy yet!

### Step 3: The Approximation

TRPO uses the old state distribution with importance sampling:

$$L^{CPI}(\theta) = \mathbb{E}_{s \sim d^{\pi_{old}}, a \sim \pi_{old}} \left[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A^{\pi_{old}}(s, a) \right]$$

### Step 4: Bounding the Error

The error introduced by using the wrong state distribution is bounded when $D_{KL}$ is small:

$$|J(\pi_{new}) - J(\pi_{old}) - L^{CPI}(\theta)| \leq C \cdot \sqrt{\mathbb{E}_s[D_{KL}(\pi_{old} \| \pi_{new})]}$$

**This is the key theorem**: If we keep $D_{KL}$ small, the surrogate $L^{CPI}$ is a good approximation of the true improvement!

---

## TRPO Algorithm

```
TRPO Algorithm:
═══════════════════════════════════════════════════════════════════════════════
Input: Initial policy π_θ, trust region size δ

for iteration = 1, 2, ... do:

    ┌─ DATA COLLECTION ───────────────────────────────────────────────────────┐
    │  1. Collect trajectories using current policy π_θ                       │
    │  2. Compute advantages A_t for all timesteps (using GAE or MC)          │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ GRADIENT COMPUTATION ──────────────────────────────────────────────────┐
    │  3. Compute policy gradient:                                            │
    │     g = ∇_θ L^CPI(θ)|_{θ=θ_old}                                        │
    │                                                                         │
    │  4. Compute Fisher Information Matrix:                                  │
    │     F = E[∇_θ log π_θ · (∇_θ log π_θ)^T]                               │
    │     (approximates Hessian of KL divergence)                             │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ NATURAL GRADIENT ──────────────────────────────────────────────────────┐
    │  5. Compute natural gradient direction:                                 │
    │     d = F^{-1} g                                                        │
    │                                                                         │
    │     (Use conjugate gradient to avoid explicit inversion)                │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ LINE SEARCH ───────────────────────────────────────────────────────────┐
    │  6. Find step size β via backtracking line search:                      │
    │                                                                         │
    │     Start with β = √(2δ / (g^T F^{-1} g))                              │
    │                                                                         │
    │     While NOT satisfied:                                                │
    │       - Check: D_KL(π_θ_old || π_{θ_old + βd}) ≤ δ  ?                  │
    │       - Check: L^CPI(θ_old + βd) > L^CPI(θ_old)  ?                     │
    │       - If not: β ← β / 2                                              │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ UPDATE ────────────────────────────────────────────────────────────────┐
    │  7. Update parameters: θ ← θ_old + βd                                   │
    └─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
```

---

## Solving the Constrained Optimization

Using Lagrangian methods:

$$\mathcal{L}(\theta, \lambda) = L^{CPI}(\theta) - \lambda \left( D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) - \delta \right)$$

The optimal step uses the **natural gradient**:

$$\theta_{new} = \theta_{old} + \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g$$

Where $F$ is the Fisher Information Matrix.

### What is the Fisher Information Matrix?

$$F = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)^T \right]$$

**Intuition**: $F$ measures how sensitive the policy distribution is to parameter changes. The natural gradient $F^{-1}g$ gives a direction that accounts for the geometry of the policy space.

### Why Natural Gradient?

Standard gradient descent assumes Euclidean geometry in parameter space. But for probability distributions, a small change in parameters can cause a large change in the distribution (or vice versa).

```
Euclidean vs Natural Gradient:
─────────────────────────────────────────────────────────────────────────────

Standard Gradient:                  Natural Gradient:
┌─────────────────────────┐        ┌─────────────────────────┐
│  Takes equal steps      │        │  Steps scaled by        │
│  in parameter space     │        │  distribution geometry  │
│                         │        │                         │
│  θ_new = θ - α∇L       │        │  θ_new = θ - αF^{-1}∇L │
│                         │        │                         │
│  May over/undershoot    │        │  More stable updates    │
│  in distribution space  │        │  in distribution space  │
└─────────────────────────┘        └─────────────────────────┘
```

---

## Complexity Issues

TRPO requires three expensive operations:

### 1. Fisher Matrix Computation

$$F = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta \cdot (\nabla_\theta \log \pi_\theta)^T \right]$$

- Size: $|\theta| \times |\theta|$ — for a network with millions of parameters, this is huge!
- Cannot store explicitly for large networks

### 2. Matrix-Vector Product $F^{-1}g$

- Direct inversion: $O(|\theta|^3)$ — completely infeasible
- **Solution**: Use Conjugate Gradient (CG)
  - Only need matrix-vector products $Fv$
  - These can be computed efficiently via autodiff
  - Still requires $k$ iterations (typically 10-20)

### 3. Line Search

- Need to verify KL constraint is satisfied
- Requires multiple forward passes through network
- Adds significant overhead

**Total Complexity**: Much higher than vanilla policy gradient!

---

## TRPO vs PPO

This complexity motivated the development of PPO:

| Aspect | TRPO | PPO |
|--------|------|-----|
| Constraint | Hard KL constraint | Soft clipping |
| Fisher matrix | Required | Not needed |
| Conjugate gradient | Required | Not needed |
| Line search | Required | Not needed |
| Guarantees | Monotonic improvement | Empirically similar |
| Implementation | Complex | Simple |

**PPO achieves similar performance with much simpler implementation** — this is why PPO became the standard for RLHF.

---

## When to Use TRPO

Despite its complexity, TRPO is still useful when:

1. **Maximum stability needed**: Safety-critical applications where you cannot afford policy collapse
2. **Theoretical analysis**: Research requiring formal guarantees
3. **Small networks**: When computational overhead is acceptable

For most practical applications, **PPO is preferred**.

---

## Quiz

<details>
<summary><strong>Q1: What problem does TRPO solve that vanilla policy gradient doesn't?</strong></summary>

**Answer**: TRPO provides **monotonic improvement guarantees**. Vanilla policy gradient has no step size control — a single large update can catastrophically collapse the policy. TRPO constrains updates to a "trust region" where the surrogate objective accurately predicts true improvement.

**Key equation**: $\max L^{CPI}(\theta)$ s.t. $D_{KL} \leq \delta$

</details>

<details>
<summary><strong>Q2: Why can't we directly maximize the true objective $J(\pi_{new}) - J(\pi_{old})$?</strong></summary>

**Answer**: The true objective requires sampling from $d^{\pi_{new}}$ (the state distribution under the new policy), but we don't have the new policy yet — we're trying to find it!

TRPO approximates this by using the old state distribution with importance sampling, and bounds the error via the KL constraint.

</details>

<details>
<summary><strong>Q3: What is the Fisher Information Matrix and why is it used?</strong></summary>

**Answer**: The Fisher matrix $F = \mathbb{E}[\nabla \log \pi \cdot (\nabla \log \pi)^T]$ measures how sensitive the policy distribution is to parameter changes.

It's used to compute the **natural gradient** $F^{-1}g$, which accounts for the geometry of probability distributions. This gives more stable updates than the standard gradient.

</details>

<details>
<summary><strong>Q4: Why did TRPO motivate the development of PPO?</strong></summary>

**Answer**: TRPO is computationally expensive because it requires:
1. Computing/inverting the Fisher matrix
2. Conjugate gradient iterations
3. Line search to satisfy constraint

PPO achieves similar stability through simple clipping, eliminating all three expensive operations. This made it practical for large-scale applications like RLHF.

</details>

---

## References

1. Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). "Trust Region Policy Optimization." ICML.
2. Kakade, S., & Langford, J. (2002). "Approximately Optimal Approximate Reinforcement Learning."
3. Amari, S. (1998). "Natural Gradient Works Efficiently in Learning." Neural Computation.

---

## What to Memorize for Interviews

| Concept | Key Point |
|---------|-----------|
| **Main idea** | Constrain updates to trust region where $D_{KL} \leq \delta$ |
| **Surrogate objective** | $L^{CPI} = \mathbb{E}[r_t(\theta) A_t]$ |
| **Constraint** | $\mathbb{E}[D_{KL}(\pi_{old} \| \pi_\theta)] \leq \delta$ |
| **Natural gradient** | $\theta_{new} = \theta_{old} + \sqrt{2\delta / g^T F^{-1} g} \cdot F^{-1}g$ |
| **Why complex** | Fisher matrix, conjugate gradient, line search |
| **Relation to PPO** | PPO simplifies TRPO with clipping |
