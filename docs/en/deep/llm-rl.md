# RL Methods for LLM Training

!!! info "Page Reorganized"
    This content has been reorganized into individual pages for better navigation. Please visit the pages below:

## Individual Method Pages

The RLHF methods have been split into dedicated pages, ordered by publication date:

1. **[RLHF Overview](rlhf-overview.md)** — Introduction to RLHF
   - Why RL for LLM training
   - The RLHF pipeline
   - Method comparison and when to use each

2. **[TRPO (2015)](trpo.md)** — Trust Region Policy Optimization
   - Theoretical foundation of policy optimization
   - Monotonic improvement guarantees
   - Natural gradient and Fisher information matrix

3. **[DPO (2023)](dpo.md)** — Direct Preference Optimization
   - Bypassing the reward model
   - Implicit reward derivation
   - Bradley-Terry model connection

4. **[GRPO (2024)](grpo.md)** — Group Relative Policy Optimization
   - Group-based advantage estimation
   - No critic required
   - Connection to ranking losses

---

## Quick Reference

| Method | Year | Key Innovation | When to Use |
|--------|------|----------------|-------------|
| [TRPO](trpo.md) | 2015 | Trust region constraint | Maximum stability needed |
| [PPO](ppo.md) | 2017 | Clipped surrogate | General RLHF |
| [DPO](dpo.md) | 2023 | Implicit reward | Limited compute, offline data |
| [GRPO](grpo.md) | 2024 | Group normalization | Verifiable tasks (math/code) |

---

## Key Equations Summary

| Method | Core Equation |
|--------|---------------|
| **RLHF Objective** | $\max_\pi \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi \|\| \pi_{ref})$ |
| **TRPO** | $\max L^{CPI}$ s.t. $D_{KL} \leq \delta$ |
| **PPO** | $L = \min(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t)$ |
| **DPO** | $L = -\log\sigma(\beta(r_w - r_l))$ where $r = \log\frac{\pi_\theta}{\pi_{ref}}$ |
| **GRPO** | $A_i = (r_i - \mu) / \sigma$ (group-normalized) |
