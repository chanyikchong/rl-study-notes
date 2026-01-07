# Proximal Policy Optimization (PPO)

## Interview Summary

**PPO** is the most widely used deep RL algorithm. It improves on vanilla policy gradient by limiting policy updates to a "trust region" — preventing too-large steps that destabilize training. Two variants: **PPO-Clip** (clips the objective) and **PPO-Penalty** (KL penalty). PPO is simple, stable, and works well across many domains. Default choice for most practical applications.

**What to memorize**: PPO-Clip objective, why clipping helps, ratio \(r_t(\theta)\), typical hyperparameters.

---

## Core Definitions

### The Problem with Vanilla Policy Gradient

Large gradient steps can:
1. Change policy too much → destabilize training
2. Move to bad policies from which recovery is hard
3. Make the old samples invalid (data is on-policy)

### Probability Ratio

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**Meaning**: How much more likely is action \(a_t\) under new vs old policy?

- \(r = 1\): Same probability
- \(r > 1\): New policy likes this action more
- \(r < 1\): New policy likes this action less

### PPO-Clip Objective

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where \(\epsilon \approx 0.2\) is the clip range.

**Intuition**: If the policy changes too much (|r-1| > ε), we clip the objective so gradients don't push further in that direction.

---

## Math and Derivations

### Why Clipping Works

Consider cases for the min:

**Case 1: \(\hat{A}_t > 0\)** (good action)
- Want to increase \(\pi(a_t|s_t)\), so \(r\) increases
- But if \(r > 1 + \epsilon\), clipped term = \((1+\epsilon) \hat{A}\), which is constant
- Gradient becomes zero — stops pushing

**Case 2: \(\hat{A}_t < 0\)** (bad action)
- Want to decrease \(\pi(a_t|s_t)\), so \(r\) decreases
- But if \(r < 1 - \epsilon\), clipped term = \((1-\epsilon) \hat{A}\), constant
- Gradient becomes zero — stops pushing

The clip creates a "trust region" around the old policy.

### PPO Full Objective

$$L(\theta) = \mathbb{E}_t\left[ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

where:
- \(L^{VF}\): Value function loss (critic)
- \(S[\pi]\): Entropy bonus (exploration)
- \(c_1 \approx 0.5\), \(c_2 \approx 0.01\)

### Generalized Advantage Estimation (GAE)

PPO typically uses GAE for advantage estimation:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\).

This balances bias and variance (see GAE chapter).

---

## Algorithm Sketch

```
Algorithm: PPO-Clip

Hyperparameters: ε (clip), γ, λ (GAE), K (epochs), M (minibatch size)

1. Initialize θ (policy), φ (value function)
2. For iteration = 1, 2, ...:
     # Collect trajectories
     Run policy π_θ for T timesteps across N parallel actors
     Compute advantages Â_t using GAE with V_φ
     Compute returns: R_t = Â_t + V_φ(s_t)

     # Store old policy probs
     For all (s_t, a_t): π_old(a_t|s_t) = π_θ(a_t|s_t)

     # Multiple epochs of updates
     For k = 1 to K:
         For each minibatch of size M:
             Compute ratio: r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)

             # Clipped surrogate objective
             L_clip = min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t)

             # Value loss
             L_vf = (V_φ(s_t) - R_t)²

             # Entropy bonus
             L_ent = -Σ π_θ(a|s) log π_θ(a|s)

             # Total loss
             L = -L_clip + c1·L_vf - c2·L_ent

             Gradient step on L
```

### Key Hyperparameters

| Hyperparameter | Typical Value |
|----------------|---------------|
| Clip range ε | 0.1 - 0.3 |
| GAE λ | 0.95 |
| Discount γ | 0.99 |
| Learning rate | 3e-4 |
| Epochs K | 3 - 10 |
| Minibatch size | 64 - 256 |
| Parallel actors | 8 - 64 |

---

## Common Pitfalls

1. **Clip range too large**: Policy can change too much, causing instability.

2. **Too many epochs**: Overfitting to current batch; policy moves too far.

3. **Value function not fit well**: Advantages become noisy; use more value updates.

4. **Forgetting to normalize advantages**: Normalize \(\hat{A}\) per batch for stability.

5. **Learning rate too high**: PPO is sensitive to learning rate; start low.

6. **Not using GAE**: Raw TD error or MC returns work but GAE is usually better.

---

## Mini Example

**CartPole with PPO:**

```python
# Collect batch
for _ in range(batch_size):
    action, log_prob = sample_action(policy, state)
    next_state, reward, done = env.step(action)
    buffer.store(state, action, reward, log_prob, done)
    state = next_state

# Compute advantages (GAE)
advantages = compute_gae(buffer.rewards, buffer.values, gamma, lam)
returns = advantages + buffer.values

# PPO updates (multiple epochs)
for epoch in range(num_epochs):
    for batch in buffer.iterate_minibatches(minibatch_size):
        # Compute new log probs
        new_log_probs = policy.log_prob(batch.states, batch.actions)

        # Ratio
        ratio = exp(new_log_probs - batch.old_log_probs)

        # Clipped objective
        surr1 = ratio * batch.advantages
        surr2 = clip(ratio, 1-eps, 1+eps) * batch.advantages
        policy_loss = -min(surr1, surr2).mean()

        # Value loss
        value_loss = MSE(critic(batch.states), batch.returns)

        # Entropy bonus
        entropy = policy.entropy(batch.states).mean()

        # Total loss
        loss = policy_loss + c1 * value_loss - c2 * entropy
        loss.backward()
        optimizer.step()
```

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why is PPO more stable than vanilla policy gradient?</summary>

**Answer**: PPO limits how much the policy can change per update via clipping. Large changes are prevented, keeping updates in a "trust region."

**Explanation**: In vanilla PG, a large gradient step can change the policy drastically, making old samples invalid and potentially moving to a bad policy. PPO's clipping ensures small, conservative updates.

**Key insight**: The clip creates a flat region in the objective when the policy moves too far, stopping the gradient.

**Common pitfall**: Thinking clipping is just for numerical stability. It's fundamentally about limiting policy change.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> What does the ratio r_t(θ) represent?</summary>

**Answer**: The ratio of action probability under the new policy vs the old policy.

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**Explanation**:
- \(r = 1\): Policies are the same for this action
- \(r > 1\): New policy is more likely to take this action
- \(r < 1\): New policy is less likely

**Key insight**: Multiplying advantage by \(r\) gives importance-weighted policy gradient.

**Common pitfall**: Forgetting to store old policy probabilities before starting updates.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Explain how the clipping mechanism works for positive and negative advantages.</summary>

**Answer**:

**Positive advantage** (\(\hat{A} > 0\)): We want to increase action probability.
- Objective is \(\min(r \cdot \hat{A}, \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot \hat{A})\)
- If \(r > 1 + \epsilon\): clipped term wins, gradient = 0
- If \(r < 1 + \epsilon\): unclipped term wins, gradient pushes \(r\) up
- Effect: Stop pushing once policy has increased enough

**Negative advantage** (\(\hat{A} < 0\)): We want to decrease action probability.
- If \(r < 1 - \epsilon\): clipped term wins (less negative), gradient = 0
- If \(r > 1 - \epsilon\): unclipped term wins, gradient pushes \(r\) down
- Effect: Stop pushing once policy has decreased enough

**Key equation**: \(L^{CLIP} = \min(r \hat{A}, \text{clip}(r) \hat{A})\)

**Common pitfall**: Getting the min/max logic wrong. Draw it out!
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Why can PPO reuse samples for multiple gradient steps?</summary>

**Answer**: Because the ratio \(r_t\) corrects for the distribution mismatch between old and new policy.

**Explanation**: Samples were collected with \(\pi_{old}\). Using them with \(\pi_\theta\) requires importance weighting:

$$\mathbb{E}_{a \sim \pi_{old}}[f(a) \frac{\pi_\theta(a)}{\pi_{old}(a)}] = \mathbb{E}_{a \sim \pi_\theta}[f(a)]$$

The ratio provides this correction. Clipping prevents weights from becoming too extreme.

**Key insight**: This makes PPO more sample-efficient than vanilla PG.

**Common pitfall**: Reusing samples for too many epochs — the correction becomes inaccurate.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> PPO is not learning well. What hyperparameters to tune first?</summary>

**Answer**: Tuning priority:
1. **Learning rate**: Try 1e-4 to 3e-4; too high causes instability
2. **Clip range ε**: Try 0.1, 0.2, 0.3
3. **Number of epochs K**: Fewer epochs if overfitting (policy changes too much)
4. **GAE λ**: Lower λ (0.9) for less variance, higher (0.99) for less bias
5. **Batch size**: Larger batches give more stable gradients
6. **Entropy coefficient**: Increase if policy collapses

**Explanation**: PPO is sensitive to learning rate and clip range. Start with paper defaults, then tune.

**Common pitfall**: Tuning too many things at once. Change one hyperparameter at a time.
</details>

---

## References

- **Schulman et al. (2017)**, Proximal Policy Optimization Algorithms
- **Schulman et al. (2016)**, High-Dimensional Continuous Control Using GAE
- **Engstrom et al. (2020)**, Implementation Matters in Deep RL

**What to memorize for interviews**: PPO-Clip objective, ratio definition, clipping logic for positive/negative advantages, typical hyperparameters, why it's stable.

**Code example**: [ppo.py](../../../rl_examples/algorithms/ppo.py)
