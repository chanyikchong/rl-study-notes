# Proximal Policy Optimization (PPO)

## Interview Summary

**PPO** is the most widely used deep RL algorithm. It improves on vanilla policy gradient by limiting policy updates to a "trust region" — preventing too-large steps that destabilize training. Two variants: **PPO-Clip** (clips the objective) and **PPO-Penalty** (KL penalty). PPO is simple, stable, and works well across many domains. Default choice for most practical applications.

**What to memorize**: PPO-Clip objective, why clipping helps, ratio $r_t(\theta)$, typical hyperparameters.

---

## Design Motivation: Why PPO Exists

### The Fundamental Problem

Policy gradient methods have a critical flaw: **step size selection is extremely difficult**.

```
Too small step → Very slow learning
Too large step → Policy collapses, may never recover
```

**Why is this hard?**

In supervised learning, if you take a bad gradient step, your next batch of data is still valid. In RL, the data comes FROM your policy. If you break the policy:

1. The new policy collects bad data
2. Bad data leads to bad gradients
3. Bad gradients make the policy worse
4. **Vicious cycle → Training collapses**

### The Core Insight

> **Key Idea**: We want to improve the policy, but we must ensure the new policy stays "close" to the old one.

This is the **Trust Region** concept: only trust your gradient within a small region around the current policy.

### Evolution of Ideas

```
Vanilla Policy Gradient
    ↓ Problem: No step size control
    ↓
TRPO (Trust Region Policy Optimization)
    ↓ Solution: Hard KL constraint
    ↓ Problem: Complex, requires conjugate gradient
    ↓
PPO (Proximal Policy Optimization)
    ↓ Solution: Soft constraint via clipping
    ✓ Simple, stable, nearly as good as TRPO
```

---

## Core Definitions

### Why "Proximal"?

"Proximal" means "nearby" — PPO keeps the new policy proximal (close) to the old policy. This prevents catastrophic policy updates.

### The Problem with Vanilla Policy Gradient

Standard policy gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a) \right]$$

**Problems:**
1. **Large gradient steps can change policy drastically** → Training destabilizes
2. **Old samples become invalid** → Data collected with $\pi_{old}$ doesn't represent $\pi_{new}$
3. **No way to control "how different" the new policy is**

### Measuring Policy Change: The Probability Ratio

How do we measure if the policy changed? Compare action probabilities:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**Intuitive meaning:**

| Ratio $r$ | Interpretation |
|-----------|----------------|
| $r = 1$ | New policy identical to old for this action |
| $r = 1.5$ | New policy is 50% more likely to take this action |
| $r = 0.5$ | New policy is 50% less likely to take this action |
| $r = 2$ | ⚠️ Policy changed a lot — might be dangerous |
| $r = 0.1$ | ⚠️ Policy changed a lot — might be dangerous |

### The PPO-Clip Objective

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $\epsilon \approx 0.2$ is the clip range.

**In plain English:**
- Use the policy gradient (ratio × advantage)
- BUT if the policy has changed too much ($r$ far from 1), clip it
- Clipping removes the incentive to change the policy further

---

## Math and Derivations

### Understanding the Clipping Mechanism

The genius of PPO is in the $\min$ operation. Let's work through it carefully.

#### Case 1: Good action ($\hat{A}_t > 0$)

We want to **increase** this action's probability.

$$L = \min\left( r \cdot \hat{A}, \; \text{clip}(r, 0.8, 1.2) \cdot \hat{A} \right)$$

| If $r$ is... | Unclipped term | Clipped term | $\min$ selects | Gradient |
|--------------|----------------|--------------|----------------|----------|
| $r = 0.9$ | $0.9 \hat{A}$ | $0.9 \hat{A}$ | Same | Push $r$ up ✓ |
| $r = 1.1$ | $1.1 \hat{A}$ | $1.1 \hat{A}$ | Same | Push $r$ up ✓ |
| $r = 1.3$ | $1.3 \hat{A}$ | $1.2 \hat{A}$ | **Clipped** (smaller) | **Zero** gradient 🛑 |

**Intuition**: Once you've increased the action probability enough (r > 1.2), STOP. Don't get greedy.

#### Case 2: Bad action ($\hat{A}_t < 0$)

We want to **decrease** this action's probability.

| If $r$ is... | Unclipped term | Clipped term | $\min$ selects | Gradient |
|--------------|----------------|--------------|----------------|----------|
| $r = 1.1$ | $1.1 \hat{A}$ (neg) | $1.1 \hat{A}$ | Same | Push $r$ down ✓ |
| $r = 0.9$ | $0.9 \hat{A}$ (neg) | $0.9 \hat{A}$ | Same | Push $r$ down ✓ |
| $r = 0.7$ | $0.7 \hat{A}$ (neg) | $0.8 \hat{A}$ (neg) | **Clipped** (less negative) | **Zero** gradient 🛑 |

**Intuition**: Once you've decreased the action probability enough (r < 0.8), STOP. Don't overdo it.

### Visualizing the Clipped Objective

```
        L^CLIP
          ^
          |      ___________  (clipped: no more gradient)
          |     /
          |    /
          |   /
          |  /
          | /
    ------+/--------|--------|--------> r (ratio)
         /|       1-ε      1+ε
        / |
   ____/  |  (clipped: no more gradient)
          |

When A > 0: Clip prevents r from going above 1+ε
When A < 0: Clip prevents r from going below 1-ε
```

### Why Clipping is Brilliant

**TRPO approach**: Solve a constrained optimization problem with KL divergence constraint. Requires:
- Computing Fisher information matrix
- Conjugate gradient solver
- Line search

**PPO approach**: Just clip the objective. The clipping naturally creates a "soft" trust region:
- Simple to implement (one extra `min` and `clip`)
- First-order optimization (just SGD/Adam)
- Nearly as effective as TRPO in practice

### The Full PPO Objective

$$L(\theta) = \mathbb{E}_t\left[ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

| Term | Purpose | Typical weight |
|------|---------|----------------|
| $L^{CLIP}$ | Policy improvement with trust region | 1.0 |
| $L^{VF}$ | Value function accuracy (for advantage estimation) | $c_1 = 0.5$ |
| $S[\pi]$ | Entropy bonus (encourages exploration) | $c_2 = 0.01$ |

### Why Multiple Epochs Work

**Vanilla PG**: One gradient step per batch, then throw away data. Wasteful!

**PPO**: Reuse the same batch for K gradient steps. Why is this okay?

The ratio $r_t(\theta)$ provides **importance sampling correction**:

$$\mathbb{E}_{a \sim \pi_{old}}\left[f(a) \cdot \frac{\pi_\theta(a)}{\pi_{old}(a)}\right] = \mathbb{E}_{a \sim \pi_\theta}[f(a)]$$

The clipping prevents the importance weights from becoming extreme (which would cause high variance).

**Result**: PPO is much more sample-efficient than vanilla PG.

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

     # Multiple epochs of updates (KEY: reuse data!)
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

### Key Design Choices Explained

| Design Choice | Why? |
|--------------|------|
| Multiple epochs (K) | Sample efficiency — reuse data |
| Minibatches | Stable gradients, fits in GPU memory |
| GAE for advantages | Balance bias-variance in advantage estimation |
| Parallel actors | Diverse data, faster collection |
| Entropy bonus | Prevent premature convergence to deterministic policy |
| Shared policy-value network | Efficiency, shared representations |

### Key Hyperparameters

| Hyperparameter | Typical Value | Effect if too high | Effect if too low |
|----------------|---------------|-------------------|-------------------|
| Clip range $\epsilon$ | 0.1 - 0.3 | Large policy changes | Too conservative |
| GAE $\lambda$ | 0.95 | High variance | High bias |
| Discount $\gamma$ | 0.99 | Long-term focus | Myopic |
| Learning rate | 3e-4 | Instability | Slow learning |
| Epochs K | 3 - 10 | Overfitting to batch | Sample inefficient |
| Minibatch size | 64 - 256 | Slow updates | Noisy gradients |

---

## Common Pitfalls

1. **Clip range too large ($\epsilon > 0.3$)**: Policy can still change too much between iterations. Start with $\epsilon = 0.2$.

2. **Too many epochs (K > 10)**: The policy overfits to the current batch; ratio correction becomes inaccurate. Monitor KL divergence between old and new policy.

3. **Value function not fit well**: Noisy advantages lead to noisy gradients. Consider more value function updates or a separate value network.

4. **Forgetting to normalize advantages**: Normalize $\hat{A}$ to zero mean, unit variance per batch. Huge stability improvement.

5. **Learning rate too high**: PPO is sensitive. Start with 3e-4 or lower, use learning rate annealing.

6. **Not using GAE**: Raw TD or Monte Carlo returns work but GAE usually performs better. Use $\lambda = 0.95$ as default.

7. **Forgetting to store old log probs**: Must store $\log \pi_{old}(a_t|s_t)$ BEFORE the update loop starts!

---

## PPO vs Other Algorithms

| Aspect | Vanilla PG | TRPO | PPO |
|--------|------------|------|-----|
| Step size control | None | Hard KL constraint | Soft (clipping) |
| Sample efficiency | Low (1 step/batch) | Low | High (K steps/batch) |
| Implementation | Simple | Complex | Simple |
| Stability | Low | High | High |
| Computation | Low | High (2nd order) | Low (1st order) |

**When to use PPO**: Almost always the default choice for continuous control, games, robotics, and LLM fine-tuning (RLHF).

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
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize!
returns = advantages + buffer.values

# PPO updates (multiple epochs)
for epoch in range(num_epochs):
    for batch in buffer.iterate_minibatches(minibatch_size):
        # Compute new log probs
        new_log_probs = policy.log_prob(batch.states, batch.actions)

        # Ratio (in log space for numerical stability)
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
<summary><strong>Q1 (Conceptual):</strong> Why does PPO exist? What problem does it solve?</summary>

**Answer**: PPO solves the step size problem in policy gradient methods. Without step size control, large updates can destabilize training because:
1. A bad policy update affects future data collection
2. This creates a vicious cycle leading to training collapse

**Explanation**: Unlike supervised learning where data is fixed, in RL the data comes from the policy. PPO keeps updates "proximal" (close) to the old policy, preventing catastrophic changes.

**Key insight**: The "trust region" concept — only trust your gradient estimate within a small region around the current policy.

**Common pitfall**: Thinking PPO is just "better" than vanilla PG. It solves a specific, critical problem.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Explain the probability ratio and what different values mean.</summary>

**Answer**: The ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ measures how much the policy changed for a specific action.

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**Interpretation:**
- $r = 1$: Policies identical for this action
- $r > 1$: New policy MORE likely to take this action
- $r < 1$: New policy LESS likely to take this action
- $r \gg 1$ or $r \ll 1$: Policy changed a lot — potentially dangerous!

**Key insight**: Multiplying advantage by $r$ gives importance-weighted policy gradient, allowing sample reuse.

**Common pitfall**: Forgetting to store old policy probabilities before starting updates.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Walk through the clipping mechanism for both positive and negative advantages.</summary>

**Answer**:

**Positive advantage** ($\hat{A} > 0$): We want to increase action probability.
- Objective: $\min(r \cdot \hat{A}, \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot \hat{A})$
- If $r > 1 + \epsilon$: clipped term $(1+\epsilon)\hat{A}$ is smaller, gradient = 0
- Effect: Stop increasing probability once $r$ exceeds $1+\epsilon$

**Negative advantage** ($\hat{A} < 0$): We want to decrease action probability.
- If $r < 1 - \epsilon$: clipped term $(1-\epsilon)\hat{A}$ is less negative, gradient = 0
- Effect: Stop decreasing probability once $r$ falls below $1-\epsilon$

**Key equation**: $L^{CLIP} = \min(r \hat{A}, \text{clip}(r) \hat{A})$

**Common pitfall**: Getting the min/max logic confused. Remember: we're MAXIMIZING the objective, so $\min$ is pessimistic (conservative).
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Why can PPO reuse samples for multiple gradient steps while vanilla PG cannot?</summary>

**Answer**: PPO uses importance sampling via the ratio $r_t$ to correct for the distribution mismatch.

Samples were collected with $\pi_{old}$, but we want gradients for $\pi_\theta$:

$$\mathbb{E}_{a \sim \pi_{old}}\left[f(a) \cdot \frac{\pi_\theta(a)}{\pi_{old}(a)}\right] = \mathbb{E}_{a \sim \pi_\theta}[f(a)]$$

The ratio provides this correction. Clipping prevents the weights from becoming extreme (which would cause high variance).

**Key insight**: This makes PPO 3-10x more sample-efficient than vanilla PG.

**Common pitfall**: Reusing samples for too many epochs — the correction becomes inaccurate as $\pi_\theta$ drifts far from $\pi_{old}$.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> PPO training is unstable. List debugging steps in priority order.</summary>

**Answer**: Debugging priority:

1. **Normalize advantages** — Most common fix! Use per-batch normalization.
2. **Lower learning rate** — Try 1e-4 instead of 3e-4
3. **Reduce clip range $\epsilon$** — Try 0.1 instead of 0.2
4. **Reduce epochs K** — Try 3 instead of 10
5. **Check value function** — Plot value loss; if high, train more
6. **Monitor KL divergence** — If KL > 0.1 between updates, policy changing too fast
7. **Increase batch size** — More stable gradient estimates
8. **Add entropy bonus** — If policy becomes deterministic too fast

**Explanation**: PPO is sensitive to learning rate and clip range. Always start with published defaults.

**Common pitfall**: Tuning too many hyperparameters at once. Change one thing at a time!
</details>

<details markdown="1">
<summary><strong>Q6 (Conceptual):</strong> Compare PPO to TRPO. Why is PPO preferred in practice?</summary>

**Answer**:

| Aspect | TRPO | PPO |
|--------|------|-----|
| Constraint | Hard KL constraint | Soft (clipping) |
| Optimization | 2nd order (Fisher matrix, conjugate gradient) | 1st order (SGD/Adam) |
| Implementation | ~500 lines | ~100 lines |
| Performance | Slightly better in some cases | Nearly as good |

**Why PPO wins:**
1. **Simpler**: No conjugate gradient, no line search
2. **Faster**: First-order methods are cheaper per step
3. **Good enough**: Performance gap is small in most tasks

**Key insight**: Engineering simplicity often beats theoretical elegance.

**Common pitfall**: Assuming TRPO is always better because it has "guarantees."
</details>

---

## References

- **Schulman et al. (2017)**, Proximal Policy Optimization Algorithms
- **Schulman et al. (2016)**, High-Dimensional Continuous Control Using GAE
- **Schulman et al. (2015)**, Trust Region Policy Optimization (TRPO)
- **Engstrom et al. (2020)**, Implementation Matters in Deep RL

**What to memorize for interviews**: PPO-Clip objective, ratio definition, clipping logic for positive/negative advantages, why multiple epochs work, comparison to TRPO, typical hyperparameters.

**Code example**: [ppo.py](../../../rl_examples/algorithms/ppo.py)
