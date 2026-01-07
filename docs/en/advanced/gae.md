# Generalized Advantage Estimation (GAE)

## Interview Summary

**GAE** is a family of advantage estimators that interpolate between high-bias (TD error) and high-variance (Monte Carlo) estimates. Controlled by parameter \(\lambda \in [0,1]\). The GAE formula: \(\hat{A}^{GAE}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}\). When \(\lambda=0\), GAE equals TD error. When \(\lambda=1\), GAE equals MC return minus baseline. Used in PPO, A3C, and most modern policy gradient methods.

**What to memorize**: GAE formula, bias-variance tradeoff, effect of λ, typical λ=0.95.

---

## Core Definitions

### The Advantage Estimation Problem

We want to estimate:

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

But we don't know \(Q^\pi\) or \(V^\pi\) exactly. Different estimators have different bias-variance properties.

### TD Error

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- **Bias**: If \(V\) is wrong, estimate is biased
- **Variance**: Low (only one random reward)

### Monte Carlo Advantage

$$\hat{A}^{MC}_t = G_t - V(s_t) = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)$$

- **Bias**: Zero (by definition of \(V^\pi\))
- **Variance**: High (sum of many rewards)

### GAE Definition

$$\hat{A}^{GAE(\gamma,\lambda)}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\).

---

## Math and Derivations

### Expanding GAE

$$\hat{A}^{GAE}_t = \delta_t + \gamma\lambda \delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \cdots$$

Substituting \(\delta_l\):

$$= (r_t + \gamma V(s_{t+1}) - V(s_t))$$

$$+ \gamma\lambda(r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1}))$$

$$+ (\gamma\lambda)^2(r_{t+2} + \gamma V(s_{t+3}) - V(s_{t+2}))$$

$$+ \cdots$$

### Special Cases

**λ = 0**:

$$\hat{A}^{GAE(\gamma,0)}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Just the TD error — low variance, potentially high bias.

**λ = 1**:

$$\hat{A}^{GAE(\gamma,1)}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}$$

This telescopes to:

$$= \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$

Monte Carlo advantage — zero bias, high variance.

### Bias-Variance Tradeoff

| λ | Bias | Variance | Behavior |
|---|------|----------|----------|
| 0 | High (if V wrong) | Low | TD error only |
| 0.5 | Medium | Medium | Balanced |
| 0.95 | Low | Medium-high | Near MC but some bias reduction |
| 1 | Zero (in expectation) | High | Full MC |

### Practical Computation

For finite-horizon episodes (length T):

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t-1}\delta_{T-1}$$

Computed recursively backward:

$$\hat{A}_{T-1} = \delta_{T-1}$$

$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$

---

## Algorithm Sketch

```
Algorithm: Compute GAE Advantages

Input: rewards r[], values V[], γ, λ, done[]
Output: advantages A[]

T = length of episode
A[T] = 0  # Bootstrap at terminal

For t = T-1 down to 0:
    If done[t+1]:
        delta = r[t] - V[t]  # No bootstrap at terminal
        A[t+1] = 0
    Else:
        delta = r[t] + γ * V[t+1] - V[t]

    A[t] = delta + γ * λ * A[t+1]

Return A[0:T]
```

### Integration with PPO

```python
# In PPO, after collecting trajectory:
values = critic(states)  # V(s) for all states
next_values = critic(next_states)

# Compute TD errors
deltas = rewards + gamma * next_values * (1 - dones) - values

# Compute GAE (backward pass)
advantages = np.zeros_like(rewards)
gae = 0
for t in reversed(range(len(rewards))):
    gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
    advantages[t] = gae

# Normalize advantages (common practice)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## Common Pitfalls

1. **Wrong direction of computation**: GAE must be computed backwards from end of episode.

2. **Not handling terminal states**: When done=True, don't bootstrap — \(\delta_t = r_t - V(s_t)\).

3. **Using λ=1 always**: Gives MC variance. λ=0.95-0.97 is usually better.

4. **Not normalizing advantages**: Large advantages cause large gradients. Normalize per batch.

5. **Confusing with TD(λ)**: GAE is for advantages; TD(λ) is for value targets. Related but different.

---

## Mini Example

**Short episode**: Rewards = [0, 0, 1], Values = [0.5, 0.6, 0.7], γ=0.99, λ=0.95

**TD errors**:
- δ₂ = 1 + 0 - 0.7 = 0.3 (terminal after step 2)
- δ₁ = 0 + 0.99×0.7 - 0.6 = 0.093
- δ₀ = 0 + 0.99×0.6 - 0.5 = 0.094

**GAE** (backward):
- Â₂ = δ₂ = 0.3
- Â₁ = δ₁ + 0.99×0.95×Â₂ = 0.093 + 0.94×0.3 = 0.375
- Â₀ = δ₀ + 0.99×0.95×Â₁ = 0.094 + 0.94×0.375 = 0.447

**Compare to MC advantage** (λ=1):
- G₀ - V(s₀) = (0 + 0.99×0 + 0.99²×1) - 0.5 = 0.98 - 0.5 = 0.48

GAE gives 0.447, which is close but slightly lower variance.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What is the bias-variance tradeoff in advantage estimation?</summary>

**Answer**:
- **High λ** (→1): Low bias (approaches true advantage), high variance (uses many rewards)
- **Low λ** (→0): High bias (depends on value function accuracy), low variance (uses one TD step)

**Explanation**: TD error uses V(s') which may be wrong — that's bias. MC uses actual returns but sums many random variables — that's variance. GAE interpolates.

**Key insight**: λ controls effective "horizon" of the estimator.

**Common pitfall**: Assuming λ=1 is always best because it's unbiased. Variance matters too!
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why is GAE used in PPO instead of just TD error?</summary>

**Answer**: PPO needs good advantage estimates for policy updates. Pure TD error is too biased (especially early when V is bad). Pure MC is too high variance. GAE provides a tunable balance.

**Explanation**: λ=0.95-0.97 keeps most of the TD variance reduction while getting low enough bias. This gives stable learning.

**Common pitfall**: Using λ=0 in PPO — policy gradients become very noisy.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Show that GAE with λ=1 equals the Monte Carlo advantage.</summary>

**Answer**:

$$\hat{A}^{GAE(\gamma,1)}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}$$

$$= \sum_{l=0}^{\infty} \gamma^l (r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l}))$$

Expanding the telescope:

$$= \sum_{l=0}^{\infty} \gamma^l r_{t+l} + \sum_{l=0}^{\infty} \gamma^{l+1} V(s_{t+l+1}) - \sum_{l=0}^{\infty} \gamma^l V(s_{t+l})$$

The V terms telescope:

$$= \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$

**Key insight**: λ=1 makes GAE sum all TD errors, which recovers full return.

**Common pitfall**: Not seeing the telescoping pattern.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Derive the recursive formula for computing GAE.</summary>

**Answer**:

$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$

**Derivation**:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

$$= \delta_t + \sum_{l=1}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

$$= \delta_t + \gamma\lambda \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+1+l}$$

$$= \delta_t + \gamma\lambda \hat{A}_{t+1}$$

**Key insight**: This allows O(T) computation by backward sweep.

**Common pitfall**: Computing forward (wrong) instead of backward (right).
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> How do you choose λ for a new problem?</summary>

**Answer**: Start with λ=0.95 (good default). Tune based on:
- **Noisy rewards/long episodes**: Lower λ (0.9-0.95) reduces variance
- **Accurate critic**: Can use higher λ (0.97-0.99)
- **Sparse rewards**: Higher λ needed to propagate signal

**Practical approach**:
1. Start with λ=0.95
2. If training is unstable, reduce λ
3. If learning is too slow, increase λ
4. Monitor advantage variance and policy entropy

**Common pitfall**: Not tuning λ — default doesn't work for all problems.
</details>

---

## References

- **Schulman et al. (2016)**, High-Dimensional Continuous Control Using GAE
- **Sutton & Barto**, Reinforcement Learning: An Introduction (TD(λ) chapter)
- **Schulman et al. (2017)**, Proximal Policy Optimization (uses GAE)

**What to memorize for interviews**: GAE formula, recursive computation, λ=0 is TD, λ=1 is MC, typical λ=0.95, bias-variance tradeoff.
