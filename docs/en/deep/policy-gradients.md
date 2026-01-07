# Policy Gradients

## Interview Summary

**Policy gradient methods** directly optimize the policy \(\pi_\theta(a|s)\) by gradient ascent on expected return. The key result is the **policy gradient theorem**: \(\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]\). **REINFORCE** is the simplest algorithm — use sampled returns as Q estimates. High variance is the main challenge; baselines reduce it.

**What to memorize**: Policy gradient theorem, REINFORCE update, log-derivative trick, why baselines reduce variance without adding bias.

---

## Core Definitions

### Policy Parameterization

Stochastic policy: \(\pi_\theta(a|s) = P(A=a|S=s; \theta)\)

Common forms:
- **Discrete**: Softmax over action logits
- **Continuous**: Gaussian \(\mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)\)

### Objective

Maximize expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

### Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a) \right]$$

**Meaning**: The gradient of the objective equals expected "score" weighted by action value.

---

## Math and Derivations

### Deriving the Policy Gradient

Start with:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_\tau P(\tau|\theta) R(\tau)$$

Take gradient:

$$\nabla J = \sum_\tau \nabla P(\tau|\theta) R(\tau)$$

Use log-derivative trick: \(\nabla P = P \nabla \log P\):

$$= \sum_\tau P(\tau|\theta) \nabla \log P(\tau|\theta) R(\tau)$$

$$= \mathbb{E}_\tau[\nabla \log P(\tau|\theta) \cdot R(\tau)]$$

Expand \(\log P(\tau|\theta)\):

$$\log P(\tau|\theta) = \log p(s_0) + \sum_{t=0}^{T} \left[ \log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t) \right]$$

Only \(\pi_\theta\) depends on \(\theta\):

$$\nabla \log P(\tau|\theta) = \sum_{t=0}^{T} \nabla \log \pi_\theta(a_t|s_t)$$

**Final form**:

$$\nabla J = \mathbb{E}_\tau\left[ \sum_{t=0}^{T} \nabla \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

where \(G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k\) is return-to-go (not total return — causality!).

### Baseline Subtraction

$$\nabla J = \mathbb{E}\left[ \nabla \log \pi_\theta(a|s) \cdot (Q^{\pi}(s,a) - b(s)) \right]$$

Any baseline \(b(s)\) that doesn't depend on \(a\) can be subtracted.

**Why unbiased?**

$$\mathbb{E}_a[\nabla \log \pi(a|s) \cdot b(s)] = b(s) \cdot \nabla \sum_a \pi(a|s) = b(s) \cdot \nabla 1 = 0$$

**Common baseline**: \(b(s) = V^\pi(s)\), giving advantage: \(A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)\).

---

## Algorithm Sketch

### REINFORCE

```
Algorithm: REINFORCE (Monte Carlo Policy Gradient)

Input: Learning rate α
Output: Policy parameters θ

1. Initialize θ arbitrarily
2. For each episode:
     Generate trajectory τ = (s_0, a_0, r_1, s_1, ..., s_T) using π_θ
     For t = 0 to T-1:
         G_t = Σ_{k=t}^{T-1} γ^{k-t} r_{k+1}  # return from t
         θ ← θ + α · γ^t · G_t · ∇_θ log π_θ(a_t|s_t)
3. Return θ
```

### REINFORCE with Baseline

```
Algorithm: REINFORCE with Baseline

1. Initialize θ (policy), w (value function)
2. For each episode:
     Generate trajectory using π_θ
     For t = 0 to T-1:
         G_t = Σ_{k=t}^{T-1} γ^{k-t} r_{k+1}
         δ = G_t - V(s_t; w)           # advantage estimate
         w ← w + α_w · δ · ∇_w V(s_t; w)  # value update
         θ ← θ + α_θ · γ^t · δ · ∇_θ log π_θ(a_t|s_t)
3. Return θ
```

---

## Common Pitfalls

1. **High variance**: REINFORCE has high variance from using full returns. Use baselines!

2. **Sample inefficiency**: On-policy — each sample used once. Much less efficient than off-policy.

3. **Reward scaling**: Large rewards → large gradients → instability. Normalize returns.

4. **Entropy collapse**: Policy becomes too deterministic too fast. Add entropy bonus.

5. **Learning rate sensitivity**: Policy gradients are very sensitive to α. Start small (1e-4).

6. **Forgetting to use return-to-go**: Using total episode return instead of \(G_t\) increases variance.

---

## Mini Example

**CartPole with REINFORCE:**

```python
# Policy: neural net → softmax → action probabilities
policy = MLP(4, [32], 2)

for episode in range(1000):
    states, actions, rewards = [], [], []

    # Collect episode
    state = env.reset()
    while not done:
        probs = softmax(policy(state))
        action = sample(probs)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    # Policy update
    for s, a, G in zip(states, actions, returns):
        log_prob = log(softmax(policy(s))[a])
        loss = -log_prob * G  # negative for gradient ascent
        loss.backward()
        optimizer.step()
```

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why optimize the policy directly instead of learning Q and acting greedily?</summary>

**Answer**: Policy gradients have advantages:
1. **Continuous actions**: Can naturally output continuous action distributions
2. **Stochastic policies**: Can represent mixed strategies
3. **Direct optimization**: Optimizes what we care about (expected return)
4. **Better convergence**: Avoids issues with max operator in Q-learning

**Explanation**: Value-based methods require a max over actions, which is problematic for continuous/large action spaces. Policy gradients bypass this.

**Common pitfall**: Assuming policy gradients are always better. Value-based methods are more sample efficient for discrete actions.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why doesn't the baseline add bias to the gradient estimate?</summary>

**Answer**: Because \(\mathbb{E}_a[\nabla \log \pi(a|s) \cdot b(s)] = 0\) for any \(b(s)\) independent of \(a\).

**Explanation**:

$$\mathbb{E}_a[\nabla \log \pi(a|s) \cdot b(s)] = b(s) \sum_a \nabla \pi(a|s) = b(s) \nabla \sum_a \pi(a|s) = b(s) \nabla 1 = 0$$

The baseline factors out and multiplies zero.

**Key insight**: Baselines shift the returns but don't change the expected gradient direction.

**Common pitfall**: Using a baseline that depends on \(a\), which does add bias.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Derive the policy gradient theorem from first principles.</summary>

**Answer**: See Math and Derivations section above. Key steps:
1. \(J(\theta) = \mathbb{E}_\tau[R(\tau)]\)
2. Log-derivative trick: \(\nabla P = P \nabla \log P\)
3. Expand \(\log P(\tau|\theta)\), gradient only through \(\pi_\theta\)
4. Use causality: action at \(t\) only affects rewards from \(t\) onward

**Key equation**:

$$\nabla J = \mathbb{E}\left[\sum_t \nabla \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

**Common pitfall**: Using total return \(R(\tau)\) instead of return-to-go \(G_t\) increases variance.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> What is the advantage function and why is it preferred over Q?</summary>

**Answer**: \(A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)\)

**Why preferred**:
- Centered around zero (negative for bad actions, positive for good)
- Lower variance than raw Q-values
- Focuses gradient on relative action quality

**Explanation**: Using Q directly, even constant high rewards lead to positive gradients for all actions. Advantage only reinforces actions better than average.

**Common pitfall**: Not using a baseline. Raw returns have high variance.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your policy gradient training is very unstable. What to try?</summary>

**Answer**: Stabilization techniques:
1. **Reduce learning rate**: Start at 1e-4 or lower
2. **Add baseline**: Subtract state value to reduce variance
3. **Normalize returns**: \((G - \mu) / \sigma\) within batch
4. **Entropy regularization**: Add \(\beta H(\pi)\) to objective to prevent premature convergence
5. **Gradient clipping**: Clip gradient norm
6. **Larger batch sizes**: Average over more trajectories

**Explanation**: Policy gradients are inherently high-variance. All these techniques reduce effective variance.

**Common pitfall**: Not using enough samples per update. Batch size matters more than in supervised learning.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 13
- **Williams (1992)**, Simple Statistical Gradient-Following Algorithms (REINFORCE)
- **Sutton et al. (2000)**, Policy Gradient Methods for RL with Function Approximation

**What to memorize for interviews**: Policy gradient theorem, log-derivative trick, REINFORCE update, baseline unbiasedness proof, advantage function.

**Code example**: [reinforce.py](../../../rl_examples/algorithms/reinforce.py)
