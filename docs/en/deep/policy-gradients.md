# Policy Gradients

## Interview Summary

**Policy gradient methods** directly optimize the policy $\pi_\theta(a|s)$ by gradient ascent on expected return. The key result is the **policy gradient theorem**: $\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$. **REINFORCE** is the simplest algorithm — use sampled returns as Q estimates. High variance is the main challenge; baselines reduce it.

**What to memorize**: Policy gradient theorem, REINFORCE update, log-derivative trick, why baselines reduce variance without adding bias.

---

## Design Motivation: Why Policy Gradients Exist

### The Limitation of Value-Based Methods

Q-learning and DQN learn a value function, then act greedily:

$$\pi(s) = \arg\max_a Q(s, a)$$

**Problem 1: Continuous actions**
- What if action space is continuous (e.g., robot joint torques)?
- $\arg\max$ over infinite actions is intractable!

**Problem 2: Stochastic policies**
- Sometimes optimal behavior is random (e.g., rock-paper-scissors)
- Value-based methods give deterministic policies

**Problem 3: Indirect optimization**
- We care about cumulative reward, but we're learning Q
- No guarantee that better Q → better policy

### The Policy Gradient Idea

> **Key Insight**: Why not optimize the policy directly?

Instead of:
```
Learn Q(s,a) → Derive policy from Q
```

Do:
```
Parameterize policy π(a|s; θ) → Optimize θ directly for reward
```

**Advantages:**
- Works naturally with continuous actions
- Can represent stochastic policies
- Directly optimizes what we care about

### The Challenge: How to Compute the Gradient?

The objective is:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

**Problem**: The expectation is over trajectories sampled from $\pi_\theta$. How do we differentiate through this sampling process?

**Solution**: The **policy gradient theorem** gives us a way!

---

## Core Definitions

### Policy Parameterization

Stochastic policy: $\pi_\theta(a|s) = P(A=a|S=s; \theta)$

**Common forms:**

| Action Type | Policy Form | Output |
|-------------|-------------|--------|
| Discrete | Softmax | $\pi(a|s) = \frac{e^{f_\theta(s,a)}}{\sum_{a'} e^{f_\theta(s,a')}}$ |
| Continuous | Gaussian | $\pi(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$ |

### Objective: Expected Return

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

**Goal**: Find $\theta$ that maximizes $J(\theta)$.

### The Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a) \right]$$

**In words**: The gradient of expected return equals:
- The "score function" $\nabla \log \pi$ (which direction increases action probability)
- Weighted by the action's value $Q(s,a)$ (how good is this action)

**Intuition**: Increase probability of good actions, decrease probability of bad actions.

---

## Math and Derivations

### Deriving the Policy Gradient (Important!)

This derivation is commonly asked in interviews.

**Step 1**: Write objective as expectation

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_\tau P(\tau|\theta) R(\tau)$$

**Step 2**: Take gradient

$$\nabla_\theta J = \sum_\tau \nabla_\theta P(\tau|\theta) \cdot R(\tau)$$

**Step 3**: Log-derivative trick

$$\nabla P = P \cdot \nabla \log P$$

Therefore:

$$\nabla_\theta J = \sum_\tau P(\tau|\theta) \cdot \nabla_\theta \log P(\tau|\theta) \cdot R(\tau)$$

$$= \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau|\theta) \cdot R(\tau)]$$

**Step 4**: Expand log probability of trajectory

$$\log P(\tau|\theta) = \log p(s_0) + \sum_{t=0}^{T-1} \left[ \log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t) \right]$$

Only $\pi_\theta$ depends on $\theta$:

$$\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**Step 5**: Final form (with causality)

$$\nabla_\theta J = \mathbb{E}_\tau\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

where $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$ is **return-to-go** (not total return!).

**Why return-to-go?** Action at time $t$ only affects rewards from $t$ onward (causality).

### The Log-Derivative Trick Explained

$$\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \cdot \nabla_\theta \log P(\tau|\theta)$$

**Why this works:**

$$\nabla \log f = \frac{\nabla f}{f}$$

$$\therefore \nabla f = f \cdot \nabla \log f$$

**Why it's useful**: Converts gradient of probability into gradient of log-probability, which we can compute for $\pi_\theta$!

### Baseline Subtraction (Variance Reduction)

$$\nabla_\theta J = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot (Q^{\pi}(s,a) - b(s)) \right]$$

Any baseline $b(s)$ that doesn't depend on $a$ can be subtracted **without changing the expected gradient**.

**Proof that baseline doesn't add bias:**

$$\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$$

$$= b(s) \sum_a \nabla_\theta \pi_\theta(a|s) = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0$$

**Best baseline**: $b(s) = V^\pi(s)$, giving the **advantage**:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

**Intuition**: Advantage tells us "how much better is action $a$ compared to the average action in state $s$?"

---

## Algorithm Sketch

### REINFORCE (Monte Carlo Policy Gradient)

```
Algorithm: REINFORCE

Input: Learning rate α, discount γ
Initialize: Policy parameters θ

For each episode:
    # Collect trajectory
    τ = []
    s = env.reset()
    while not done:
        a ~ π_θ(·|s)                    # Sample action
        s', r, done = env.step(a)
        τ.append((s, a, r))
        s = s'

    # Compute returns-to-go
    G = 0
    returns = []
    for (s, a, r) in reversed(τ):
        G = r + γ * G
        returns.insert(0, G)

    # Policy gradient update
    for (s, a, r), G in zip(τ, returns):
        θ ← θ + α · G · ∇_θ log π_θ(a|s)

Return θ
```

### REINFORCE with Baseline

```
Algorithm: REINFORCE with Baseline

Initialize: θ (policy), φ (value function)

For each episode:
    Collect trajectory τ using π_θ
    Compute returns G_t

    For each timestep t:
        # Advantage = return - baseline
        A_t = G_t - V_φ(s_t)

        # Update value function (baseline)
        φ ← φ + α_φ · A_t · ∇_φ V_φ(s_t)

        # Update policy
        θ ← θ + α_θ · A_t · ∇_θ log π_θ(a_t|s_t)

Return θ
```

### Key Design Choices

| Choice | Purpose |
|--------|---------|
| Return-to-go $G_t$ | Causality — only future rewards matter |
| Baseline $V(s)$ | Variance reduction without bias |
| Normalize returns | Stability — $(G - \mu) / \sigma$ |
| Entropy bonus | Prevent premature convergence |

---

## Value-Based vs Policy-Based: When to Use Which?

| Aspect | Value-Based (DQN) | Policy-Based (PG) |
|--------|-------------------|-------------------|
| Action space | Discrete only | Any (continuous!) |
| Policy type | Deterministic | Stochastic |
| Sample efficiency | High (replay) | Low (on-policy) |
| Stability | Can diverge | More stable |
| Convergence | To local optimum of Q | To local optimum of J |
| Best for | Discrete, complex values | Continuous, simple values |

**Rule of thumb:**
- Discrete actions, need sample efficiency → DQN
- Continuous actions, or need stochastic policy → Policy Gradient

---

## Common Pitfalls

1. **High variance**: REINFORCE without baseline has very high variance. Always use a baseline!

2. **Using total return instead of return-to-go**: Including past rewards in gradient is pointless variance.

3. **Sample inefficiency**: Each sample used once. This is fundamental to on-policy methods.

4. **Learning rate too high**: Policy gradients are sensitive. Start at 1e-4 or lower.

5. **Entropy collapse**: Policy becomes deterministic too fast. Add entropy bonus: $L = L_{PG} + \beta H(\pi)$.

6. **Reward scaling**: Large rewards → large gradients → instability. Normalize returns!

7. **Forgetting to sample actions**: Must sample from $\pi_\theta$, not take argmax!

---

## Mini Example

**CartPole with REINFORCE:**

```python
# Policy network: state → action probabilities
policy = MLP(input_dim=4, hidden=[32], output_dim=2)
optimizer = Adam(policy.parameters(), lr=1e-3)

for episode in range(1000):
    states, actions, rewards = [], [], []

    # Collect episode
    state = env.reset()
    done = False
    while not done:
        # Sample action from policy
        probs = softmax(policy(state))
        action = sample_categorical(probs)

        next_state, reward, done = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # Compute returns-to-go
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    # Normalize returns (variance reduction)
    returns = (returns - mean(returns)) / (std(returns) + 1e-8)

    # Policy gradient update
    loss = 0
    for s, a, G in zip(states, actions, returns):
        log_prob = log(softmax(policy(s))[a])
        loss -= log_prob * G  # Negative because optimizer minimizes

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why use policy gradients instead of value-based methods?</summary>

**Answer**: Policy gradients have key advantages:
1. **Continuous actions**: No need to solve $\arg\max_a Q(s,a)$
2. **Stochastic policies**: Can represent mixed strategies
3. **Direct optimization**: Optimizes expected return directly
4. **Simplicity**: No target networks, no replay buffer

**When to prefer value-based**: Discrete actions with need for sample efficiency (DQN with replay).

**Key insight**: Policy gradient is the only option for continuous control without discretization.

**Common pitfall**: Using policy gradient for simple discrete problems where DQN would be more sample-efficient.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Explain the log-derivative trick and why it's essential.</summary>

**Answer**: The log-derivative trick converts:

$$\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] \text{ (hard)} \rightarrow \mathbb{E}_{x \sim p_\theta}[\nabla_\theta \log p_\theta(x) \cdot f(x)] \text{ (easy)}$$

**Why essential:**
- We can't differentiate through sampling
- But we CAN compute $\nabla \log \pi_\theta(a|s)$ for any sampled action
- Allows Monte Carlo estimation of gradient

**The trick:**
$$\nabla p = p \cdot \nabla \log p$$

**Common pitfall**: Forgetting this is the core of why policy gradient works.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Prove that subtracting a state-dependent baseline doesn't add bias.</summary>

**Answer**: We need to show $\mathbb{E}_a[\nabla \log \pi(a|s) \cdot b(s)] = 0$:

$$\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)]$$

$$= b(s) \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$$

$$= b(s) \sum_a \pi_\theta(a|s) \cdot \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

$$= b(s) \sum_a \nabla_\theta \pi_\theta(a|s)$$

$$= b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s)$$

$$= b(s) \cdot \nabla_\theta 1 = 0$$

**Key insight**: The baseline factors out and multiplies zero because probabilities sum to 1.

**Common pitfall**: Using a baseline that depends on $a$ — this DOES add bias!
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> What is the advantage function and why is it better than Q?</summary>

**Answer**: $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$

**Why better than Q:**

| Using Q | Using Advantage |
|---------|-----------------|
| All Q-values might be large positive | Centered around zero |
| Good actions: positive, bad actions: also positive! | Good: positive, bad: negative |
| High variance | Lower variance |

**Intuition**: "How much better is this action than average?"
- $A > 0$: Better than average → increase probability
- $A < 0$: Worse than average → decrease probability
- $A = 0$: Average → no change

**Common pitfall**: Not using a baseline. Raw Q values work but have much higher variance.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your policy gradient training has very high variance. What to do?</summary>

**Answer**: Variance reduction techniques:

1. **Add baseline**: Use $V(s)$ to compute advantage. Most important!
2. **Normalize returns**: $(G - \mu) / \sigma$ per batch
3. **Larger batch size**: Average over more trajectories
4. **Use return-to-go**: Not total episode return
5. **GAE**: Generalized Advantage Estimation (later topic)
6. **Entropy regularization**: Prevents policy collapse
7. **Lower learning rate**: More stable updates

**Priority order**: Baseline > normalization > batch size > others.

**Common pitfall**: Trying to reduce variance by reducing learning rate alone. Address the source!
</details>

<details markdown="1">
<summary><strong>Q6 (Conceptual):</strong> Why is REINFORCE on-policy? What does that mean for sample efficiency?</summary>

**Answer**: **On-policy** means we can only learn from data collected by the current policy $\pi_\theta$.

**Why on-policy:**
- Policy gradient is $\mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta \cdot G]$
- Expectation is under $\pi_\theta$
- Data from old $\pi_{\theta_{old}}$ doesn't give valid gradient for current $\pi_\theta$

**Sample efficiency implications:**
- Each transition used for ONE gradient update, then discarded
- Cannot use experience replay (data would be off-policy)
- Much less sample-efficient than DQN

**Solutions**: Actor-critic (reduce variance), PPO/TRPO (reuse samples with correction).

**Common pitfall**: Trying to add replay buffer to REINFORCE — this breaks the algorithm!
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 13
- **Williams (1992)**, Simple Statistical Gradient-Following Algorithms (REINFORCE)
- **Sutton et al. (2000)**, Policy Gradient Methods for RL with Function Approximation

**What to memorize for interviews**: Policy gradient theorem, log-derivative trick derivation, baseline unbiasedness proof, advantage function, on-policy vs off-policy.

**Code example**: [reinforce.py](../../../rl_examples/algorithms/reinforce.py)
