# Computing Value Functions in Practice

## Interview Summary

Computing value functions seems intractable — they're defined as expectations over infinitely many future trajectories. In practice, RL uses three approaches: **Dynamic Programming** (model-based, iterative Bellman updates), **Monte Carlo** (sample complete episodes, average returns), and **Temporal Difference** (bootstrap using current estimates). For large state spaces, **function approximation** generalizes across states. The key insight: the Bellman equation's recursive structure converts an infinite-horizon problem into tractable one-step updates.

**What to memorize**: Why naive computation is intractable, DP/MC/TD approaches, bootstrapping concept, bias-variance tradeoff between MC and TD.

---

## Design Motivation: The Problem

### The Intractable Definition

Value functions are defined as expectations over all future trajectories:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$$

**The problem**: This expectation is over:
- All possible action sequences (exponentially many)
- All possible state transitions (stochastic environment)
- Infinite time horizon (for continuing tasks)

**Naive approach**: Enumerate all trajectories, weight by probability, sum weighted returns.

**Why this fails**:
- Number of trajectories: $|A|^T \times$ (branching from stochastic transitions)
- For $|A|=4$, $T=100$: $4^{100} \approx 10^{60}$ trajectories
- Computationally impossible!

### The Key Insight: Bellman Equations

The Bellman equation provides the solution:

$$V(s) = \mathbb{E}\left[ r + \gamma V(s') \right]$$

**Why this helps**:
- Converts infinite-horizon expectation into **one-step** expectation
- Future values are summarized by $V(s')$
- We only need to look one step ahead!

This recursive structure is what makes RL tractable.

---

## Core Definitions

### The Three Approaches

| Approach | Requires | Updates | Bias | Variance |
|----------|----------|---------|------|----------|
| **Dynamic Programming** | Model ($P$, $R$) | Full expectation over $s'$ | None | None |
| **Monte Carlo** | Complete episodes | After episode ends | None | High |
| **Temporal Difference** | Single transitions | Every step | Yes (if $V$ wrong) | Low |

### What "Bootstrapping" Means

**Bootstrapping** = using your own estimates to update your estimates.

- **Not bootstrapping** (MC): Use actual observed returns $G_t$
- **Bootstrapping** (TD, DP): Use estimated future value $V(s')$

```
MC:  V(s) ← average of actual returns G
TD:  V(s) ← V(s) + α[r + γV(s') - V(s)]
                            ↑
                    This is the bootstrap!
```

---

## The Three Approaches Explained

### Approach 1: Dynamic Programming (Model-Based)

**Requirement**: Full MDP model — transition probabilities $P(s'|s,a)$ and rewards $R(s,a)$.

**Method**: Iteratively apply Bellman equation:

$$V_{k+1}(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$

**How it works**:

```
Initialize V(s) = 0 for all states

Repeat until convergence:
    For each state s:
        V_new(s) = Σ_a π(a|s) × [R(s,a) + γ × Σ_s' P(s'|s,a) × V(s')]
    V = V_new

Return V
```

**Why it converges**: The Bellman operator is a contraction mapping. Each iteration brings $V$ closer to the true value. Convergence guaranteed for $\gamma < 1$.

**Complexity**: $O(|S|^2 |A|)$ per iteration — feasible only for small state spaces.

**Example**:

```
Two-state MDP with policy π(stay) = 1:
- State A: R(A) = 2, stays in A
- State B: R(B) = 1, stays in B
- γ = 0.9

Iteration 0: V(A) = 0, V(B) = 0
Iteration 1: V(A) = 2 + 0.9×0 = 2
             V(B) = 1 + 0.9×0 = 1
Iteration 2: V(A) = 2 + 0.9×2 = 3.8
             V(B) = 1 + 0.9×1 = 1.9
...
Converges to: V(A) = 2/(1-0.9) = 20
              V(B) = 1/(1-0.9) = 10
```

---

### Approach 2: Monte Carlo (Model-Free, Sample-Based)

**Requirement**: Ability to run episodes (no model needed).

**Method**: Replace expectation with sample average:

$$V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i(s)$$

where $G_i(s)$ is the return observed starting from state $s$ in episode $i$.

**How it works**:

```
Initialize V(s) = 0, counts(s) = 0 for all states

For each episode:
    Generate trajectory: s₀, a₀, r₁, s₁, a₁, r₂, ..., s_T

    For each state s_t in trajectory:
        G_t = r_{t+1} + γr_{t+2} + γ²r_{t+3} + ... + γ^(T-t)r_T
        counts(s_t) += 1
        V(s_t) += (G_t - V(s_t)) / counts(s_t)   # Running average

Return V
```

**Why it works**: Law of Large Numbers — sample average converges to true expectation.

**Pros**:
- No model needed
- No bias (uses actual returns)
- Works for non-Markovian problems

**Cons**:
- High variance (each return is sum of many random variables)
- Must wait for episode to end
- Only works for episodic tasks

**Example**:

```
GridWorld: Start at A, goal at G (reward +10), γ = 0.9

Episode 1: A → B → C → G  (rewards: 0, 0, 10)
           G(A) = 0 + 0.9×0 + 0.81×10 = 8.1

Episode 2: A → B → G      (rewards: 0, 10)
           G(A) = 0 + 0.9×10 = 9.0

Episode 3: A → C → B → G  (rewards: 0, 0, 10)
           G(A) = 0 + 0.9×0 + 0.81×10 = 8.1

V(A) ≈ (8.1 + 9.0 + 8.1) / 3 = 8.4
```

---

### Approach 3: Temporal Difference (Model-Free, Bootstrapping)

**Requirement**: Single transitions (no complete episodes needed).

**Method**: Update toward one-step bootstrap target:

$$V(s) \leftarrow V(s) + \alpha \left[ r + \gamma V(s') - V(s) \right]$$

**The key idea**: Use current estimate $V(s')$ as a stand-in for all future returns.

**How it works**:

```
Initialize V(s) arbitrarily

For each step in environment:
    Observe current state s
    Take action a, observe reward r, next state s'

    # TD update
    TD_target = r + γ × V(s')
    TD_error = TD_target - V(s)
    V(s) = V(s) + α × TD_error

    s = s'
```

**Why it works** (intuition):
- If $V(s')$ were correct, then $r + \gamma V(s')$ is an unbiased sample of $V(s)$
- Errors in $V(s')$ get corrected as we update all states together
- The whole system becomes self-consistent over time

**Pros**:
- Updates every step (no waiting for episode end)
- Lower variance than MC
- Works for continuing (non-episodic) tasks

**Cons**:
- Biased if $V$ estimates are wrong (converges as $V$ improves)
- Needs more careful learning rate tuning

**Example**:

```
Same GridWorld, α = 0.1, γ = 0.9

Initial: V(A) = 0, V(B) = 0, V(C) = 0, V(G) = 0

Step 1: A → B (r = 0)
        TD_target = 0 + 0.9 × V(B) = 0 + 0.9 × 0 = 0
        V(A) = 0 + 0.1 × (0 - 0) = 0

Step 2: B → G (r = 10)
        TD_target = 10 + 0.9 × V(G) = 10 + 0.9 × 0 = 10
        V(B) = 0 + 0.1 × (10 - 0) = 1.0

Step 3: A → B (r = 0)  [new episode]
        TD_target = 0 + 0.9 × V(B) = 0 + 0.9 × 1.0 = 0.9
        V(A) = 0 + 0.1 × (0.9 - 0) = 0.09

# Value propagates backward through states!
```

---

### Approach 4: Function Approximation (Scaling Up)

**Problem**: Can't store $V(s)$ for every state when $|S|$ is huge or continuous.

**Solution**: Learn a parameterized function $V_\theta(s)$ that generalizes:

$$V_\theta(s) \approx V^\pi(s)$$

**Method**: Gradient descent on TD error:

$$\theta \leftarrow \theta + \alpha \left[ r + \gamma V_\theta(s') - V_\theta(s) \right] \nabla_\theta V_\theta(s)$$

**How it works**:

```
Initialize neural network V_θ

For each transition (s, a, r, s'):
    # Compute TD target (no gradient through this!)
    target = r + γ × V_θ(s').detach()

    # Compute loss
    loss = (target - V_θ(s))²

    # Gradient step
    θ = θ - α × ∇_θ loss
```

**Why it works**:
- Neural network learns patterns: "similar states have similar values"
- Generalizes to unseen states
- Enables learning in high-dimensional spaces (images, continuous states)

**Challenges**:
- No convergence guarantees (deadly triad)
- Requires careful stabilization (target networks, etc.)

---

## Comparison: MC vs TD

| Aspect | Monte Carlo | TD Learning |
|--------|-------------|-------------|
| **Updates** | After episode | Every step |
| **Bias** | Unbiased | Biased (initially) |
| **Variance** | High | Low |
| **Bootstrap** | No | Yes |
| **Continuing tasks** | No | Yes |
| **Initial estimates matter** | No | Yes |

### The Bias-Variance Tradeoff

**Monte Carlo**:
- Uses actual return $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...$
- Sum of many random variables → high variance
- But it's the true return → no bias

**TD Learning**:
- Uses $r_t + \gamma V(s_{t+1})$ — only one random reward
- Much lower variance
- But if $V(s_{t+1})$ is wrong → biased
- Bias decreases as $V$ improves

**In practice**: TD usually wins because variance reduction is more valuable than eliminating small bias.

---

## The Deep Insight

The Bellman equation is what makes RL tractable:

$$V(s) = \mathbb{E}\left[ r + \gamma V(s') \right]$$

**This recursive structure means**:
- We don't need to simulate infinitely into the future
- We only need to look **one step ahead**
- Future value is summarized by our current estimate $V(s')$
- Iterative updates gradually propagate correct values

**Without the Bellman equation**, RL would require complete trajectory enumeration — computationally impossible for any real problem.

---

## Algorithm Sketch

### Unified View of Value Computation

```
Target for V(s):
├── DP:    Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]   (full expectation)
├── MC:    G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...          (sampled return)
└── TD(0): r_t + γV(s_{t+1})                               (bootstrap)

Update rule:
V(s) ← V(s) + α × [target - V(s)]
```

### TD(n): Bridging MC and TD

We can use n-step returns to interpolate:

$$G_t^{(n)} = r_t + \gamma r_{t+1} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$$

| n | Method | Bias | Variance |
|---|--------|------|----------|
| 1 | TD(0) | High | Low |
| 2-5 | TD(n) | Medium | Medium |
| ∞ | MC | None | High |

---

## Common Pitfalls

1. **Thinking we enumerate trajectories**: We never compute full expectations explicitly — we use samples or iterative updates.

2. **Forgetting bootstrapping introduces bias**: TD estimates are biased when value estimates are wrong. This is usually acceptable because variance reduction is more important.

3. **Using MC for continuing tasks**: Monte Carlo requires episodes to end. For continuing tasks, use TD.

4. **Ignoring the model requirement for DP**: Dynamic programming requires knowing $P(s'|s,a)$ and $R(s,a)$. Without a model, use MC or TD.

5. **Not understanding why TD works**: TD "pulls itself up by its bootstraps" — estimates improve each other. This seems circular but converges because of the contraction property.

6. **Expecting fast convergence**: Value propagation takes many iterations. In a chain of 100 states, information must propagate through all of them.

---

## Mini Example: Comparing All Three Methods

**Setup**: Linear chain MDP with 5 states, goal at state 5.

```
[1] → [2] → [3] → [4] → [5=Goal]
         reward = 1 at goal, 0 elsewhere
         γ = 0.9
```

True values: $V(1) = 0.9^4 = 0.66$, $V(2) = 0.9^3 = 0.73$, $V(3) = 0.9^2 = 0.81$, $V(4) = 0.9$

### Dynamic Programming

```
Iteration 0: V = [0, 0, 0, 0, 0]
Iteration 1: V = [0, 0, 0, 0.9, 1.0]  # Goal value = 1, state 4 gets 0.9×1
Iteration 2: V = [0, 0, 0.81, 0.9, 1.0]  # Value propagates back
Iteration 3: V = [0, 0.73, 0.81, 0.9, 1.0]
Iteration 4: V = [0.66, 0.73, 0.81, 0.9, 1.0]  # Converged!
```

**4 iterations to propagate through 4 states.**

### Monte Carlo

```
Episode 1: 1→2→3→4→5 (r=1)
  G(1) = 0.9⁴×1 = 0.66
  G(2) = 0.9³×1 = 0.73
  ...

Episode 2: 1→2→3→4→5 (r=1)
  Same returns (deterministic environment)

V(1) ≈ 0.66 after just 1 episode!
```

**Immediate convergence for deterministic environment** (but high variance if stochastic).

### TD Learning

```
α = 0.1, initial V = [0, 0, 0, 0, 0]

Step 1→2: V(1) += 0.1×(0 + 0.9×0 - 0) = 0
Step 2→3: V(2) += 0.1×(0 + 0.9×0 - 0) = 0
Step 3→4: V(3) += 0.1×(0 + 0.9×0 - 0) = 0
Step 4→5: V(4) += 0.1×(1 + 0.9×0 - 0) = 0.1  # First learning!
Step 5=goal: episode ends

# Next episode
Step 1→2: V(1) += 0.1×(0 + 0.9×0 - 0) = 0
...
Step 3→4: V(3) += 0.1×(0 + 0.9×0.1 - 0) = 0.009  # Value propagates!
Step 4→5: V(4) += 0.1×(1 + 0 - 0.1) = 0.19

# Values slowly propagate backward through many episodes
```

**Slower than MC for this simple case, but lower variance for stochastic environments.**

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why can't we compute V(s) by enumerating all possible trajectories?</summary>

**Answer**: The number of trajectories is exponentially large and often infinite.

**Explanation**: For $|A|$ actions over $T$ timesteps, there are $O(|A|^T)$ possible action sequences. With stochastic transitions, each action leads to multiple possible next states, further multiplying the paths. For continuing tasks ($T = \infty$), there are infinitely many trajectories.

**Example**: With 4 actions and 100 timesteps: $4^{100} \approx 10^{60}$ trajectories — more than atoms in the universe!

**Key insight**: The Bellman equation avoids this by using the recursive structure: $V(s) = E[r + \gamma V(s')]$.

**Common pitfall**: Thinking RL actually computes expectations explicitly. It uses samples (MC, TD) or iterative updates (DP).
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> What does "bootstrapping" mean and which methods use it?</summary>

**Answer**: Bootstrapping means using your own estimates to update your estimates.

**Explanation**:
- **TD and DP bootstrap**: Update $V(s)$ using $V(s')$, which is itself an estimate
- **MC doesn't bootstrap**: Uses actual sampled returns $G_t$

**Formula comparison**:
- TD: $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$ — uses $V(s')$
- MC: $V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$ — uses actual return

**Trade-off**: Bootstrapping introduces bias (if estimates are wrong) but reduces variance (fewer random variables).

**Common pitfall**: Thinking bootstrapping is circular and shouldn't work. It converges because the Bellman operator is a contraction.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Explain the bias-variance tradeoff between MC and TD.</summary>

**Answer**: MC has no bias but high variance; TD has low variance but initial bias.

**Explanation**:

**Monte Carlo**:
- Target: $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...$
- This is the true return → unbiased estimate of $V(s)$
- Sum of many random variables → high variance

**TD Learning**:
- Target: $r_t + \gamma V(s_{t+1})$
- Only one random reward → low variance
- If $V(s_{t+1})$ is wrong → biased
- Bias decreases as $V$ improves through learning

**Key equation**: $\text{Var}(G_t) = \text{Var}(\sum_k \gamma^k r_k) \gg \text{Var}(r_t + \gamma V(s'))$

**Common pitfall**: Thinking bias is always bad. In practice, TD's variance reduction usually outweighs its bias.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Why does value iteration converge? What is the contraction property?</summary>

**Answer**: The Bellman operator is a contraction mapping — it brings value estimates closer together with each iteration.

**Explanation**: For any two value functions $V_1, V_2$:

$$\|T V_1 - T V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$$

where $T$ is the Bellman operator. Since $\gamma < 1$, the distance between any two value functions shrinks by factor $\gamma$ each iteration.

**Consequence**:
- After $k$ iterations, error is at most $\gamma^k \times$ initial error
- For $\gamma = 0.99$, after 100 iterations: $0.99^{100} \approx 0.37$
- For $\gamma = 0.9$, after 100 iterations: $0.9^{100} \approx 10^{-5}$

**Common pitfall**: Using $\gamma = 1$ for continuing tasks. No contraction → no convergence guarantee!
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your TD learning is very slow to converge. What could help?</summary>

**Answer**: Several techniques can speed up convergence:

1. **Increase learning rate $\alpha$**: Faster updates, but too high causes instability
2. **Use eligibility traces (TD(λ))**: Propagate credit to multiple past states at once
3. **Better initialization**: Start with values close to true values if possible
4. **Prioritized updates**: Update states with largest TD errors first (prioritized sweeping)
5. **Use n-step returns**: Balance between TD(0) and MC for better bias-variance tradeoff

**Explanation**: TD propagates values one step at a time. In a long chain, information takes many episodes to propagate from goal to start. Eligibility traces and n-step returns let information "skip" multiple states.

**Common pitfall**: Setting $\alpha$ too high. This causes oscillation and divergence. Start with $\alpha = 0.1$ or smaller.
</details>

<details markdown="1">
<summary><strong>Q6 (Conceptual):</strong> When should you use DP vs MC vs TD?</summary>

**Answer**: It depends on what you have access to:

| Situation | Best Method |
|-----------|-------------|
| Have model ($P$, $R$) | Dynamic Programming |
| No model, episodic tasks | Monte Carlo or TD |
| No model, continuing tasks | TD (MC requires episodes) |
| Need low variance | TD (with eligibility traces) |
| Need unbiased estimates | Monte Carlo |
| Online learning required | TD |

**Explanation**:
- **DP** is most efficient when you have the model, but models are often unavailable
- **MC** is unbiased but high variance and requires complete episodes
- **TD** is most flexible — works online, works for continuing tasks, lower variance

**Common pitfall**: Using MC for continuing tasks (infinite episodes) or trying DP without a model.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapters 4-6
- **Szepesvári (2010)**, Algorithms for Reinforcement Learning
- **Silver's RL Course**, Lectures 3-4: Dynamic Programming and Model-Free Prediction

**What to memorize for interviews**: The three approaches (DP/MC/TD), bootstrapping definition, bias-variance tradeoff, why Bellman equation makes RL tractable, contraction property for convergence.
