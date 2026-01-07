# Exploration Strategies

## Interview Summary

**Exploration** is the process of taking actions to discover new information about the environment. Key tension: **exploitation** (use known good actions) vs **exploration** (try unknown actions). Common strategies: **ε-greedy** (random with probability ε), **softmax/Boltzmann** (probabilistic based on Q-values), **UCB** (optimism in the face of uncertainty), **entropy bonus** (reward policy randomness). Deep exploration (coordinated across timesteps) is an active research area.

**What to memorize**: ε-greedy, UCB formula, entropy bonus, exploration-exploitation tradeoff.

---

## Core Definitions

### The Exploration-Exploitation Dilemma

- **Exploitation**: Choose actions known to be good → short-term optimal
- **Exploration**: Try less-known actions → may find better long-term strategies

Both are necessary. Pure exploitation can get stuck in local optima. Pure exploration doesn't capitalize on knowledge.

### ε-Greedy

$$a = \begin{cases} \arg\max_a Q(s,a) & \text{with probability } 1-\epsilon \\ \text{random action} & \text{with probability } \epsilon \end{cases}$$

**Advantages**: Simple, no additional computation
**Disadvantages**: Explores randomly (doesn't prioritize promising actions), all non-greedy actions equally likely

### Softmax (Boltzmann) Exploration

$$P(a|s) = \frac{\exp(Q(s,a) / \tau)}{\sum_{a'} \exp(Q(s,a') / \tau)}$$

where \(\tau\) is the temperature:
- \(\tau \to 0\): Greedy (deterministic)
- \(\tau \to \infty\): Uniform random
- \(\tau = 1\): Standard softmax

**Advantage**: Higher-value actions explored more often

### Upper Confidence Bound (UCB)

$$a = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln t}{N(s,a)}} \right]$$

**Intuition**: Add "exploration bonus" that's large for rarely-tried actions.
- \(t\): Total time steps
- \(N(s,a)\): Number of times action \(a\) taken in state \(s\)
- \(c\): Exploration coefficient

**Principle**: "Optimism in the face of uncertainty"

---

## Math and Derivations

### UCB Regret Bound

For the multi-armed bandit setting, UCB achieves:

$$\text{Regret}(T) = O(\sqrt{KT \ln T})$$

where \(K\) is number of arms. This is near-optimal.

### Entropy Bonus in Policy Gradient

Add entropy to the objective:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[R] + \beta H(\pi_\theta)$$

where:

$$H(\pi_\theta(s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

**Effect**: Penalizes deterministic policies, encourages exploration.

**Gradient**:

$$\nabla_\theta H = -\sum_a (\log \pi_\theta(a|s) + 1) \nabla_\theta \pi_\theta(a|s)$$

### Intrinsic Motivation

Add intrinsic reward for visiting novel states:

$$r^{total} = r^{extrinsic} + \beta \cdot r^{intrinsic}$$

Common intrinsic rewards:
- **Curiosity** (ICM): Prediction error of next state
- **Count-based**: \(1/\sqrt{N(s)}\) for visit count
- **RND**: Random network prediction error

---

## Algorithm Sketch

### ε-Greedy with Decay

```
Algorithm: ε-Greedy with Linear Decay

Initial: ε_start = 1.0, ε_end = 0.01, decay_steps = 100000

For each step t:
    ε = max(ε_end, ε_start - (ε_start - ε_end) * t / decay_steps)
    With probability ε:
        a = random action
    Else:
        a = argmax_a Q(s, a)
```

### UCB for Bandits

```
Algorithm: UCB1

Initialize: N(a) = 0, Q(a) = 0 for all actions

For t = 1 to T:
    If any action has N(a) = 0:
        Choose that action
    Else:
        a = argmax_a [Q(a) + c * sqrt(ln(t) / N(a))]

    Observe reward r
    N(a) ← N(a) + 1
    Q(a) ← Q(a) + (r - Q(a)) / N(a)
```

---

## Common Pitfalls

1. **ε never decays**: Agent keeps making random mistakes. Decay ε over training.

2. **ε decays too fast**: Not enough exploration early on. Start with ε = 1.0, decay slowly.

3. **Ignoring state-dependent exploration**: ε-greedy explores uniformly; some states need more exploration.

4. **Entropy coefficient too high/low**: Too high → random policy. Too low → no exploration.

5. **UCB with function approximation**: Count-based UCB is hard with large state spaces. Need pseudo-counts or neural approaches.

---

## Mini Example

**10-Armed Bandit:**

True means: [0.1, 0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.8, 0.9, 0.35]
Optimal arm: 9 (mean = 0.9)

After 1000 steps:
- **ε-greedy (ε=0.1)**: Regret ≈ 50, samples all arms but wastes time on bad ones
- **UCB**: Regret ≈ 30, quickly focuses on good arms while verifying
- **Pure greedy**: Regret varies wildly, might lock onto suboptimal arm

**Key insight**: UCB explores efficiently by prioritizing uncertain arms.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What is the exploration-exploitation tradeoff?</summary>

**Answer**: The tension between:
- **Exploitation**: Using current knowledge to maximize immediate reward
- **Exploration**: Taking uncertain actions to gain information

**Explanation**: Early in learning, exploration is crucial — you don't know what's best. Later, exploitation becomes more valuable — you should use what you've learned. The optimal balance depends on the horizon (how much time left).

**Key insight**: In finite-horizon settings, exploration should decrease over time.

**Common pitfall**: Always using fixed ε — doesn't adapt to learning progress.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why is softmax (Boltzmann) better than ε-greedy for some problems?</summary>

**Answer**: Softmax explores promising actions more often, while ε-greedy treats all non-greedy actions equally.

**Explanation**: With ε-greedy, if Q-values are [10, 9, 1, 1], the three non-greedy actions are equally likely. Softmax makes the 9-action much more likely than the 1-actions.

**Trade-off**: Softmax requires computing exponentials and is sensitive to Q-value scale.

**Common pitfall**: Using wrong temperature τ. Too low → like greedy. Too high → like random.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Derive the UCB formula intuition from confidence intervals.</summary>

**Answer**: UCB adds an upper confidence bound to the estimated value.

**Derivation intuition**:
- \(Q(a)\) is the sample mean of rewards for action \(a\)
- By Hoeffding's inequality, true mean is within \(\sqrt{\frac{\ln(1/\delta)}{2N(a)}}\) with probability \(1-\delta\)
- Setting \(\delta = 1/t\) gives the \(\sqrt{\frac{\ln t}{N(a)}}\) bonus
- We act as if true value is at the upper bound (optimistic)

**Key equation**: \(a = \arg\max [Q(a) + c\sqrt{\frac{\ln t}{N(a)}}]\)

**Common pitfall**: Using UCB in RL without adapting for bootstrapping and function approximation.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> How does entropy regularization encourage exploration?</summary>

**Answer**: Adding \(\beta H(\pi)\) to the objective rewards high-entropy (more random) policies.

$$H(\pi(s)) = -\sum_a \pi(a|s) \log \pi(a|s)$$

**Effect**:
- Deterministic policy: H = 0
- Uniform policy: H = log|A| (maximum)

By maximizing H alongside reward, we prevent premature convergence to deterministic policies.

**Common pitfall**: β too large makes policy ignore rewards; β too small has no effect.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your RL agent learns slowly and seems to miss good actions. How to improve exploration?</summary>

**Answer**: Strategies to try:
1. **Increase initial ε**: Start at 1.0, decay to 0.01 over training
2. **Add entropy bonus**: Use β ≈ 0.01 in policy gradient objective
3. **Try UCB-style bonuses**: Add optimism to Q-values
4. **Use curiosity/intrinsic motivation**: ICM, RND for sparse reward environments
5. **Diverse experience collection**: Multiple parallel environments with different seeds
6. **Parameter noise**: Add noise to network weights instead of actions

**Explanation**: The right approach depends on the environment. Dense rewards → ε-greedy often sufficient. Sparse rewards → need intrinsic motivation.

**Common pitfall**: Assuming one exploration strategy works for all problems.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 2
- **Auer et al. (2002)**, Finite-time Analysis of the Multiarmed Bandit Problem (UCB)
- **Pathak et al. (2017)**, Curiosity-driven Exploration (ICM)
- **Burda et al. (2019)**, Exploration by Random Network Distillation

**What to memorize for interviews**: ε-greedy, softmax, UCB formula, entropy bonus, intrinsic motivation concepts.
