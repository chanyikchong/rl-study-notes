# Policy and Value Functions

## Interview Summary

A **policy** \(\pi(a|s)\) maps states to action probabilities. The **state-value function** \(V^\pi(s)\) gives expected return starting from \(s\) and following \(\pi\). The **action-value function** \(Q^\pi(s,a)\) gives expected return starting from \(s\), taking action \(a\), then following \(\pi\). The goal of RL is to find the **optimal policy** \(\pi^*\) that maximizes value for all states.

**What to memorize**: Definitions of \(V^\pi(s)\), \(Q^\pi(s,a)\), relationship \(V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)\), optimal value notation \(V^*(s)\), \(Q^*(s,a)\).

---

## Core Definitions

### Policy

A **policy** \(\pi\) specifies behavior:

- **Deterministic policy**: \(\pi(s) = a\) — maps state directly to action
- **Stochastic policy**: \(\pi(a|s) = P(A_t = a | S_t = s)\) — probability distribution over actions

**Why stochastic?**
1. Can represent mixed strategies (game theory)
2. Enables exploration during learning
3. Some problems require randomization for optimality (adversarial settings)

### State-Value Function

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s\right]$$

**Meaning**: Expected cumulative discounted reward starting from state \(s\) and following policy \(\pi\) thereafter.

### Action-Value Function

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s, A_t = a\right]$$

**Meaning**: Expected cumulative discounted reward starting from state \(s\), taking action \(a\), then following policy \(\pi\).

### Relationship Between V and Q

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) \, Q^\pi(s, a)$$

**Meaning**: The state value is the expected action value under the policy's action distribution.

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')$$

**Meaning**: The action value is immediate reward plus discounted expected value of the next state.

---

## Math and Derivations

### Deriving V from Q

Starting from definitions:

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

$$= \mathbb{E}_\pi[\mathbb{E}_\pi[G_t | S_t = s, A_t] | S_t = s]$$

$$= \sum_a P(A_t = a | S_t = s) \cdot \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

$$= \sum_a \pi(a|s) \, Q^\pi(s, a)$$

### Optimal Value Functions

The **optimal state-value function**:

$$V^*(s) = \max_\pi V^\pi(s) = \max_a Q^*(s, a)$$

The **optimal action-value function**:

$$Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

### Optimal Policy

Given \(Q^*\), the optimal policy is:

$$\pi^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_{a'} Q^*(s, a') \\ 0 & \text{otherwise} \end{cases}$$

**Key insight**: If we know \(Q^*\), acting optimally is easy — just pick the action with highest Q-value.

### Advantage Function

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**Meaning**: How much better is action \(a\) compared to the average action under \(\pi\)?

**Properties**:
- \(\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0\) (advantages average to zero)
- \(A^\pi(s, a) > 0\) means action \(a\) is better than average
- Used heavily in policy gradient methods (A2C, PPO)

---

## Algorithm Sketch

Value functions are not algorithms themselves but are **what we learn**:

1. **Policy Evaluation**: Given \(\pi\), compute \(V^\pi\) or \(Q^\pi\)
2. **Policy Improvement**: Given \(V^\pi\) or \(Q^\pi\), construct better policy
3. **Value-based methods**: Learn \(Q^*\), derive \(\pi^*\) as argmax
4. **Policy-based methods**: Directly optimize \(\pi\) parameters

---

## Common Pitfalls

1. **Confusing \(V\) and \(Q\)**:
   - \(V(s)\) — no action specified, averages over policy
   - \(Q(s,a)\) — specific action given

2. **Forgetting the expectation**: Values are expected returns, not single-sample returns.

3. **Optimal ≠ Greedy with respect to current estimates**: Being greedy with respect to \(Q^\pi\) gives the improved policy, but only greedy with respect to \(Q^*\) gives the optimal policy.

4. **Not understanding when Q-values exist**: Q-values are well-defined for any policy, not just the optimal one.

---

## Mini Example

**Two-state MDP:**

```
State A ──action 0 (r=5)──> Terminal
    │
    └──action 1 (r=0)──> State B ──action 0 (r=10)──> Terminal
```

With \(\gamma = 0.9\):

**For policy "always action 0":**
- \(V^\pi(A) = 5\), \(V^\pi(B) = 10\)
- \(Q^\pi(A, 0) = 5\), \(Q^\pi(A, 1) = 0 + 0.9 \times 10 = 9\)

**Optimal policy**: Take action 1 at A (to reach B), then action 0 at B.
- \(V^*(A) = 9\), \(V^*(B) = 10\)

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What is the difference between V(s) and Q(s,a)?</summary>

**Answer**:
- \(V^\pi(s)\): Expected return from state \(s\), averaging over actions according to \(\pi\)
- \(Q^\pi(s,a)\): Expected return from state \(s\) after taking specific action \(a\)

**Explanation**: \(V\) summarizes the value of being in a state (under a policy), while \(Q\) evaluates specific state-action pairs. They're related by \(V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)\).

**Key equation**: \(V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)\)

**Common pitfall**: Using \(Q\) when you mean \(V\) or vice versa. In Q-learning we learn \(Q\); in policy evaluation we often compute \(V\).
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why is the advantage function useful in policy gradients?</summary>

**Answer**: The advantage \(A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)\) reduces variance in policy gradient estimates while keeping the gradient unbiased.

**Explanation**: Policy gradients weight actions by returns. High-return states would increase probability of all actions (even bad ones). The advantage subtracts a baseline \(V^\pi(s)\), so only actions better than average are reinforced.

**Key equation**: \(A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)\)

**Common pitfall**: Forgetting that \(\mathbb{E}[A^\pi(s,a)] = 0\). The baseline doesn't bias the gradient because it doesn't depend on the action.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Prove that V*(s) = max_a Q*(s,a)</summary>

**Answer**: By definition of optimality.

**Explanation**:

$$V^*(s) = \max_\pi V^\pi(s) = \max_\pi \sum_a \pi(a|s) Q^\pi(s,a)$$

For the optimal policy, we act greedily:

$$\pi^*(a|s) = 1 \text{ if } a = \arg\max_{a'} Q^*(s,a')$$

Therefore:

$$V^*(s) = \sum_a \pi^*(a|s) Q^*(s,a) = Q^*(s, \arg\max_a Q^*(s,a)) = \max_a Q^*(s,a)$$

**Key equation**: \(V^*(s) = \max_a Q^*(s,a)\)

**Common pitfall**: This only holds for \(V^*\) and \(Q^*\), not for arbitrary policy values.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Given Q*(s,a), how do you extract the optimal policy?</summary>

**Answer**: \(\pi^*(s) = \arg\max_a Q^*(s,a)\)

**Explanation**: Once we have the optimal action-value function, the optimal action in any state is simply the one with the highest Q-value. This is why Q-learning is popular — we can derive the optimal policy directly from learned Q-values without needing a model.

**Key equation**: \(\pi^*(s) = \arg\max_a Q^*(s,a)\)

**Common pitfall**: This requires knowing \(Q^*\) exactly. During learning, we have estimates \(\hat{Q}\), and acting greedily with respect to estimates (exploitation) must be balanced with exploration.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your Q-values are all very large positive numbers. What might be wrong?</summary>

**Answer**: Possible issues:
1. Discount factor \(\gamma\) too close to 1 in continuing tasks (values explode)
2. Positive reward cycles (agent loops to accumulate reward)
3. Missing terminal state handling
4. Reward scale too large
5. Value function diverging (deadly triad)

**Explanation**: Q-values should roughly reflect discounted cumulative rewards. If rewards are bounded (e.g., -1 to +1) and \(\gamma = 0.99\), max Q should be around \(1/(1-0.99) = 100\). Much larger values indicate something is wrong.

**Common pitfall**: Not normalizing or clipping rewards. Large rewards cause large gradients and unstable learning. DQN clips rewards to \([-1, 1]\).
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 3.5-3.8
- **Szepesvári (2010)**, Algorithms for Reinforcement Learning
- **Silver's RL Course**, Lecture 2: Markov Decision Processes

**What to memorize for interviews**: Definitions of \(V^\pi\), \(Q^\pi\), \(V^*\), \(Q^*\), advantage function, V-Q relationship, how to extract optimal policy from \(Q^*\).
