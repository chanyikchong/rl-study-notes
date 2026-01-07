# Bellman Equations

## Interview Summary

**Bellman equations** express value functions recursively: the value of a state equals immediate reward plus discounted value of successor states. The **Bellman expectation equation** holds for any policy \(\pi\). The **Bellman optimality equation** holds for optimal \(V^*\) and \(Q^*\). These equations are the foundation of RL — dynamic programming, TD learning, and Q-learning all exploit them.

**What to memorize**: Both forms of Bellman expectation (for \(V^\pi\) and \(Q^\pi\)), Bellman optimality equations, backup diagrams.

---

## Core Definitions

### Bellman Expectation Equation for \(V^\pi\)

$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

**Meaning**: The value of state \(s\) is the expected immediate reward plus discounted expected value of the next state, where expectations are over actions (from policy) and transitions (from environment).

### Bellman Expectation Equation for \(Q^\pi\)

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s', a')$$

**Meaning**: The value of taking action \(a\) in state \(s\) equals immediate reward plus discounted expected Q-value of the next state-action pair.

### Bellman Optimality Equation for \(V^*\)

$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]$$

**Meaning**: The optimal value is achieved by taking the best action (max instead of expectation over policy).

### Bellman Optimality Equation for \(Q^*\)

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s', a')$$

**Meaning**: The optimal Q-value equals immediate reward plus discounted optimal value of the next state.

---

## Math and Derivations

### Deriving Bellman Expectation for \(V^\pi\)

Starting from the definition:

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

$$= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s]$$

$$= \mathbb{E}_\pi[R_{t+1} | S_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s]$$

Expanding the expectations:

$$= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s') + \gamma \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) V^\pi(s')$$

Simplifying (assuming \(R(s,a)\) doesn't depend on \(s'\)):

$$= \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

### Matrix Form (Linear System)

For a fixed policy, the Bellman equation is linear in \(V^\pi\):

$$\mathbf{V}^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi$$

Solving:

$$\mathbf{V}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{R}^\pi$$

**Complexity**: \(O(|S|^3)\) — only feasible for small state spaces.

### Why Optimality Uses Max

For the optimal policy, we always choose the best action:

$$\pi^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_a Q^*(s,a) \\ 0 & \text{otherwise} \end{cases}$$

So the expectation over policy becomes a max:

$$V^*(s) = \sum_a \pi^*(a|s) Q^*(s,a) = \max_a Q^*(s,a)$$

---

## Algorithm Sketch

Bellman equations enable key algorithms:

| Algorithm | Uses | Key Idea |
|-----------|------|----------|
| Policy Evaluation | Bellman expectation for \(V^\pi\) | Iterate until convergence |
| Policy Iteration | Both equations | Alternate evaluation and improvement |
| Value Iteration | Bellman optimality for \(V^*\) | Iterate with max operator |
| Q-Learning | Bellman optimality for \(Q^*\) | Sample-based update with max |
| SARSA | Bellman expectation for \(Q^\pi\) | Sample-based on-policy update |

### Backup Diagrams

**V-function backup** (expectation):
```
      (s)
     / | \
    a  a  a     ← sum over actions (weighted by π)
   /|\ |\ |\
  s' s' s' ...  ← sum over next states (weighted by P)
```

**Q-function backup** (optimality):
```
    (s,a)
    / | \
   s' s' s'     ← sum over next states (weighted by P)
   |  |  |
   max a'       ← max over next actions
```

---

## Common Pitfalls

1. **Confusing expectation vs optimality equations**: Expectation uses policy \(\pi\) for weighting; optimality uses max. Using the wrong one leads to incorrect algorithms.

2. **Forgetting the self-referential nature**: Bellman equations define \(V\) in terms of \(V\) — they're fixed-point equations, not direct formulas.

3. **Matrix inversion scaling**: The closed-form solution \((\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1}\) is \(O(|S|^3)\) — impractical for large state spaces.

4. **Assuming deterministic transitions**: The equations include \(\sum_{s'} P(s'|s,a)\) — don't forget stochasticity in transitions.

---

## Mini Example

**Two-state MDP:**

- States: \(S = \{A, B\}\)
- Actions: \(A = \{0, 1\}\) (same effect in both states)
- Transitions: From A, action 0 → stay A, action 1 → go to B. From B, any action → stay B.
- Rewards: \(R(A, 0) = 1\), \(R(A, 1) = 0\), \(R(B, \cdot) = 2\)
- \(\gamma = 0.9\)

**Policy**: Always action 0.

**Bellman equation for \(V^\pi(A)\)**:

$$V^\pi(A) = R(A, 0) + \gamma \cdot 1 \cdot V^\pi(A) = 1 + 0.9 V^\pi(A)$$

$$V^\pi(A) = \frac{1}{1-0.9} = 10$$

**Bellman optimality for \(V^*(A)\)**:

$$V^*(A) = \max\{1 + 0.9 V^*(A), \; 0 + 0.9 V^*(B)\}$$

Since \(V^*(B) = \frac{2}{1-0.9} = 20\):

$$V^*(A) = \max\{10, \; 0 + 0.9 \times 20\} = \max\{10, 18\} = 18$$

**Insight**: Optimal policy is to go to B (action 1) because \(V^*(B) > V^*(A)\) under staying.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What is the key difference between Bellman expectation and Bellman optimality equations?</summary>

**Answer**:
- **Expectation**: Uses policy \(\pi(a|s)\) to weight actions — describes value under a given policy
- **Optimality**: Uses \(\max_a\) over actions — describes value under the optimal policy

**Explanation**: The expectation equation evaluates any fixed policy. The optimality equation finds the best possible behavior. Policy iteration uses expectation (to evaluate), then improvement (using max). Value iteration uses optimality directly.

**Key equations**:
- Expectation: \(V^\pi(s) = \sum_a \pi(a|s)[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')]\)
- Optimality: \(V^*(s) = \max_a[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')]\)

**Common pitfall**: Using optimality equation when you want to evaluate a specific (non-optimal) policy.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why are Bellman equations called "recursive" or "self-referential"?</summary>

**Answer**: The value of a state is defined in terms of values of other states (successors), which are themselves defined the same way.

**Explanation**: \(V(s)\) appears on both sides of the equation. This isn't a bug — it's a fixed-point equation. We solve it by iteration (dynamic programming) or sampling (TD learning). The recursion terminates at terminal states where \(V(s_{terminal}) = 0\).

**Key equation**: \(V(s) = R + \gamma V(s')\) — \(V\) defined in terms of \(V\).

**Common pitfall**: Thinking you can directly compute \(V(s)\) without iteration. The equation is a constraint, not a formula.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Write the Bellman equation for Q*(s,a) and explain each term.</summary>

**Answer**:

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')$$

**Explanation**:
- \(R(s,a)\): Immediate reward for taking action \(a\) in state \(s\)
- \(\gamma\): Discount factor (how much we value future)
- \(\sum_{s'} P(s'|s,a)\): Expectation over stochastic transitions
- \(\max_{a'} Q^*(s',a')\): Best action value in next state (optimality)

**Common pitfall**: In Q-learning, we sample \(s'\) instead of summing. The update becomes: \(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]\)
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Why can we solve V^π in closed form but not V*?</summary>

**Answer**: \(V^\pi\) satisfies a linear system (can invert a matrix). \(V^*\) has a \(\max\) operator, making it nonlinear.

**Explanation**:
- Bellman expectation: \(\mathbf{V}^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi \Rightarrow \mathbf{V}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{R}^\pi\)
- Bellman optimality: The \(\max\) prevents matrix form. We must use iterative methods (value iteration).

**Key insight**: This is why policy iteration alternates — evaluation (linear) can be solved exactly, but finding the best policy requires the max.

**Common pitfall**: Attempting to "analytically solve" for \(V^*\) — it requires iteration.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your value iteration is not converging. What could be wrong?</summary>

**Answer**: Common issues:
1. \(\gamma = 1\) for continuing tasks (no convergence guarantee)
2. Bug in transition probabilities (don't sum to 1)
3. Reward magnitude too large (numerical instability)
4. Not enough iterations (convergence can be slow)
5. Using wrong Bellman equation (expectation instead of optimality)

**Explanation**: Value iteration converges for \(\gamma < 1\) because the Bellman operator is a contraction mapping. Check: (1) terminal states have value 0, (2) \(\gamma < 1\), (3) transitions are proper probability distributions, (4) monitor \(\max_s |V_{k+1}(s) - V_k(s)|\) for convergence.

**Common pitfall**: Setting convergence threshold too tight (e.g., \(10^{-10}\)) leads to unnecessary iterations. Practical threshold is often \(10^{-4}\) or based on policy stability.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 3.5, 4.1
- **Bellman (1957)**, Dynamic Programming
- **Bertsekas & Tsitsiklis**, Neuro-Dynamic Programming, Chapter 2

**What to memorize for interviews**: All four Bellman equations (expectation and optimality, for V and Q), the recursive structure, why optimality uses max, contraction mapping intuition for convergence.
