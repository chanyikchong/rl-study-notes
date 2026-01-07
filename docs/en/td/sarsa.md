# SARSA

## Interview Summary

**SARSA** (State-Action-Reward-State-Action) is an on-policy TD control algorithm. It updates Q-values using the tuple \((S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})\) — hence the name. The update uses the action actually taken, making it on-policy. Converges to optimal Q under GLIE conditions. More conservative than Q-learning because it accounts for its own exploration.

**What to memorize**: SARSA update rule, on-policy nature, convergence to \(Q^\pi\) (not \(Q^*\) unless exploration vanishes), comparison with Q-learning.

---

## Core Definitions

### SARSA Update Rule

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

**Components**:
- \(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})\): TD target (what we move toward)
- \(Q(S_t, A_t)\): Current estimate
- \(\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\): TD error

### On-Policy

The target \(Q(S_{t+1}, A_{t+1})\) uses the action \(A_{t+1}\) that was actually taken (sampled from the policy). This means SARSA learns the value of the policy being followed, including exploration.

### Why "SARSA"?

The name comes from the quintuple needed: \((S, A, R, S', A')\).

---

## Math and Derivations

### Relationship to Bellman Equation

SARSA is a sample-based approximation of:

$$Q^\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) | S_t=s, A_t=a]$$

Each update is a stochastic gradient step toward this fixed point.

### Convergence

Under standard conditions (GLIE policy, appropriate learning rate decay), SARSA converges:

$$Q(s,a) \to Q^\pi(s,a)$$

If \(\pi\) becomes greedy in the limit, then \(Q \to Q^*\).

### TD Error

$$\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$

The TD error measures surprise: how different was the actual return from what we expected?

---

## Algorithm Sketch

```
Algorithm: SARSA (On-policy TD Control)

Input: α, γ, ε
Output: Q ≈ Q*

1. Initialize Q(s,a) arbitrarily (Q(terminal, ·) = 0)
2. For each episode:
     S = initial state
     A = ε-greedy action from Q(S, ·)
     While S is not terminal:
         Take action A, observe R, S'
         A' = ε-greedy action from Q(S', ·)
         Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
         S ← S', A ← A'
3. Return Q
```

**Key point**: \(A'\) is chosen before the update, using the same policy that generated \(A\).

---

## Common Pitfalls

1. **Confusing with Q-learning**: SARSA uses \(A'\) from policy; Q-learning uses \(\max_{a'}\). Different algorithms!

2. **Not choosing A' before update**: You must select \(A'\) to compute the target. Don't update then choose.

3. **Terminal state handling**: When \(S'\) is terminal, use \(Q(S', A') = 0\).

4. **Expecting Q* with exploration**: With fixed \(\epsilon > 0\), SARSA converges to \(Q^\epsilon\) (value of ε-greedy policy), not \(Q^*\).

5. **Learning rate**: Too high → oscillation. Too low → slow. Decay over time for convergence.

---

## Mini Example

**Cliff Walking:**

```
+---+---+---+---+
| S |   |   | G |
+---+---+---+---+
| C | C | C | C |   C = cliff (R = -100, return to S)
+---+---+---+---+
```

Reward: -1 per step, -100 for cliff.

**SARSA with ε = 0.1**: Learns to walk along the safe path (top row) because it accounts for occasional random falls into the cliff.

**Q-learning**: Learns to walk near the cliff edge because it assumes optimal (greedy) action will be taken.

**Key insight**: SARSA is more conservative — it learns the value of what it actually does (including mistakes).

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why is SARSA called "on-policy"?</summary>

**Answer**: Because it uses the action \(A'\) actually taken under the current policy to compute the TD target.

**Explanation**: The target \(R + \gamma Q(S', A')\) reflects what the policy will actually do next (including exploration). This means SARSA learns \(Q^\pi\) for the behavior policy, not the optimal \(Q^*\).

**Contrast**: Q-learning uses \(\max_{a'} Q(S', a')\) regardless of what action was taken — it's off-policy.

**Common pitfall**: With fixed \(\epsilon\), SARSA converges to the ε-greedy policy's values, which are lower than optimal due to exploration.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> In cliff walking, why does SARSA learn a safer path than Q-learning?</summary>

**Answer**: SARSA learns the value of the ε-greedy policy, which includes occasional random cliff falls. The safe path has lower expected penalty under this policy.

**Explanation**: SARSA evaluates "what will happen if I keep using this exploratory policy?" Near the cliff, random actions sometimes fall in. Q-learning evaluates "what's optimal assuming perfect execution?" — it ignores exploration.

**Key insight**: SARSA is more robust when the policy being learned is also being executed.

**Common pitfall**: Thinking Q-learning's path is "wrong" — it's optimal for a greedy policy, just risky during learning.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Write the SARSA update and explain why A' must be chosen from the policy.</summary>

**Answer**:

$$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma Q(S',A') - Q(S,A)]$$

\(A'\) must be from the policy because SARSA estimates:

$$Q^\pi(s,a) = \mathbb{E}_\pi[R + \gamma Q^\pi(S',A')]$$

The expectation is over \(A' \sim \pi(\cdot|S')\). By sampling \(A'\) from \(\pi\), we get an unbiased estimate.

**Key point**: If \(A'\) came from a different policy, we'd be estimating something other than \(Q^\pi\).

**Common pitfall**: Choosing \(A'\) greedily would make it off-policy (like Q-learning).
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> How does SARSA's convergence differ from Q-learning's?</summary>

**Answer**:
- **SARSA**: Converges to \(Q^\pi\) where \(\pi\) is the behavior policy (e.g., ε-greedy). Only converges to \(Q^*\) if \(\epsilon \to 0\) (GLIE).
- **Q-learning**: Converges to \(Q^*\) directly, regardless of behavior policy (as long as all pairs visited).

**Explanation**: SARSA's target depends on what action will be taken; Q-learning's target assumes optimal action. SARSA needs GLIE; Q-learning doesn't.

**Key equation difference**:
- SARSA: \(R + \gamma Q(S', A')\) where \(A' \sim \pi\)
- Q-learning: \(R + \gamma \max_{a'} Q(S', a')\)

**Common pitfall**: Assuming SARSA always finds the optimal policy — it finds optimal for the limiting policy.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> SARSA is oscillating and not converging. What's wrong?</summary>

**Answer**: Likely issues:
1. **Learning rate too high**: Try \(\alpha = 0.1\) or lower
2. **No learning rate decay**: Need \(\sum \alpha = \infty\), \(\sum \alpha^2 < \infty\)
3. **ε not decaying**: Fixed exploration prevents convergence to optimal
4. **Environment is stochastic**: May need more episodes
5. **Initialization issues**: Poor initial Q can slow convergence

**Explanation**: Convergence requires proper learning rate schedule. Common choice: \(\alpha_n = \frac{1}{n^{0.8}}\) or similar.

**Common pitfall**: Using constant \(\alpha = 1.0\) — this doesn't converge, just oscillates around the solution.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 6.4
- **Rummery & Niranjan (1994)**, On-Line Q-Learning Using Connectionist Systems
- **Singh et al. (2000)**, Convergence Results for Single-Step On-Policy RL Algorithms

**What to memorize for interviews**: SARSA update, on-policy definition, cliff walking comparison with Q-learning, convergence to \(Q^\pi\).

**Code example**: [sarsa.py](../../../rl_examples/algorithms/sarsa.py)
