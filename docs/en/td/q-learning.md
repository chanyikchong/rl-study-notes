# Q-Learning

## Interview Summary

**Q-learning** is an off-policy TD control algorithm that learns \(Q^*\) directly. The key insight: use \(\max_{a'} Q(s', a')\) in the target, regardless of what action was actually taken. This allows learning optimal values while following an exploratory policy. One of the most important algorithms in RL — foundation for DQN.

**What to memorize**: Q-learning update rule, off-policy nature (behavior vs target policy), convergence to \(Q^*\), comparison with SARSA.

---

## Core Definitions

### Q-Learning Update Rule

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

**Key difference from SARSA**: Uses \(\max_{a'}\) instead of the action actually taken.

### Off-Policy Learning

- **Behavior policy** (\(b\)): Policy used to generate actions (e.g., ε-greedy)
- **Target policy** (\(\pi\)): Policy being learned (greedy w.r.t. Q)

Q-learning learns about the greedy policy \(\pi^*\) while following any behavior policy \(b\) that covers all actions.

### TD Target

$$\text{Target} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$$

This is the Bellman optimality equation sampled at one transition.

---

## Math and Derivations

### Relationship to Bellman Optimality

Q-learning approximates:

$$Q^*(s, a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') | S_t=s, A_t=a]$$

Each update is a sample-based step toward this fixed point.

### Convergence Theorem

**Theorem** (Watkins & Dayan, 1992): Q-learning converges to \(Q^*\) with probability 1 if:
1. All state-action pairs visited infinitely often
2. Learning rate satisfies: \(\sum_t \alpha_t = \infty\), \(\sum_t \alpha_t^2 < \infty\)
3. Rewards are bounded

**Note**: Convergence is to \(Q^*\) regardless of the behavior policy!

### Why Off-Policy Works

The max operator decouples learning from behavior:
- We sample \((s, a, r, s')\) from any policy
- We update toward optimal regardless of how we got there
- As long as we visit all pairs, we learn about all of them

---

## Algorithm Sketch

```
Algorithm: Q-Learning (Off-policy TD Control)

Input: α, γ, ε
Output: Q ≈ Q*

1. Initialize Q(s,a) arbitrarily (Q(terminal, ·) = 0)
2. For each episode:
     S = initial state
     While S is not terminal:
         A = ε-greedy action from Q(S, ·)
         Take action A, observe R, S'
         Q(S,A) ← Q(S,A) + α[R + γ max_a' Q(S',a') - Q(S,A)]
         S ← S'
3. Return Q
```

**Simplicity**: Note we don't need to choose \(A'\) before updating — we just compute max.

---

## Common Pitfalls

1. **Overestimation bias**: The max operator tends to overestimate Q-values (see Double Q-learning).

2. **Exploration required**: Off-policy doesn't mean no exploration needed! Must still visit all (s,a) pairs.

3. **Function approximation instability**: Q-learning with function approximation can diverge (deadly triad).

4. **Confusing with SARSA**: Q-learning uses max; SARSA uses actual next action.

5. **Ignoring behavior policy requirements**: Behavior policy must cover all actions for convergence.

---

## Mini Example

**Simple Grid:**
```
+---+---+
| S | G |
+---+---+
```
Actions: left, right. Reward: +1 at G, 0 otherwise. \(\gamma = 0.9\), \(\alpha = 0.5\).

**Episode 1** (behavior: random):
- S, right, R=1, G (terminal)
- Update: \(Q(S, right) = 0 + 0.5[1 + 0.9 \times 0 - 0] = 0.5\)

**Episode 2** (behavior: ε-greedy):
- S, right, R=1, G (terminal) — greedy chose right
- Update: \(Q(S, right) = 0.5 + 0.5[1 + 0 - 0.5] = 0.75\)

**Episode 3**:
- \(Q(S, right) = 0.75 + 0.5[1 - 0.75] = 0.875\)

**Converges to**: \(Q^*(S, right) = 1\), \(Q^*(S, left) = 0.9\) (if left loops to S).

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What makes Q-learning "off-policy"?</summary>

**Answer**: The TD target uses \(\max_{a'} Q(s', a')\), which assumes the greedy action will be taken next — regardless of what action the behavior policy actually chose.

**Explanation**: Off-policy means the policy being evaluated/improved (target policy = greedy) differs from the policy generating data (behavior policy = exploratory). The max decouples these.

**Key insight**: This allows learning optimal values while exploring.

**Common pitfall**: Thinking off-policy means "no exploration needed." You still need exploration to visit all states!
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why does Q-learning suffer from overestimation bias?</summary>

**Answer**: The max over noisy Q-estimates tends to select overestimated values.

**Explanation**: If \(Q(s', a_1)\) is overestimated due to noise, \(\max_a Q(s', a)\) picks it. Over many updates, this bias accumulates. Even unbiased noise leads to biased max.

**Key equation**: \(\mathbb{E}[\max_i X_i] \geq \max_i \mathbb{E}[X_i]\) (Jensen-like inequality for max).

**Solution**: Double Q-learning uses two Q-functions to decouple selection and evaluation.

**Common pitfall**: Ignoring this in DQN leads to overestimated values and suboptimal policies.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Compare Q-learning and SARSA update rules. When do they differ?</summary>

**Answer**:
- **Q-learning**: \(Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \max_{a'} Q(S',a') - Q(S,A)]\)
- **SARSA**: \(Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma Q(S',A') - Q(S,A)]\)

They differ when \(A' \neq \arg\max_{a'} Q(S', a')\), i.e., when the behavior policy doesn't choose the greedy action.

**Explanation**: With ε-greedy, this happens ε fraction of the time. With deterministic greedy, they're identical.

**Key insight**: As ε → 0, SARSA → Q-learning. But during learning, the difference matters.

**Common pitfall**: Thinking they're the same algorithm with different names.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> State the convergence conditions for Q-learning.</summary>

**Answer**: Q-learning converges to \(Q^*\) w.p. 1 if:
1. All (s,a) pairs visited infinitely often
2. Learning rate: \(\sum_t \alpha_t = \infty\) and \(\sum_t \alpha_t^2 < \infty\)
3. Rewards are bounded
4. MDP is finite

**Explanation**: Condition 1 ensures all pairs are learned. Condition 2 ensures we can reach the target (\(\sum \alpha = \infty\)) but stop there (\(\sum \alpha^2 < \infty\) bounds noise).

**Common learning rates**: \(\alpha_t = 1/t\) or \(\alpha_t = 1/t^{0.8}\).

**Common pitfall**: Constant \(\alpha\) doesn't satisfy condition 2 — no convergence guarantee (but often works in practice).
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Q-learning is diverging with function approximation. Why?</summary>

**Answer**: The "deadly triad" — combination of:
1. Function approximation (generalization)
2. Bootstrapping (using estimates in targets)
3. Off-policy learning

This combination can lead to instability and divergence.

**Explanation**: Function approximation means updating one state affects others. Bootstrapping compounds errors. Off-policy means the data distribution doesn't match what we're evaluating.

**Solutions**:
- Target networks (DQN)
- Experience replay with priority
- Soft target updates
- Regularization

**Common pitfall**: Assuming tabular convergence guarantees extend to function approximation.
</details>

---

## References

- **Watkins & Dayan (1992)**, Q-Learning — original convergence proof
- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 6.5
- **Mnih et al. (2015)**, Human-level control through deep RL — DQN paper

**What to memorize for interviews**: Update rule with max, off-policy definition, convergence conditions, overestimation bias, deadly triad.

**Code example**: [q_learning.py](../../../rl_examples/algorithms/q_learning.py)
