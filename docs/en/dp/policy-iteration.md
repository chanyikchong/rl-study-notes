# Policy Iteration

## Interview Summary

**Policy iteration** alternates between policy evaluation (compute \(V^\pi\)) and policy improvement (make policy greedy with respect to \(V^\pi\)). It converges to the optimal policy in a finite number of iterations. Each improvement step is guaranteed to be no worse. This is the foundation for understanding actor-critic methods.

**What to memorize**: Two-step structure (evaluate, improve), policy improvement theorem, guaranteed monotonic improvement, finite convergence.

---

## Core Definitions

### Algorithm Structure

1. **Policy Evaluation**: Compute \(V^\pi\) for current policy \(\pi\)
2. **Policy Improvement**: Create new policy \(\pi'\) greedy w.r.t. \(V^\pi\)
3. Repeat until policy doesn't change

### Policy Improvement Step

$$\pi'(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

Or equivalently:

$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

**Meaning**: For each state, pick the action that looks best given current value estimates.

---

## Math and Derivations

### Policy Improvement Theorem

**Theorem**: For any policy \(\pi\), if we define:

$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

Then:

$$V^{\pi'}(s) \geq V^\pi(s) \quad \forall s$$

**Proof sketch**:

$$V^\pi(s) \leq Q^\pi(s, \pi'(s)) = \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t=s, A_t=\pi'(s)]$$

$$\leq \mathbb{E}[R_{t+1} + \gamma Q^\pi(S_{t+1}, \pi'(S_{t+1})) | S_t=s, A_t=\pi'(s)]$$

$$\leq \ldots \leq V^{\pi'}(s)$$

By repeated application, the new policy is at least as good everywhere.

### Convergence to Optimality

**Key insight**: In a finite MDP, there are finitely many deterministic policies (\(|A|^{|S|}\)). Each iteration strictly improves or stays the same. Since we can't improve forever, we must converge to optimal.

### When Does Improvement Stop?

Improvement stops when:

$$V^\pi(s) = \max_a Q^\pi(s, a) \quad \forall s$$

This is exactly the Bellman optimality equation! So \(V^\pi = V^*\) and \(\pi = \pi^*\).

---

## Algorithm Sketch

```
Algorithm: Policy Iteration

Input: MDP (S, A, P, R, γ)
Output: Optimal policy π*, optimal value V*

1. Initialize π(s) arbitrarily for all s
2. Repeat:
     # Policy Evaluation
     Compute V^π using iterative policy evaluation

     # Policy Improvement
     policy_stable = True
     For each s ∈ S:
         old_action = π(s)
         π(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
         If old_action ≠ π(s):
             policy_stable = False
   Until policy_stable
3. Return π, V
```

### Complexity

- **Per evaluation**: \(O(|S|^2|A|)\) per iteration, \(O(\frac{1}{1-\gamma})\) iterations
- **Number of policy iterations**: At most \(|A|^{|S|}\), but typically \(O(|S||A|)\) in practice
- **Total**: Often much faster than value iteration in practice

---

## Common Pitfalls

1. **Full vs partial evaluation**: You don't need to converge \(V^\pi\) exactly. Even one sweep of policy evaluation can work (modified policy iteration).

2. **Breaking ties arbitrarily**: When multiple actions have the same Q-value, any choice is fine. But be consistent to detect convergence.

3. **Stochastic policies**: Standard policy iteration produces deterministic policies. For stochastic, you need soft improvement.

4. **Confusing with value iteration**: Policy iteration evaluates fully then improves. Value iteration combines into one max step.

---

## Mini Example

**Grid World (2x2):**

```
+---+---+
| A | G |   G = goal, reward +1
+---+---+
| B | C |   All moves have reward 0 except reaching G
+---+---+
```

Actions: up, down, left, right (deterministic). \(\gamma = 0.9\).

**Initial policy**: All states go right.

**Iteration 1 - Evaluate**:
- \(V(G) = 0\) (terminal)
- \(V(A) = 0 + 0.9 \times 0 = 0\)... wait, reaching G gives +1
- Actually: \(V(A) = 1\) (reward for A→G)
- \(V(B) = 0 + 0.9 \times V(C) = 0.9 V(C)\)
- \(V(C) = 0 + 0.9 \times 0 = 0\) (C→right→C... stuck!)

**Iteration 1 - Improve**:
- At B: compare up (\(V(A)=1\)) vs right (\(V(C)=0\)) → switch to up
- At C: compare up (\(V(G)=0\) but reward 1) vs right (0) → switch to up

**After improvement**: B→up, C→up, A→right (already optimal).

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why is policy iteration guaranteed to converge?</summary>

**Answer**: There are finitely many deterministic policies. Each improvement step produces a policy that is at least as good (policy improvement theorem). Since we can only improve a finite number of times, we must converge.

**Explanation**: If \(\pi' \neq \pi\) after improvement, then \(V^{\pi'} \geq V^\pi\) with strict inequality for at least one state. Since there are only \(|A|^{|S|}\) possible policies, we can't improve forever.

**Key insight**: Convergence is finite (not just asymptotic like value iteration).

**Common pitfall**: Thinking policy iteration might oscillate. The improvement theorem guarantees monotonic progress.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> What's the relationship between policy iteration and actor-critic?</summary>

**Answer**: Actor-critic is essentially online policy iteration:
- **Critic** = policy evaluation (estimates \(V^\pi\) or \(Q^\pi\))
- **Actor** = policy improvement (updates policy toward better actions)

**Explanation**: Policy iteration does full evaluation then full improvement. Actor-critic interleaves: update value estimate a little, update policy a little, repeat. The underlying logic is the same.

**Common pitfall**: Thinking actor-critic is fundamentally different. It's really just incremental policy iteration with function approximation.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> State the policy improvement theorem precisely.</summary>

**Answer**: Let \(\pi\) be any policy and \(\pi'\) be greedy with respect to \(Q^\pi\):

$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

Then:

$$V^{\pi'}(s) \geq V^\pi(s) \quad \forall s \in S$$

with equality iff \(\pi\) is already optimal.

**Explanation**: Acting greedily with respect to current value estimates never makes things worse. If it doesn't change the policy, we've found the optimal.

**Key equation**: \(V^\pi(s) \leq Q^\pi(s, \pi'(s)) \leq V^{\pi'}(s)\)

**Common pitfall**: This assumes we compute \(Q^\pi\) correctly. With function approximation, no guarantees!
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Why might you not need exact policy evaluation?</summary>

**Answer**: Even approximate \(V^\pi\) can yield the correct greedy policy. The relative ordering of actions matters, not absolute values.

**Explanation**: If \(Q^\pi(s,a_1) > Q^\pi(s,a_2)\), small errors in \(V\) won't change this ordering. So we can use **modified policy iteration**: do only a few evaluation sweeps before improving. Extreme case: one sweep per improvement = value iteration.

**Key insight**: Faster overall convergence by trading exact evaluation for more improvement steps.

**Common pitfall**: Over-investing in evaluation. A few sweeps often suffice.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your policy iteration keeps switching between two policies. What's wrong?</summary>

**Answer**: This shouldn't happen with correct implementation. Debug steps:
1. Check for ties in Q-values with inconsistent tie-breaking
2. Verify evaluation convergence threshold isn't too loose
3. Check for floating-point precision issues
4. Ensure you're comparing policies correctly (not values)

**Explanation**: The policy improvement theorem guarantees strict improvement or optimality. Oscillation indicates a bug. Most likely: tie-breaking isn't deterministic, or evaluation isn't converging properly.

**Common pitfall**: Using `>` vs `>=` in argmax can cause issues with ties. Use a consistent rule like "smallest action index wins ties."
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 4.2-4.3
- **Howard (1960)**, Dynamic Programming and Markov Processes — original policy iteration
- **Bertsekas**, Dynamic Programming and Optimal Control

**What to memorize for interviews**: Two-step structure, policy improvement theorem statement, why convergence is guaranteed, connection to actor-critic.

**Code example**: [policy_iteration.py](../../../rl_examples/algorithms/policy_iteration.py)
