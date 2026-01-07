# Value Iteration

## Interview Summary

**Value iteration** directly computes optimal values \(V^*\) by iterating the Bellman optimality equation. Unlike policy iteration, it doesn't maintain an explicit policy during iteration — just the max over actions. Converges asymptotically (not in finite steps). Simpler to implement than policy iteration; often preferred when \(|A|\) is small.

**What to memorize**: Update rule with max, relationship to Bellman optimality, difference from policy iteration, convergence guarantee.

---

## Core Definitions

### Bellman Optimality Operator

$$(\mathcal{T}^* V)(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

### Iterative Update

$$V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$

**Key difference from policy evaluation**: Uses \(\max\) instead of expectation over policy.

### Extracting the Policy

After convergence, extract optimal policy:

$$\pi^*(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]$$

---

## Math and Derivations

### Contraction Property

The Bellman optimality operator \(\mathcal{T}^*\) is also a contraction:

$$\|\mathcal{T}^* V_1 - \mathcal{T}^* V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$$

**Proof intuition**: The max doesn't break contraction because:

$$|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$$

### Convergence Rate

Same as policy evaluation:

$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$$

### Value Iteration as Policy Iteration

Value iteration can be seen as policy iteration with **one** sweep of policy evaluation:
- Start with \(\pi_k\) (implicit: greedy w.r.t. \(V_k\))
- Do one sweep of evaluation
- Immediately improve to \(\pi_{k+1}\)

This is why value iteration is called "truncated policy iteration."

### Q-Value Form

We can also iterate on Q-values directly:

$$Q_{k+1}(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q_k(s', a')$$

This is the basis for Q-learning (sample-based version).

---

## Algorithm Sketch

```
Algorithm: Value Iteration

Input: MDP (S, A, P, R, γ), threshold θ
Output: Optimal policy π*, approximately optimal V*

1. Initialize V(s) = 0 for all s
2. Repeat:
     Δ = 0
     For each s ∈ S:
         v = V(s)
         V(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
         Δ = max(Δ, |v - V(s)|)
   Until Δ < θ
3. Extract policy:
   For each s ∈ S:
     π(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
4. Return π, V
```

### Complexity

- **Per iteration**: \(O(|S|^2|A|)\)
- **Iterations to convergence**: \(O(\frac{1}{1-\gamma} \log \frac{1}{\theta})\)
- Typically more iterations than policy iteration, but each iteration is cheaper

---

## Common Pitfalls

1. **No explicit policy during iteration**: Don't try to track a policy — just track values. Extract policy at the end.

2. **Stopping too early**: Value iteration converges asymptotically. Need to check policy stability for true termination.

3. **Confusing with policy iteration**: Value iteration uses max (one equation). Policy iteration uses two steps (evaluate with expectation, then improve with max).

4. **Action space scaling**: Each state requires max over all actions. Large action spaces are expensive.

---

## Mini Example

**Simple 2-state MDP:**

States: {A, B}. From A: action 1 stays (r=2), action 2 goes to B (r=0). From B: only action stays (r=1). \(\gamma = 0.5\).

**Iteration 0**: \(V = [0, 0]\)

**Iteration 1**:
- \(V(A) = \max\{2 + 0.5 \times 0, \; 0 + 0.5 \times 0\} = 2\)
- \(V(B) = 1 + 0.5 \times 0 = 1\)
- \(V = [2, 1]\)

**Iteration 2**:
- \(V(A) = \max\{2 + 0.5 \times 2, \; 0 + 0.5 \times 1\} = \max\{3, 0.5\} = 3\)
- \(V(B) = 1 + 0.5 \times 1 = 1.5\)
- \(V = [3, 1.5]\)

**Iteration 3**:
- \(V(A) = \max\{2 + 0.5 \times 3, \; 0 + 0.5 \times 1.5\} = \max\{3.5, 0.75\} = 3.5\)
- \(V(B) = 1 + 0.5 \times 1.5 = 1.75\)

**Converges to**: \(V^*(A) = 4\), \(V^*(B) = 2\) (geometric series: \(2 + 2 \times 0.5 + 2 \times 0.25 + \ldots = 4\))

**Optimal policy**: Stay in A forever.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What's the key difference between value iteration and policy iteration?</summary>

**Answer**:
- **Value iteration**: One equation with max; no explicit policy; converges asymptotically
- **Policy iteration**: Two steps (evaluate, improve); explicit policy; converges in finite iterations

**Explanation**: Value iteration combines evaluation and improvement into one update by using max. Policy iteration separates them. Value iteration is simpler but converges slower. Policy iteration converges faster but each iteration is more expensive.

**Common pitfall**: Thinking value iteration is strictly better because it's simpler. For many problems, policy iteration converges in fewer total operations.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why does value iteration converge more slowly than policy iteration?</summary>

**Answer**: Value iteration does only one sweep of evaluation before improving, while policy iteration does full evaluation. The extra evaluation helps each improvement step be more accurate.

**Explanation**: Think of it as exploration vs exploitation in optimization. Policy iteration fully exploits current value estimates before changing direction. Value iteration is more greedy — changing direction every sweep.

**Key insight**: Modified policy iteration (few evaluation sweeps) often hits a sweet spot.

**Common pitfall**: Assuming more iterations means more computation. Each value iteration step is cheaper than a full policy evaluation.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Prove the Bellman optimality operator is a contraction.</summary>

**Answer**: For any two value functions \(V_1, V_2\):

$$|(\mathcal{T}^* V_1)(s) - (\mathcal{T}^* V_2)(s)|$$

$$= |\max_a [R + \gamma \sum_{s'} P(s'|s,a) V_1(s')] - \max_a [R + \gamma \sum_{s'} P(s'|s,a) V_2(s')]|$$

$$\leq \max_a |\gamma \sum_{s'} P(s'|s,a) (V_1(s') - V_2(s'))|$$

$$\leq \gamma \max_a \sum_{s'} P(s'|s,a) |V_1(s') - V_2(s')|$$

$$\leq \gamma \|V_1 - V_2\|_\infty$$

**Key step**: \(|\max f - \max g| \leq \max |f - g|\).

**Common pitfall**: The max doesn't break contraction — this is often confusing intuitively.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> When is value iteration preferred over policy iteration?</summary>

**Answer**: Value iteration is preferred when:
1. Action space is small (max is cheap)
2. We only need approximate values (can stop early)
3. Implementation simplicity matters
4. State space is large relative to action space

**Explanation**: Policy iteration's advantage is faster convergence (fewer outer iterations). But each iteration requires solving policy evaluation to convergence. If evaluation is expensive (large state space) and we're okay with approximation, value iteration wins.

**Common pitfall**: Assuming one is always better. The choice depends on the problem structure.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Value iteration is slow for your MDP. How do you speed it up?</summary>

**Answer**: Strategies:
1. **Prioritized sweeping**: Update states with largest Bellman error first
2. **Gauss-Seidel (in-place)**: Use updated values immediately
3. **Asynchronous updates**: Focus on relevant states
4. **Reduce \(\gamma\)**: Faster convergence (if acceptable)
5. **Early termination**: Extract policy before full convergence
6. **Use policy iteration**: Might be faster for this problem

**Explanation**: Prioritized sweeping can reduce iterations by 10-100x. The key insight: updates propagate from goal states backward. Prioritize states whose successors just changed.

**Common pitfall**: Not considering switching to policy iteration. Sometimes the "simpler" algorithm is actually slower.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 4.4
- **Bellman (1957)**, Dynamic Programming
- **Bertsekas**, Dynamic Programming and Optimal Control, Volume 1

**What to memorize for interviews**: Update rule with max, contraction property (same \(\gamma^k\) rate), difference from policy iteration, when to prefer each.

**Code example**: [value_iteration.py](../../../rl_examples/algorithms/value_iteration.py)
