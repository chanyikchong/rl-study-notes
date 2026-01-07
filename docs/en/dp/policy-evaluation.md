# Policy Evaluation

## Interview Summary

**Policy evaluation** computes \(V^\pi(s)\) for a given policy \(\pi\). It iteratively applies the Bellman expectation equation until convergence. This is the "E" step in policy iteration. Complexity is \(O(|S|^2|A|)\) per iteration. Key insight: we're solving a system of linear equations via fixed-point iteration.

**What to memorize**: Iterative update rule, convergence guarantee for \(\gamma < 1\), relationship to Bellman expectation equation.

---

## Core Definitions

### Problem Statement

**Given**: MDP \((S, A, P, R, \gamma)\) and policy \(\pi\)

**Find**: \(V^\pi(s)\) for all \(s \in S\)

### Iterative Update Rule

$$V_{k+1}(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$

**Meaning**: Update each state's value using current estimates of successor values.

### Convergence Criterion

Stop when:

$$\max_s |V_{k+1}(s) - V_k(s)| < \theta$$

where \(\theta\) is a small threshold (e.g., \(10^{-4}\)).

---

## Math and Derivations

### Why Iteration Works: Contraction Mapping

The Bellman operator \(\mathcal{T}^\pi\) is a **contraction**:

$$\|\mathcal{T}^\pi V_1 - \mathcal{T}^\pi V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$$

**Meaning**: Applying the Bellman update brings any two value functions closer together by factor \(\gamma\).

**Consequence**: By Banach fixed-point theorem, iteration converges to unique fixed point \(V^\pi\).

### Convergence Rate

After \(k\) iterations:

$$\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$$

**Example**: With \(\gamma = 0.9\), after 100 iterations, error is reduced by factor \(0.9^{100} \approx 0.00003\).

### Matrix Form Solution

The Bellman equation is linear:

$$\mathbf{V}^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi$$

Direct solution:

$$\mathbf{V}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{R}^\pi$$

**Complexity**: \(O(|S|^3)\) for matrix inversion — only practical for small state spaces.

---

## Algorithm Sketch

```
Algorithm: Iterative Policy Evaluation

Input: MDP, policy π, threshold θ
Output: V^π

1. Initialize V(s) = 0 for all s (or arbitrarily)
2. Repeat:
     Δ = 0
     For each s ∈ S:
         v = V(s)
         V(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
         Δ = max(Δ, |v - V(s)|)
   Until Δ < θ
3. Return V
```

**In-place vs Two-array**: In-place updates (overwriting \(V(s)\) immediately) often converge faster due to using fresher estimates. Sutton & Barto uses in-place.

### Complexity

- **Per iteration**: \(O(|S|^2 |A|)\) — for each state, sum over actions and next states
- **Number of iterations**: \(O(\frac{1}{1-\gamma} \log \frac{1}{\theta})\) approximately

---

## Common Pitfalls

1. **Forgetting to handle terminal states**: Terminal states should have \(V(s_{terminal}) = 0\) (no future rewards).

2. **Wrong update order**: For in-place updates, the order matters for speed but not correctness. Random or prioritized order can help.

3. **Not checking convergence properly**: Check max absolute change, not sum or average.

4. **Initializing values poorly**: Zero initialization is safe. Random initialization can be faster but may cause issues if values are extreme.

---

## Mini Example

**3-state chain MDP:**

```
S0 ---(a, r=0)---> S1 ---(a, r=0)---> S2 (terminal, r=1)
```

Policy: Only one action available (deterministic).
\(\gamma = 0.9\)

**Iteration:**
- \(V_0 = [0, 0, 0]\)
- \(V_1\): \(V(S2) = 0\), \(V(S1) = 0 + 0.9 \times 0 = 0\), \(V(S0) = 0 + 0.9 \times 0 = 0\)

Wait — we get reward 1 when reaching terminal! Let's fix:
- \(V(S1) = 0 + 0.9 \times 1 = 0.9\) (reward is received on transition to S2)
- Actually, reward on last step: \(V(S1) = R(S1 \to S2) + \gamma \cdot 0 = 1\)
- \(V(S0) = R(S0 \to S1) + \gamma V(S1) = 0 + 0.9 \times 1 = 0.9\)

**Final**: \(V^\pi = [0.9, 1, 0]\)

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why does policy evaluation always converge?</summary>

**Answer**: The Bellman operator is a contraction mapping with factor \(\gamma < 1\), so by the Banach fixed-point theorem, iteration converges to a unique fixed point.

**Explanation**: Each application of the Bellman update brings any value estimate closer to the true value. The contraction property \(\|\mathcal{T}^\pi V_1 - \mathcal{T}^\pi V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty\) guarantees this.

**Key equation**: \(\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty\)

**Common pitfall**: Convergence only guaranteed for \(\gamma < 1\). With \(\gamma = 1\), the system may not contract.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> What's the difference between in-place and two-array updates?</summary>

**Answer**:
- **Two-array**: Store \(V_k\) and \(V_{k+1}\) separately; use old values for all updates
- **In-place**: Update \(V(s)\) immediately; later states use updated values

**Explanation**: Both converge to the same answer. In-place often converges faster because it uses fresher estimates. The order of state updates affects speed (not correctness) for in-place.

**Common pitfall**: Thinking in-place is incorrect. It's actually the standard approach in Sutton & Barto.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Write the matrix form of the Bellman expectation equation and its solution.</summary>

**Answer**:

$$\mathbf{V}^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi$$

$$\mathbf{V}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{R}^\pi$$

**Explanation**:
- \(\mathbf{V}^\pi\) is vector of state values
- \(\mathbf{R}^\pi\) is vector of expected immediate rewards under \(\pi\)
- \(\mathbf{P}^\pi\) is transition matrix under \(\pi\): \(P^\pi_{ss'} = \sum_a \pi(a|s) P(s'|s,a)\)
- Rearranging: \((\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{V}^\pi = \mathbf{R}^\pi\)

**Common pitfall**: Matrix inversion is \(O(|S|^3)\) — impractical for large state spaces. Use iteration instead.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> How many iterations are needed to reduce error by factor of 1000 with γ=0.99?</summary>

**Answer**: Approximately 690 iterations.

**Explanation**: We need \(\gamma^k < 0.001\), so \(k > \frac{\log(0.001)}{\log(0.99)} = \frac{-6.9}{-0.01} \approx 690\).

**Key insight**: Larger \(\gamma\) means slower convergence. This is intuitive: far-sighted agents need more iterations to propagate value information.

**Common pitfall**: Underestimating iterations needed for high \(\gamma\). In practice, use \(\gamma = 0.99\) but accept approximate convergence.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your policy evaluation is running very slowly. How do you speed it up?</summary>

**Answer**: Strategies:
1. **Prioritized sweeping**: Update states with largest Bellman error first
2. **Asynchronous updates**: Don't sweep through all states each iteration
3. **Lower precision threshold**: Accept \(\theta = 10^{-2}\) instead of \(10^{-6}\)
4. **Lower \(\gamma\)**: Faster convergence (if acceptable for the task)
5. **Gauss-Seidel** (in-place): Often faster than Jacobi (two-array)

**Explanation**: The bottleneck is usually the number of iterations. Prioritized sweeping can dramatically reduce this by focusing on states where values are changing most.

**Common pitfall**: Using too tight a convergence threshold. For policy iteration, we often don't need exact \(V^\pi\) — approximate evaluation suffices.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 4.1
- **Bertsekas**, Dynamic Programming and Optimal Control, Vol 2
- **Puterman**, Markov Decision Processes, Chapter 6

**What to memorize for interviews**: Update rule, contraction property, convergence rate \(\gamma^k\), matrix form (but note impracticality), in-place vs two-array.

**Code example**: [policy_evaluation.py](../../../rl_examples/algorithms/policy_evaluation.py)
