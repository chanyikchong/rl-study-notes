# Expected SARSA

## Interview Summary

**Expected SARSA** uses the expected Q-value over all possible next actions, weighted by policy probabilities: \(\sum_{a'} \pi(a'|s') Q(s', a')\). This reduces variance compared to SARSA (which samples one action) while maintaining on-policy semantics. With a greedy target policy, Expected SARSA becomes Q-learning. It's a flexible middle ground.

**What to memorize**: Update rule with expectation, variance reduction vs SARSA, relationship to Q-learning (greedy case), on-policy but lower variance.

---

## Core Definitions

### Expected SARSA Update Rule

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a'} \pi(a'|S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

**Key insight**: Instead of sampling \(A'\) and using \(Q(S', A')\), we compute the expectation over all actions.

### Relationship to Other Algorithms

| Algorithm | Target uses |
|-----------|-------------|
| SARSA | \(Q(S', A')\) where \(A' \sim \pi\) |
| Expected SARSA | \(\sum_{a'} \pi(a' \mid S') Q(S', a')\) |
| Q-learning | \(\max_{a'} Q(S', a')\) |

**Observation**: Q-learning is Expected SARSA with a greedy target policy.

---

## Math and Derivations

### Variance Reduction

SARSA uses a single sample:

$$\text{SARSA target} = R + \gamma Q(S', A')$$

Expected SARSA uses the true expectation:

$$\text{Expected SARSA target} = R + \gamma \sum_{a'} \pi(a'|S') Q(S', a')$$

**Variance comparison**:
- SARSA has variance from sampling \(A'\)
- Expected SARSA eliminates this variance (only transition variance remains)

### Bellman Equation Relationship

Expected SARSA updates toward:

$$Q^\pi(s, a) = \mathbb{E}[R_{t+1}] + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s', a')$$

This is exactly the Bellman expectation equation for \(Q^\pi\).

### Convergence

Expected SARSA converges to \(Q^\pi\) (the Q-function for policy \(\pi\)). If \(\pi\) is ε-greedy and ε → 0, it converges to \(Q^*\).

---

## Algorithm Sketch

```
Algorithm: Expected SARSA

Input: α, γ, ε
Output: Q ≈ Q^π

1. Initialize Q(s,a) arbitrarily (Q(terminal, ·) = 0)
2. For each episode:
     S = initial state
     While S is not terminal:
         A = ε-greedy action from Q(S, ·)
         Take action A, observe R, S'
         expected_q = Σ_a' π(a'|S') Q(S', a')
         Q(S,A) ← Q(S,A) + α[R + γ · expected_q - Q(S,A)]
         S ← S'
3. Return Q
```

### Computing the Expectation

For ε-greedy policy:

$$\sum_{a'} \pi(a'|s') Q(s', a') = (1-\epsilon) \max_{a'} Q(s', a') + \frac{\epsilon}{|A|} \sum_{a'} Q(s', a')$$

**Interpretation**: Weight greedy action by \((1-\epsilon)\), distribute \(\epsilon\) uniformly.

---

## Common Pitfalls

1. **Overhead for large action spaces**: Need to sum over all actions. SARSA only samples one.

2. **Confusing with Q-learning**: Expected SARSA uses \(\pi\) (possibly exploratory); Q-learning uses greedy.

3. **On-policy semantics**: Despite computing an expectation, it's still on-policy (learns \(Q^\pi\), not \(Q^*\)).

4. **Policy must be known**: Unlike SARSA, you need to know action probabilities to compute the expectation.

---

## Mini Example

**2-action bandit setting** (single state):

Actions: A, B. \(Q(A) = 10\), \(Q(B) = 5\). Policy: ε-greedy with ε = 0.2.

**SARSA**: Sample \(A'\). 80% chance pick A (gives \(Q = 10\)), 20% chance pick B (gives \(Q = 5\)).

**Expected SARSA**:

$$\mathbb{E}[Q] = 0.9 \times 10 + 0.1 \times 5 + 0.1 \times 10 + 0.9 \times 5 \, ?$$

Wait, let me recalculate:
- ε-greedy: prob(A) = 0.8 + 0.1 = 0.9? No.
- With ε = 0.2: prob(greedy) = 0.8, prob(random each) = 0.1
- If A is greedy: prob(A) = 0.8 + 0.2/2 = 0.9, prob(B) = 0.2/2 = 0.1

$$\mathbb{E}[Q] = 0.9 \times 10 + 0.1 \times 5 = 9.5$$

**Key point**: Expected SARSA always gets 9.5; SARSA sometimes gets 10, sometimes 5, but averages to 9.5.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> How does Expected SARSA reduce variance compared to SARSA?</summary>

**Answer**: SARSA samples one action \(A'\) and uses \(Q(S', A')\). Expected SARSA computes the exact expectation \(\sum_{a'} \pi(a'|S') Q(S', a')\), eliminating sampling variance.

**Explanation**: The randomness in \(A'\) contributes variance to SARSA. By averaging over all possible \(A'\), Expected SARSA removes this noise source.

**Key insight**: Only transition variance remains (randomness in \(S'\)).

**Common pitfall**: Thinking less variance is always better. In large action spaces, computing the expectation is expensive.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> When is Expected SARSA equivalent to Q-learning?</summary>

**Answer**: When the target policy \(\pi\) is greedy with respect to Q.

**Explanation**: If \(\pi(a'|s') = 1\) for \(a' = \arg\max Q(s', a')\) and 0 otherwise:

$$\sum_{a'} \pi(a'|s') Q(s', a') = \max_{a'} Q(s', a')$$

This is exactly Q-learning's target.

**Key insight**: Q-learning is a special case of Expected SARSA with greedy target policy.

**Common pitfall**: Thinking Q-learning and Expected SARSA are fundamentally different algorithms.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Derive the Expected SARSA target for an ε-greedy policy.</summary>

**Answer**: With ε-greedy, let \(a^* = \arg\max_{a'} Q(s', a')\):

$$\sum_{a'} \pi(a'|s') Q(s', a') = (1-\epsilon) Q(s', a^*) + \epsilon \cdot \frac{1}{|A|} \sum_{a'} Q(s', a')$$

**Simplification**: Let \(\bar{Q}(s') = \frac{1}{|A|} \sum_{a'} Q(s', a')\) (average Q):

$$= (1-\epsilon) \max_{a'} Q(s', a') + \epsilon \cdot \bar{Q}(s')$$

**Explanation**: Greedy action is chosen with prob \((1-\epsilon)\), random action with prob \(\epsilon\).

**Common pitfall**: Forgetting that the random part has probability \(\epsilon/|A|\) per action, not \(\epsilon\).
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> What is the computational cost of Expected SARSA vs SARSA?</summary>

**Answer**:
- **SARSA**: O(1) for target computation (just look up one Q value)
- **Expected SARSA**: O(|A|) for target computation (sum over all actions)

**Explanation**: For each update, Expected SARSA must iterate through all actions. In large action spaces (e.g., continuous), this is infeasible without approximations.

**Trade-off**: Lower variance but higher computation per update.

**Common pitfall**: Using Expected SARSA with very large action spaces without considering cost.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> When should you prefer Expected SARSA over SARSA or Q-learning?</summary>

**Answer**: Prefer Expected SARSA when:
1. Action space is small (expectation is cheap)
2. You want on-policy learning with lower variance
3. You're using function approximation (stability matters)
4. The policy is known/controllable

Prefer SARSA when: action space is large, on-policy is fine
Prefer Q-learning when: you want off-policy learning of optimal Q

**Explanation**: Expected SARSA is a variance-reduced version of SARSA. It's a good default when you can afford the computation.

**Common pitfall**: Defaulting to Q-learning when Expected SARSA might be more stable.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 6.6
- **Van Seijen et al. (2009)**, A Theoretical and Empirical Analysis of Expected Sarsa

**What to memorize for interviews**: Update rule with expectation, variance reduction mechanism, relationship to Q-learning, ε-greedy expectation formula.

**Code example**: [expected_sarsa.py](../../../rl_examples/algorithms/expected_sarsa.py)
