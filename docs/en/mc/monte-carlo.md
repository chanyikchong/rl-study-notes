# Monte Carlo Methods

## Interview Summary

**Monte Carlo (MC) methods** learn value functions from complete episodes without needing a model. Key idea: average returns from visited states. Two variants: **first-visit MC** (only first occurrence per episode) and **every-visit MC** (all occurrences). MC has zero bias but high variance. Only works for episodic tasks. Forms the conceptual bridge between DP and TD learning.

**What to memorize**: MC update rule, first-visit vs every-visit, high variance/zero bias, exploring starts vs ε-soft policies.

---

## Core Definitions

### Monte Carlo Estimation

**Idea**: The value of a state is the expected return. Estimate it by averaging observed returns.

$$V(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s)$$

where \(G_i(s)\) is the return observed after visiting \(s\) in episode \(i\).

### First-Visit vs Every-Visit

- **First-visit MC**: Only count the first time \(s\) is visited in each episode
- **Every-visit MC**: Count every visit to \(s\)

Both converge to \(V^\pi(s)\) as \(N(s) \to \infty\). First-visit has slightly better theoretical properties (unbiased estimator).

### Incremental Mean Update

Instead of storing all returns:

$$V(s) \leftarrow V(s) + \alpha \left[ G - V(s) \right]$$

where \(\alpha = \frac{1}{N(s)}\) for exact averaging, or fixed \(\alpha\) for non-stationary environments.

---

## Math and Derivations

### Why MC is Unbiased

$$\mathbb{E}[G_t | S_t = s] = V^\pi(s)$$

By definition of \(V^\pi\). MC directly samples \(G_t\), so the average converges to the expectation.

### Variance of MC Estimates

$$\text{Var}(G_t) = \text{Var}\left( \sum_{k=0}^{T-t} \gamma^k R_{t+k+1} \right)$$

Variance grows with episode length. Long episodes → high variance → slow learning.

**Contrast with TD**: TD bootstraps from \(V(s')\), which reduces variance but introduces bias (if \(V\) is wrong).

### MC for Q-values

To learn Q-values (for control without a model):

$$Q(s,a) \approx \frac{1}{N(s,a)} \sum_{i=1}^{N(s,a)} G_i(s,a)$$

Need to visit all \((s,a)\) pairs → requires exploration.

---

## Algorithm Sketch

### MC Prediction (Policy Evaluation)

```
Algorithm: First-Visit MC Prediction

Input: Policy π, number of episodes
Output: V^π

1. Initialize V(s) arbitrarily, Returns(s) = [] for all s
2. For each episode:
     Generate episode: S_0, A_0, R_1, ..., S_T following π
     G = 0
     For t = T-1, T-2, ..., 0:
         G = γG + R_{t+1}
         If S_t not in {S_0, ..., S_{t-1}}:  # first visit check
             Append G to Returns(S_t)
             V(S_t) = mean(Returns(S_t))
3. Return V
```

### MC Control (Finding Optimal Policy)

```
Algorithm: MC Control with ε-soft Policy

Input: ε, number of episodes
Output: Approximately optimal π

1. Initialize Q(s,a) arbitrarily, π = ε-greedy w.r.t. Q
2. For each episode:
     Generate episode following π
     G = 0
     For t = T-1, T-2, ..., 0:
         G = γG + R_{t+1}
         Update Q(S_t, A_t) using G
     π = ε-greedy w.r.t. Q
3. Return π
```

### Exploring Starts

Alternative to ε-soft: start episodes from random (s,a) pairs. Ensures all pairs are visited. Impractical in real environments.

---

## Common Pitfalls

1. **Applying MC to continuing tasks**: MC requires complete episodes. Infinite tasks have no episode end.

2. **Poor exploration**: MC needs to visit all state-action pairs. Without exploration, some pairs never get updated.

3. **High variance with long episodes**: Returns accumulate noise from all steps. Consider using TD for long episodes.

4. **Forgetting to handle terminal states**: Returns should not include anything after terminal state.

5. **Using fixed α with first-visit**: If using incremental update, \(\alpha = 1/N(s)\) for exact average.

---

## Mini Example

**Simple episode:**
```
S0 → (R=0) → S1 → (R=0) → S2 → (R=1) → Terminal
```

With \(\gamma = 0.9\):
- \(G(S2) = 1\)
- \(G(S1) = 0 + 0.9 \times 1 = 0.9\)
- \(G(S0) = 0 + 0.9 \times 0.9 = 0.81\)

**After 1 episode**:
- \(V(S0) = 0.81\), \(V(S1) = 0.9\), \(V(S2) = 1\)

**After 100 episodes** (same trajectory each time):
- Values remain the same (low variance because trajectory is deterministic)

**With stochastic transitions**: Values converge to expectations over many episodes.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What is the key advantage of MC over DP?</summary>

**Answer**: MC doesn't require a model (transition probabilities \(P(s'|s,a)\)). It learns directly from experience.

**Explanation**: DP needs the complete model to compute expectations. MC samples trajectories from the environment, so it only needs to be able to interact (not know the dynamics).

**Key insight**: This makes MC applicable to real-world problems where models are unavailable.

**Common pitfall**: MC still needs complete episodes, which is its own limitation.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why does MC have high variance compared to TD?</summary>

**Answer**: MC uses the full return \(G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\), which sums many random variables. Each random reward adds to the variance.

**Explanation**: Variance of a sum grows with the number of terms. TD uses \(R_{t+1} + \gamma V(S_{t+1})\), which is just one random reward plus a learned estimate. The estimate \(V(S_{t+1})\) is biased but lower variance.

**Key equation**: \(\text{Var}(G) \gg \text{Var}(R + \gamma V)\) when episodes are long.

**Common pitfall**: High variance means more episodes needed for convergence, not that MC is wrong.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Derive the incremental update rule for MC.</summary>

**Answer**: Starting from the mean update:

After \(n\) visits, \(V_n = \frac{1}{n} \sum_{i=1}^{n} G_i\).

After \(n+1\) visits:

$$V_{n+1} = \frac{1}{n+1} \sum_{i=1}^{n+1} G_i = \frac{1}{n+1} \left( nV_n + G_{n+1} \right)$$

$$= V_n + \frac{1}{n+1}(G_{n+1} - V_n)$$

General form with learning rate \(\alpha\):

$$V(s) \leftarrow V(s) + \alpha [G - V(s)]$$

**Explanation**: The update moves \(V(s)\) toward \(G\) by fraction \(\alpha\). With \(\alpha = 1/n\), this computes exact average.

**Common pitfall**: Fixed \(\alpha\) doesn't compute the true average — it weights recent returns more heavily.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Why must we use ε-soft or exploring starts in MC control?</summary>

**Answer**: To ensure all state-action pairs are visited infinitely often, guaranteeing convergence.

**Explanation**: MC estimates Q(s,a) by averaging returns after (s,a). If we never visit (s,a), we can't estimate it. A deterministic policy might never visit some pairs → no learning for those.

**Key requirement**: GLIE (Greedy in the Limit with Infinite Exploration): explore all pairs infinitely, but become greedy eventually.

**Common pitfall**: Without exploration, MC control can converge to a suboptimal policy because it never discovers better actions.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> MC learning is very slow for your environment. What might help?</summary>

**Answer**: Potential improvements:
1. **Switch to TD**: Lower variance, learns during episodes
2. **Episode truncation**: Limit episode length, treat truncation as terminal
3. **Baseline subtraction**: Use advantage instead of raw returns
4. **Increase batch size**: Average over more episodes per update
5. **Reward shaping**: Add intermediate rewards to reduce horizon

**Explanation**: MC's sample complexity is \(O(\text{episode length}^2)\) due to variance. TD with bootstrapping is often orders of magnitude faster.

**Common pitfall**: Assuming the problem requires MC. In most cases, TD or n-step methods are preferred.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 5
- **Singh & Sutton (1996)**, Reinforcement Learning with Replacing Eligibility Traces
- **Kakade (2001)**, A Natural Policy Gradient

**What to memorize for interviews**: MC update rule, first-visit vs every-visit, unbiased but high variance, requires episodes, exploring starts or ε-soft.

**Code example**: [mc_control.py](../../rl_examples/algorithms/mc_control.py)
