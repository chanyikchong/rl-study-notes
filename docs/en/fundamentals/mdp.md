# MDP Basics

## Interview Summary

A **Markov Decision Process (MDP)** is the mathematical framework for sequential decision-making. Key components: states \(S\), actions \(A\), transition dynamics \(P(s'|s,a)\), rewards \(R(s,a,s')\), and discount factor \(\gamma\). The Markov property states that the future depends only on the current state, not the history. MDPs are the foundation for all RL algorithms — understand this thoroughly.

**What to memorize**: MDP tuple \((S, A, P, R, \gamma)\), Markov property definition, return formula \(G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\).

---

## Core Definitions

### MDP Tuple

An MDP is defined by the tuple \((S, A, P, R, \gamma)\):

| Component | Symbol | Description |
|-----------|--------|-------------|
| State space | \(S\) | Set of all possible states |
| Action space | \(A\) | Set of all possible actions (can be state-dependent \(A(s)\)) |
| Transition function | \(P(s' \mid s,a)\) | Probability of reaching \(s'\) given state \(s\) and action \(a\) |
| Reward function | \(R(s,a,s')\) or \(R(s,a)\) | Immediate reward signal |
| Discount factor | \(\gamma \in [0,1]\) | How much to value future rewards |

### The Markov Property

$$P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0) = P(S_{t+1} | S_t, A_t)$$

**Meaning**: The probability of transitioning to any future state depends only on the current state and action, not on the history of how we got here.

### Trajectory and Episode

A **trajectory** (or episode) is a sequence:

$$\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \ldots)$$

An **episode** ends when a terminal state is reached.

---

## Math and Derivations

### Return (Cumulative Reward)

The **return** \(G_t\) is the total discounted reward from time \(t\):

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Why discount?**
1. Mathematical convenience: ensures \(G_t\) is finite for infinite horizons
2. Preference for immediate rewards (behavioral realism)
3. Models uncertainty about the future

**Recursive relationship** (important for TD learning):

$$G_t = R_{t+1} + \gamma G_{t+1}$$

### Special Cases of \(\gamma\)

| \(\gamma\) value | Behavior |
|------------------|----------|
| \(\gamma = 0\) | Myopic: only care about immediate reward |
| \(\gamma = 1\) | Undiscounted: all rewards equally important (only valid for episodic tasks) |
| \(\gamma \to 1\) | Far-sighted: care about long-term consequences |

### Expected Return

Since transitions are stochastic, we care about the **expected return**:

$$\mathbb{E}[G_t | S_t = s]$$

This expectation is taken over the randomness in transitions and policy.

---

## Algorithm Sketch

MDPs themselves don't have an "algorithm" — they're the problem formulation. The goal is to find a **policy** \(\pi\) that maximizes expected return. Methods include:

1. **Dynamic Programming** (requires known \(P\) and \(R\))
2. **Monte Carlo** (learn from complete episodes)
3. **Temporal Difference** (bootstrap from estimates)
4. **Deep RL** (use neural networks for large state spaces)

---

## Common Pitfalls

1. **Confusing \(R_t\) vs \(R_{t+1}\)**: Convention varies. Sutton & Barto use \(R_{t+1}\) for reward received after taking action \(A_t\) in state \(S_t\).

2. **Forgetting the Markov assumption**: If your state doesn't capture all relevant information, the Markov property is violated (partial observability → POMDP).

3. **Using \(\gamma = 1\) for continuing tasks**: Can lead to infinite returns. Only use for episodic tasks with guaranteed termination.

4. **State vs observation confusion**: In practice, we often have observations \(O\) that are functions of the true state. This matters for partial observability.

---

## Mini Example

**Gridworld MDP:**

```
+---+---+---+---+
| S |   |   | G |
+---+---+---+---+
|   | X |   |   |
+---+---+---+---+
```

- \(S\): 8 non-wall cells (states)
- \(A = \{\text{up}, \text{down}, \text{left}, \text{right}\}\)
- \(P\): Deterministic (or 0.8 intended direction, 0.1 each perpendicular)
- \(R\): -1 per step, +10 at goal G, -10 at X
- \(\gamma = 0.9\)

**Trajectory example**: S → right → right → right → G gives return: \(-1 + 0.9(-1) + 0.9^2(-1) + 0.9^3(10) = 4.39\)

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What is the Markov property and why is it important for RL?</summary>

**Answer**: The Markov property states that \(P(S_{t+1}|S_t, A_t) = P(S_{t+1}|S_t, A_t, S_{t-1}, \ldots, S_0)\) — the future is independent of the past given the present.

**Explanation**: This is crucial because it allows us to make decisions based only on the current state without needing to remember the entire history. This dramatically simplifies computation and enables the recursive Bellman equations.

**Key equation**: \(P(S_{t+1}|S_t, A_t)\)

**Common pitfall**: Designing states that don't capture all relevant information violates the Markov property. For example, in Atari games, a single frame doesn't capture velocity — that's why DQN uses frame stacking.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> What happens to agent behavior as γ approaches 0 vs 1?</summary>

**Answer**:
- \(\gamma \to 0\): Agent becomes myopic, only maximizing immediate reward
- \(\gamma \to 1\): Agent becomes far-sighted, valuing long-term consequences equally

**Explanation**: The discount factor \(\gamma\) controls the effective planning horizon. With \(\gamma = 0\), the return is just \(R_{t+1}\). With \(\gamma = 0.99\), rewards 100 steps away still contribute significantly (\(0.99^{100} \approx 0.37\)).

**Key equation**: \(G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\)

**Common pitfall**: Using \(\gamma = 1\) for non-episodic (continuing) tasks leads to infinite returns. Always use \(\gamma < 1\) for continuing tasks.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Prove that G_t = R_{t+1} + γG_{t+1}</summary>

**Answer**: Direct algebraic manipulation.

**Explanation**:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

$$= R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots)$$

$$= R_{t+1} + \gamma G_{t+1}$$

**Key equation**: \(G_t = R_{t+1} + \gamma G_{t+1}\)

**Common pitfall**: This recursive relationship is the foundation of TD learning. Forgetting this leads to confusion about how TD bootstraps from value estimates.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> If γ = 0.9 and rewards are +1 at every step forever, what is G_0?</summary>

**Answer**: \(G_0 = 10\)

**Explanation**: This is a geometric series:

$$G_0 = \sum_{k=0}^{\infty} \gamma^k \cdot 1 = \frac{1}{1-\gamma} = \frac{1}{1-0.9} = 10$$

**Key equation**: For constant reward \(r\): \(G_t = \frac{r}{1-\gamma}\)

**Common pitfall**: Forgetting this formula. It's useful for quick sanity checks and understanding the scale of value functions.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> How would you debug an RL agent that seems to be acting randomly?</summary>

**Answer**: Check these in order:
1. Verify reward signal is being received correctly (print rewards)
2. Check if state representation captures relevant information (Markov property)
3. Verify discount factor isn't too low (myopic behavior)
4. Check exploration rate (ε might be too high)
5. Ensure enough training steps have occurred

**Explanation**: Random behavior usually indicates the agent hasn't learned anything useful. The most common causes are reward bugs (not receiving signal), poor state representation (can't distinguish situations), or insufficient training.

**Common pitfall**: Starting debugging with the algorithm when the problem is often in the environment setup or reward function.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 3
- **Bellman (1957)**, Dynamic Programming — original formulation of MDPs
- **Puterman (1994)**, Markov Decision Processes — comprehensive mathematical treatment

**What to memorize for interviews**: MDP tuple definition, Markov property statement, return formula with discounting, recursive return relationship.
