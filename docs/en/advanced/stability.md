# Stability Issues in Deep RL

## Interview Summary

Deep RL can be unstable: Q-values diverge, policies collapse, or performance oscillates. The **deadly triad** (function approximation + bootstrapping + off-policy) explains many failures. Key stability techniques: **target networks** (frozen Q for targets), **experience replay** (break correlation), **reward clipping**, **gradient clipping**. Understanding when and why things break is crucial for debugging.

**What to memorize**: Deadly triad components, why each causes problems, common stabilization techniques.

---

## Core Definitions

### The Deadly Triad

Three elements that together can cause divergence:

1. **Function Approximation**: Generalization across states
2. **Bootstrapping**: Using value estimates as targets (TD learning)
3. **Off-policy Learning**: Learning about one policy from another's data

**Key insight**: Each element alone is fine. It's the combination that's deadly.

### Divergence Types

1. **Value divergence**: Q(s,a) → ±∞
2. **Policy collapse**: π becomes deterministic on bad actions
3. **Catastrophic forgetting**: Agent forgets how to solve earlier parts of task

### Overestimation Bias

$$\mathbb{E}[\max_a Q(s,a)] \geq \max_a \mathbb{E}[Q(s,a)]$$

With noise in Q-estimates, max picks overestimated values. This accumulates through bootstrapping.

---

## Math and Derivations

### Why the Triad Causes Problems

**Function approximation + Bootstrapping**:
- Update state \(s\) based on \(V(s')\)
- But updating parameters affects \(V(s')\) too (generalization)
- Can create positive feedback loops: \(V(s') \uparrow \Rightarrow V(s) \uparrow \Rightarrow V(s') \uparrow\)

**Off-policy + Bootstrapping**:
- Target \(V(s')\) is for a different policy than data
- Updates may not converge to anything meaningful

**Off-policy + Function Approximation**:
- Data distribution doesn't match state distribution of target policy
- Some states over-represented, others under-represented
- Generalization amplifies errors in under-represented regions

### Counterexample: Baird's Star

Classic example of divergence with linear FA and TD:
```
     *
    /|\
   / | \
  *  *  *  (5 outer states)
     |
     *     (1 central state)
```

With off-policy sampling, Q-values diverge to infinity despite bounded rewards.

### Target Network Analysis

Without target network:

$$\theta \leftarrow \theta + \alpha (r + \gamma \max_a Q(s',a;\theta) - Q(s,a;\theta)) \nabla_\theta Q(s,a;\theta)$$

Target changes at every step → chasing moving goal.

With target network \(\theta^-\):

$$\theta \leftarrow \theta + \alpha (r + \gamma \max_a Q(s',a;\theta^-) - Q(s,a;\theta)) \nabla_\theta Q(s,a;\theta)$$

Target is fixed for C steps → stable regression target.

---

## Algorithm Sketch

### Stability Techniques Summary

| Technique | What It Does | Helps With |
|-----------|--------------|------------|
| Target network | Freeze target Q | Bootstrapping stability |
| Experience replay | Random sampling | Break correlation |
| Double DQN | Separate select/evaluate | Overestimation |
| Dueling architecture | Separate V and A | Value estimation |
| Prioritized replay | Focus on high-error | Sample efficiency |
| Reward clipping | Bound rewards to [-1,1] | Gradient magnitude |
| Gradient clipping | Bound gradient norm | Exploding gradients |
| Layer normalization | Normalize activations | Training stability |

### Diagnosing Instability

```
Debugging Checklist:

1. Q-values exploding?
   → Check reward scale, add reward clipping
   → Reduce learning rate
   → Add target network (if not using)

2. Q-values all similar?
   → May be underfitting; increase capacity
   → Check if environment gives varied rewards

3. Policy entropy collapsing?
   → Add entropy bonus
   → Check temperature in softmax

4. Performance oscillating?
   → Reduce learning rate
   → Increase target update period
   → Use larger replay buffer

5. Sudden performance drop?
   → Catastrophic forgetting; larger replay buffer
   → Check for bugs in done/terminal handling
```

---

## Common Pitfalls

1. **No target network**: Essential for DQN stability. Update every 1K-10K steps.

2. **Replay buffer too small**: Need diversity. Use 100K-1M transitions.

3. **Learning rate too high**: Deep RL is sensitive. Start at 1e-4.

4. **Reward scale too large**: Causes large gradients. Clip or normalize rewards.

5. **Not clipping gradients**: Large gradients → large parameter changes → instability.

6. **Using on-policy methods as off-policy**: Policy gradient methods need fresh data.

---

## Mini Example

**Divergence Demonstration:**

Simple 2-state MDP with linear FA:
- States: s₁, s₂
- Features: φ(s₁) = [1, 0], φ(s₂) = [0, 1]
- Weights: w = [w₁, w₂]
- Transition: s₁ → s₂ → s₁ (deterministic cycle)
- Reward: r = 0 always
- γ = 0.99

With off-policy sampling (always start from s₁):
```
w₁ updated based on V(s₂) = w₂
w₂ updated based on V(s₁) = w₁
```

If w₁ slightly overestimated:
- w₂ increases to match
- w₁ increases further
- Positive feedback → divergence

**Solution**: On-policy sampling or target networks.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Explain the deadly triad and why each component contributes to instability.</summary>

**Answer**:

1. **Function approximation**: Changes to weights affect many states. An update to state s changes values everywhere, potentially in harmful ways.

2. **Bootstrapping**: Targets depend on current estimates. If estimates are wrong, we regress toward wrong values. Errors compound.

3. **Off-policy**: Data comes from a different policy than we're learning about. Some state-action pairs are underrepresented, leading to poor estimates that generalization amplifies.

**Together**: Errors propagate (bootstrapping), affect unrelated states (FA), and can't be corrected by data distribution (off-policy).

**Common pitfall**: Thinking any single element is the problem. It's the combination.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> How do target networks prevent instability?</summary>

**Answer**: Target networks freeze the TD target for C steps, making learning like supervised regression with fixed labels.

**Explanation**: Without target networks:
- Target: r + γ max Q(s', a'; θ)
- Every gradient step changes θ, changing the target
- Like shooting at a moving goal

With target networks:
- Target: r + γ max Q(s', a'; θ⁻)
- θ⁻ is fixed for C steps
- We're regressing to stable values

**Common pitfall**: Updating target too frequently. C should be large (1000-10000 steps).
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Explain overestimation bias mathematically.</summary>

**Answer**: For random variables X₁, ..., Xₙ:

$$\mathbb{E}[\max_i X_i] \geq \max_i \mathbb{E}[X_i]$$

**Application to Q-learning**: Q(s,a) has estimation noise. When we compute max_a Q(s,a), we tend to select the action with upward noise. This selection bias means we overestimate.

**Bootstrapping compounds it**: Overestimated Q(s') → overestimated target → overestimated Q(s) → ...

**Solution**: Double DQN decouples selection (which action) from evaluation (what's its value):

$$y = r + \gamma Q(s', \arg\max_a Q(s',a;\theta); \theta^-)$$

**Common pitfall**: Ignoring this in practice. Always use Double DQN.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Why does off-policy learning need importance sampling in theory?</summary>

**Answer**: We want to estimate:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s,a)]$$

But our data has actions from behavior policy b:

$$\mathbb{E}_{a \sim b}[Q(s,a)] \neq V^\pi(s)$$

Correction via importance sampling:

$$\mathbb{E}_{a \sim b}\left[\frac{\pi(a|s)}{b(a|s)} Q(s,a)\right] = V^\pi(s)$$

**Q-learning's trick**: Use max instead of expectation over π. This sidesteps importance sampling but still has issues (deadly triad).

**Common pitfall**: Q-learning doesn't need importance weights for convergence in tabular case, but issues arise with FA.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your DQN training shows Q-values increasing without bound. How to fix?</summary>

**Answer**: Step-by-step debugging:

1. **Check rewards**: Are they bounded? Clip to [-1, 1]
2. **Target network**: Is it being used? Update every 10K steps, not every step
3. **Learning rate**: Reduce to 1e-5 to 1e-4
4. **Double DQN**: Implement to reduce overestimation
5. **Gradient clipping**: Clip gradients to norm 10
6. **Network architecture**: Reduce size if overfitting
7. **Terminal states**: Make sure V(terminal) = 0

**Explanation**: Diverging Q-values usually mean bootstrapping errors are compounding. Target networks and Double DQN are the most important fixes.

**Common pitfall**: Trying to fix with more data when the algorithm is fundamentally unstable.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 11
- **Tsitsiklis & Van Roy (1997)**, An Analysis of TD Learning with Function Approximation
- **Van Hasselt et al. (2018)**, Deep RL and the Deadly Triad
- **Baird (1995)**, Residual Algorithms (famous counterexample)

**What to memorize for interviews**: Deadly triad components, target network purpose, overestimation bias, Double DQN, common stability techniques.
