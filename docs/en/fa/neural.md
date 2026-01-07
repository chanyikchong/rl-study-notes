# Neural Network Function Approximation

## Interview Summary

**Neural network function approximation** uses deep networks to represent value functions: \(\hat{V}(s; \boldsymbol{\theta})\) or \(\hat{Q}(s, a; \boldsymbol{\theta})\). Key advantage: learns features automatically (no hand-engineering). Key challenge: no convergence guarantees, instability (deadly triad). Modern deep RL (DQN, A3C, PPO) all use neural FA. Understanding the stability issues is crucial for interviews.

**What to memorize**: Neural TD update, deadly triad, DQN tricks (replay, target network), why instability happens.

---

## Core Definitions

### Neural Value Function

$$\hat{V}(s; \boldsymbol{\theta}) = f_\theta(s)$$

where \(f_\theta\) is a neural network with parameters \(\boldsymbol{\theta}\).

### Neural Q-Function

$$\hat{Q}(s, a; \boldsymbol{\theta}) = f_\theta(s, a) \quad \text{or} \quad \hat{Q}(s, \cdot; \boldsymbol{\theta}) = f_\theta(s)$$

The second form outputs Q-values for all actions (more efficient for discrete actions).

### Gradient

$$\nabla_\theta \hat{V}(s; \boldsymbol{\theta}) = \nabla_\theta f_\theta(s)$$

Computed via backpropagation. Unlike linear FA, gradient is a complex function of \(\boldsymbol{\theta}\).

---

## Math and Derivations

### Semi-gradient TD Update

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \delta_t \nabla_\theta \hat{V}(S_t; \boldsymbol{\theta})$$

where \(\delta_t = R_{t+1} + \gamma \hat{V}(S_{t+1}; \boldsymbol{\theta}) - \hat{V}(S_t; \boldsymbol{\theta})\).

**Same as linear FA**, but \(\nabla_\theta\) is now a neural network gradient.

### Why Neural + TD Can Diverge

The **deadly triad**:
1. **Function approximation**: Updates to one state affect others
2. **Bootstrapping**: Target depends on current estimates
3. **Off-policy**: Data distribution ≠ target distribution

Together, these can cause values to spiral to infinity.

### Loss Function View

DQN minimizes:

$$L(\boldsymbol{\theta}) = \mathbb{E}_{(s,a,r,s') \sim D}\left[ (y - \hat{Q}(s, a; \boldsymbol{\theta}))^2 \right]$$

where target \(y = r + \gamma \max_{a'} \hat{Q}(s', a'; \boldsymbol{\theta}^-)\).

**Note**: Target network \(\boldsymbol{\theta}^-\) is frozen — this stabilizes training.

---

## Algorithm Sketch

### Naive Neural TD (Unstable)

```
Algorithm: Neural TD (Don't use directly)

1. Initialize θ randomly
2. For each step:
     Observe (S, A, R, S')
     δ = R + γ V(S'; θ) - V(S; θ)
     θ ← θ + α · δ · ∇_θ V(S; θ)
```

**Problem**: Correlated updates, moving targets → divergence.

### DQN-style (Stable)

```
Algorithm: DQN (Stable neural Q-learning)

1. Initialize θ, θ^- = θ, replay buffer D
2. For each step:
     A = ε-greedy from Q(S, ·; θ)
     Execute A, observe R, S'
     Store (S, A, R, S') in D

     Sample minibatch {(s, a, r, s')} from D
     Compute targets: y = r + γ max_a' Q(s', a'; θ^-)
     Loss: L = Σ(y - Q(s, a; θ))²
     θ ← θ - α∇_θ L

     Every C steps: θ^- ← θ
```

**Key additions**: Replay buffer (breaks correlation), target network (stable targets).

---

## Common Pitfalls

1. **No replay buffer**: Sequential data is correlated → biased gradients → divergence.

2. **No target network**: Chasing moving targets → oscillation/divergence.

3. **Learning rate too high**: Neural networks need small learning rates (1e-4 to 1e-3).

4. **Reward scaling**: Large rewards → large gradients → instability. Clip or normalize rewards.

5. **Network too large/small**: Overparameterization can help but costs computation. Underparameterization limits representation.

6. **Ignoring double Q-learning**: Standard Q-learning overestimates → use Double DQN.

---

## Mini Example

**CartPole with Neural Q:**

Network: 4 inputs (state) → 128 hidden → 2 outputs (Q for left/right)

Training:
1. Collect experience with ε-greedy
2. Store in replay buffer (size 10000)
3. Sample batch of 32 transitions
4. Compute TD targets using target network
5. Gradient step on MSE loss
6. Update target network every 100 steps

**Result**: Solves CartPole in ~200 episodes.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What is the "deadly triad" and why does it cause problems?</summary>

**Answer**: The deadly triad consists of:
1. Function approximation
2. Bootstrapping
3. Off-policy learning

**Explanation**: Function approximation means updating one state affects others. Bootstrapping uses current estimates as targets (self-referential). Off-policy means data distribution doesn't match what we're learning about. Together: updates can amplify errors and spiral out of control.

**Key insight**: Each element alone is fine. TD without FA converges. FA with MC converges. It's the combination that's dangerous.

**Common pitfall**: Thinking any one element is the culprit. All three are needed for divergence.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why does experience replay help stabilize learning?</summary>

**Answer**: It breaks correlation between consecutive samples.

**Explanation**: Without replay, consecutive transitions are highly correlated (from same trajectory). This violates the i.i.d. assumption of SGD and leads to biased gradients. Replay shuffles experience, making minibatches more i.i.d.

**Additional benefit**: Data efficiency — each transition used multiple times.

**Common pitfall**: Thinking replay is just for efficiency. The decorrelation is more important.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Explain how target networks work and why they help.</summary>

**Answer**: Target network \(\theta^-\) is a periodic copy of main network \(\theta\). Targets use \(\theta^-\):

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

**Explanation**: Without target network, the target changes every gradient step (chasing a moving target). By freezing \(\theta^-\), targets are stable for C steps, allowing gradient descent to make progress.

**Key insight**: We're doing regression with fixed targets (like supervised learning), which is stable.

**Common pitfall**: Updating \(\theta^-\) too frequently negates the benefit.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> What is the overestimation bias in neural Q-learning?</summary>

**Answer**: The max operator applied to noisy Q-estimates tends to overestimate.

$$\mathbb{E}[\max_a Q(s, a)] \geq \max_a \mathbb{E}[Q(s, a)]$$

**Explanation**: Neural networks have estimation error. The max picks whichever action has highest estimated Q — which is often an overestimate. This accumulates through bootstrapping.

**Solution**: Double DQN: use one network to select action, another to evaluate:

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

**Common pitfall**: Ignoring this in practice. Double DQN is usually better.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your DQN isn't learning. What to debug?</summary>

**Answer**: Debugging checklist:
1. **Rewards visible**: Print rewards to verify environment works
2. **Exploration sufficient**: Is ε high enough? Decaying too fast?
3. **Learning rate**: Try 1e-4 to 1e-3
4. **Replay buffer size**: At least 10K, preferably 100K+
5. **Target update frequency**: Every 1000-10000 steps
6. **Network architecture**: Try simple networks first
7. **Gradient clipping**: Clip to [-1, 1] or use Huber loss
8. **Reward clipping/normalization**: Keep rewards in [-1, 1]

**Explanation**: DQN has many hyperparameters. Start with known-working settings from papers, then tune.

**Common pitfall**: Debugging by changing many things at once. Change one thing at a time.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 11
- **Mnih et al. (2015)**, Human-level control through deep RL (DQN)
- **Van Hasselt et al. (2016)**, Deep Reinforcement Learning with Double Q-learning
- **Tsitsiklis & Van Roy (1997)**, Analysis of TD(λ) with Function Approximation

**What to memorize for interviews**: Deadly triad, replay buffer purpose, target network purpose, Double DQN, overestimation bias.

**Code example**: [dqn.py](../../../rl_examples/algorithms/dqn.py)
