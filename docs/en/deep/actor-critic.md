# Actor-Critic Methods

## Interview Summary

**Actor-Critic** methods combine policy gradients (actor) with value function learning (critic). The **actor** updates the policy using policy gradients, while the **critic** estimates value functions to reduce variance. This is the foundation for A2C, A3C, and PPO. Key insight: use TD error as a low-variance estimate of advantage, enabling per-step updates instead of waiting for episode end.

**What to memorize**: Actor-critic architecture, TD error as advantage, A2C update rules, A3C's asynchronous training, why critic reduces variance.

---

## Design Motivation: Why Actor-Critic Exists

### The Problem with Pure Policy Gradients

REINFORCE works but has a critical flaw:

$$\nabla J(\theta) = \mathbb{E}\left[ \nabla \log \pi_\theta(a|s) \cdot G_t \right]$$

**Problem 1: High variance**
- $G_t = r_t + r_{t+1} + r_{t+2} + ...$ is a sum of many random rewards
- Each reward adds noise → variance grows with episode length
- High variance means slow, unstable learning

**Problem 2: Must wait for episode end**
- Need full trajectory to compute $G_t$
- Can't learn during an episode
- Wasteful for long episodes

**Problem 3: Credit assignment is blurry**
- If episode gets reward 100 at the end, ALL actions get credit
- Hard to know which specific action was good

### The Actor-Critic Idea

> **Key Insight**: What if we use a learned value function to estimate returns BEFORE the episode ends?

```
REINFORCE:     G_t = r_t + r_{t+1} + r_{t+2} + ... + r_T    (wait for end)
Actor-Critic:  δ_t = r_t + γ V(s_{t+1}) - V(s_t)           (one step!)
```

The critic provides an estimate of future returns, so we don't need to wait.

### Evolution of Actor-Critic Methods

```
REINFORCE (high variance, episode-based)
     ↓ Add learned value function baseline
Actor-Critic (lower variance, per-step updates)
     ↓ Synchronous parallel workers + batching
A2C (Advantage Actor-Critic) (more stable)
     ↓ Asynchronous parallel workers
A3C (faster, more exploration)
     ↓ Trust region constraint
PPO (safe, large updates)
```

### Why This Architecture Works

| Component | What It Does | Benefit |
|-----------|-------------|---------|
| Actor $\pi_\theta(a\|s)$ | Decides actions | Can learn stochastic, continuous policies |
| Critic $V_\phi(s)$ | Evaluates states | Reduces variance by estimating future returns |
| TD Error $\delta_t$ | Advantage estimate | Low variance, unbiased in expectation |
| Entropy bonus | Encourages exploration | Prevents premature convergence |

**The magic**: Actor gets gradient signal from critic's evaluation. Critic learns from actual rewards. They bootstrap each other!

---

## Core Definitions

### Actor-Critic Architecture

**Actor**: Policy $\pi_\theta(a|s)$ — decides what actions to take
**Critic**: Value function $V_\phi(s)$ or $Q_\phi(s,a)$ — evaluates how good states/actions are

```
      State s
         ↓
   ┌─────┴─────┐
   ↓           ↓
 Actor       Critic
π_θ(a|s)    V_φ(s)
   ↓           ↓
 Action    Advantage
   a        estimate
   ↓           ↓
   └─────┬─────┘
         ↓
   Policy Gradient
   θ ← θ + α·δ·∇log π
```

### Why Combine Policy and Value Learning?

| Pure Policy Gradient | Pure Value-Based | Actor-Critic |
|---------------------|------------------|--------------|
| High variance | Lower variance | **Lower variance** |
| Stochastic policies | Deterministic only | **Stochastic** |
| Continuous actions | Discrete only | **Continuous** |
| On-policy | Can be off-policy | Mostly on-policy |
| Episode-based | Step-based | **Step-based** |

Actor-Critic gets the best of both worlds!

### TD Error as Advantage Estimate

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

**Why this is brilliant:**
- $r_t + \gamma V(s_{t+1})$ = one-step estimate of $Q(s_t, a_t)$
- $V(s_t)$ = expected value from state $s_t$
- Difference = how much better was action $a_t$ than average = Advantage!

---

## Math and Derivations

### Why TD Error Estimates Advantage (Important!)

This derivation is commonly asked in interviews.

**Claim**: $\mathbb{E}[\delta_t | s_t, a_t] = A^\pi(s_t, a_t)$

**Proof**:

$$\mathbb{E}[\delta_t | s_t, a_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) | s_t, a_t] - V^\pi(s_t)$$

By definition of $Q^\pi$:

$$\mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) | s_t, a_t] = Q^\pi(s_t, a_t)$$

Therefore:

$$\mathbb{E}[\delta_t | s_t, a_t] = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

**Key insight**: TD error is an **unbiased** estimate of advantage!

**Caveat**: This assumes $V^\pi$ is perfect. With learned $V_\phi$, we get biased estimates. But the variance reduction usually outweighs the bias.

### Actor Update (Policy Gradient with Critic)

$$\nabla_\theta J \approx \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s, a) \right]$$

where $\hat{A}$ is the advantage estimate:

| Method | Advantage Estimate | Properties |
|--------|-------------------|------------|
| Monte Carlo | $\hat{A} = G_t - V_\phi(s_t)$ | Unbiased, high variance |
| TD(0) | $\hat{A} = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ | Biased, low variance |
| n-step | $\hat{A} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)$ | Interpolates |
| GAE | Weighted average of n-step returns | Best of both (see GAE chapter) |

### Critic Update (TD Learning)

$$\phi \leftarrow \phi + \alpha_c \cdot \delta_t \cdot \nabla_\phi V_\phi(s_t)$$

Or with MSE loss:

$$L(\phi) = \mathbb{E}\left[ (V_\phi(s) - V^{target})^2 \right]$$

where $V^{target} = r + \gamma V_\phi(s')$ (TD) or $G_t$ (Monte Carlo).

### Variance Reduction Analysis

**Why does the critic reduce variance?**

REINFORCE variance:

$$\text{Var}(G_t) = \text{Var}\left(\sum_{k=t}^{T} \gamma^{k-t} r_k\right) \approx O(T - t)$$

Actor-Critic variance:

$$\text{Var}(\delta_t) = \text{Var}(r_t + \gamma V(s_{t+1}) - V(s_t)) \approx O(1)$$

**Intuition**: REINFORCE sums $T-t$ random variables. TD error uses only ONE random reward plus learned estimates. Huge variance reduction!

---

## Algorithm Sketch

### One-Step Actor-Critic (A2C Style)

```
Algorithm: Advantage Actor-Critic (A2C)

Parameters: θ (actor), φ (critic)
Hyperparameters: α_θ, α_φ, γ, β (entropy coefficient)

1. Initialize θ, φ
2. For each step:
     Observe state s
     Sample action a ~ π_θ(·|s)
     Execute a, observe r, s', done

     # Compute TD error (advantage estimate)
     If done:
         δ = r - V_φ(s)              # No future value at terminal
     Else:
         δ = r + γ V_φ(s') - V_φ(s)

     # Critic update (minimize TD error)
     φ ← φ + α_φ · δ · ∇_φ V_φ(s)

     # Actor update (policy gradient with advantage)
     θ ← θ + α_θ · δ · ∇_θ log π_θ(a|s)

     # Optional: entropy bonus for exploration
     θ ← θ + β · ∇_θ H(π_θ(·|s))

     s ← s'
```

### Batched A2C (More Stable)

```
Algorithm: Batched A2C

1. Collect n steps of experience: [(s_0, a_0, r_0), ..., (s_{n-1}, a_{n-1}, r_{n-1}), s_n]

2. Compute returns (backward):
   R_n = V_φ(s_n)  if not done, else 0
   For t = n-1 to 0:
       R_t = r_t + γ R_{t+1}
       A_t = R_t - V_φ(s_t)  # Advantage

3. Update critic:
   L_critic = (1/n) Σ (R_t - V_φ(s_t))²
   φ ← φ - α_φ ∇_φ L_critic

4. Update actor:
   L_actor = -(1/n) Σ log π_θ(a_t|s_t) · A_t - β H(π_θ)
   θ ← θ - α_θ ∇_θ L_actor
```

### A3C (Asynchronous Advantage Actor-Critic)

```
Algorithm: A3C (High-level)

Global parameters: θ_global, φ_global (shared across workers)

Each worker (in parallel):
1. Sync local parameters: θ ← θ_global, φ ← φ_global
2. Collect T steps of experience in local environment
3. Compute local gradients:
   ∇θ L_actor, ∇φ L_critic
4. Apply gradients to GLOBAL parameters (asynchronously):
   θ_global ← θ_global - α∇θ
   φ_global ← φ_global - α∇φ
5. Repeat from step 1
```

**Why asynchronous works:**
- Different workers explore different parts of state space
- Gradients are "stale" but still useful on average
- Much faster wall-clock time than sequential
- Natural exploration through diverse experiences

### Key Design Choices

| Choice | Options | Recommendation |
|--------|---------|----------------|
| Shared vs separate networks | Shared (actor/critic share layers) or separate | Separate for stability, shared for efficiency |
| Advantage estimation | TD(0), n-step, GAE | GAE for best performance |
| Critic target | TD ($r + \gamma V(s')$) or n-step returns | n-step with GAE |
| Entropy coefficient β | 0.01 to 0.1 | Start with 0.01, tune |
| Number of workers (A3C) | CPU cores | 16-32 typical |

---

## A2C vs A3C: When to Use Which?

| Aspect | A2C (Synchronous) | A3C (Asynchronous) |
|--------|-------------------|-------------------|
| Parallelism | Wait for all workers | Workers run independently |
| Gradient quality | Fresh, consistent | Stale, but diverse |
| GPU utilization | Good (batched) | Poor (many small updates) |
| Implementation | Simpler | More complex |
| Modern usage | **Preferred** | Less common now |

**Current best practice**: A2C with large batches + PPO clipping, or just use PPO.

---

## Common Pitfalls

1. **Critic learns too slowly**: If critic is bad, actor gets noisy gradients. Critic often needs higher learning rate or more training steps.

2. **Shared network interference**: Sharing layers between actor/critic can cause gradient interference. Solutions:
   - Separate networks entirely
   - Gradient scaling (lower weight on critic gradient)
   - Stop gradient at shared layers for one head

3. **Entropy collapse**: Policy becomes deterministic too fast. Add entropy bonus: $L = -\log \pi(a|s) \cdot A - \beta H(\pi)$.

4. **Forgetting to stop gradient**: Actor gradient should NOT flow through the advantage estimate. Use `.detach()` or `stop_gradient()`.

5. **Not handling terminal states**: $V(s_{terminal}) = 0$, don't bootstrap from it.

6. **Wrong learning rate ratio**: Actor and critic often need different learning rates. Typical: $\alpha_\phi = 3\alpha_\theta$ to $10\alpha_\theta$.

---

## Mini Example

**CartPole Actor-Critic:**

```python
# Networks (separate for stability)
actor = MLP(state_dim=4, hidden=[32], output_dim=2)   # policy logits
critic = MLP(state_dim=4, hidden=[32], output_dim=1)  # value
actor_opt = Adam(actor.parameters(), lr=1e-3)
critic_opt = Adam(critic.parameters(), lr=3e-3)  # Critic learns faster

gamma = 0.99
entropy_coef = 0.01

for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        # Forward pass
        logits = actor(state)
        probs = softmax(logits)
        value = critic(state)

        # Sample action from policy
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Environment step
        next_state, reward, done = env.step(action)

        # Compute TD error (advantage estimate)
        next_value = 0 if done else critic(next_state).detach()
        td_error = reward + gamma * next_value - value

        # Critic update (minimize squared TD error)
        critic_loss = td_error ** 2
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # Actor update (policy gradient with advantage)
        # Note: td_error.detach() - don't backprop through critic!
        advantage = td_error.detach()
        actor_loss = -log_prob * advantage - entropy_coef * entropy
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        state = next_state
```

**Expected**: Solves CartPole in ~200-400 episodes (slower than DQN but works for continuous actions).

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why use actor-critic instead of pure REINFORCE?</summary>

**Answer**: Lower variance from the critic, enabling faster and more stable learning.

**Explanation**: REINFORCE uses full episode returns $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$, which sum many random rewards and have high variance (grows with episode length). Actor-critic uses TD error $\delta_t = r + \gamma V(s') - V(s)$, which involves only ONE reward plus learned estimates.

**Key insight**: Variance reduction comes from replacing unknown future rewards with a learned estimate $V(s')$.

**Trade-off**: TD error introduces bias (if $V$ is wrong), but the variance reduction usually wins. This is the classic bias-variance tradeoff.

**Common pitfall**: Thinking actor-critic is always better. For very short episodes with dense rewards, REINFORCE might be competitive and simpler.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> What's the role of entropy regularization in actor-critic?</summary>

**Answer**: Prevents premature convergence to deterministic policies, encouraging continued exploration.

**Explanation**: Without entropy bonus, policy gradients naturally push toward deterministic policies — if action $a$ is slightly better than $b$, its probability increases, making $a$ even more likely, until $\pi(a|s) \approx 1$. Adding $\beta H(\pi) = -\beta \sum_a \pi(a|s) \log \pi(a|s)$ to the objective rewards keeping probability spread across actions.

**Why it matters:**
- Early training: Prevents locking into suboptimal actions before sufficient exploration
- Stochastic environments: Optimal policy might genuinely be stochastic
- Stability: Prevents overconfident updates

**Implementation**:

$$L_{actor} = -\log \pi(a|s) \cdot A - \beta H(\pi)$$

**Common pitfall**: Setting $\beta$ too high makes the policy too random and slow to converge. Too low allows entropy collapse. Typical range: 0.001 to 0.1.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Prove that E[δ_t | s_t, a_t] = A^π(s_t, a_t).</summary>

**Answer**:

**Step 1**: Expand TD error definition

$$\mathbb{E}[\delta_t | s_t, a_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) | s_t, a_t] - V^\pi(s_t)$$

**Step 2**: Recognize the first term as $Q^\pi$

$$\mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) | s_t, a_t]$$

$$= R(s_t, a_t) + \gamma \sum_{s'} P(s' \mid s_t, a_t) V^\pi(s')$$

$$= Q^\pi(s_t, a_t)$$

(This is exactly the Bellman equation for Q!)

**Step 3**: Substitute back

$$\mathbb{E}[\delta_t | s_t, a_t] = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

**Key insight**: TD error is an **unbiased** estimator of advantage in expectation. This is why we can use it in place of the true advantage.

**Common pitfall**: This derivation requires the **true** value function $V^\pi$. With a learned $V_\phi$, we get a biased estimate. The quality of your advantage estimates depends entirely on your critic's accuracy.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> How does A3C achieve exploration without explicit exploration bonus?</summary>

**Answer**: Parallel workers with different random seeds naturally explore different parts of state space, providing implicit exploration diversity.

**Detailed explanation**:
- Each of $N$ workers maintains its own environment instance
- Workers have different random seeds → different action sequences
- Workers are at different points in state space at any time
- Global network receives diverse gradients from all workers
- This diversity acts as implicit exploration

**Mathematical view**: If one worker is at state $s_1$ and another at $s_2$, the global network learns about both regions simultaneously. This is like having a "broader" replay buffer without actually storing transitions.

**Additional mechanisms typically used**:
1. Entropy bonus is still standard in A3C
2. Stochastic policy (sampling from $\pi$) provides within-worker exploration
3. Asynchronous updates prevent workers from converging to same behavior

**Why parallelism helps exploration**:
- Single worker: Explores one trajectory at a time
- N workers: Explores N different trajectories simultaneously
- More coverage of state space per wall-clock time

**Common pitfall**: Thinking parallelism is only for speed. It genuinely improves exploration and training stability through experience diversity.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your actor-critic isn't learning. Actor loss is oscillating wildly. What to debug?</summary>

**Answer**: Systematic debugging steps:

**1. Check the critic FIRST** (most common issue)
- If critic is inaccurate, advantage estimates are wrong → actor gets random gradients
- Monitor: Is critic loss decreasing? Is $V(s)$ in reasonable range?
- Fix: Train critic longer, use higher $\alpha_\phi$, or use n-step returns

**2. Check gradient flow**
- Is `td_error.detach()` used for actor update?
- Without detach: Actor gradient flows through critic → interference
- Fix: `advantage = td_error.detach()` in actor loss

**3. Reduce actor learning rate**
- High $\alpha_\theta$ → policy changes too fast → critic can't keep up → oscillation
- Typical ratio: $\alpha_\phi = 3-10 \times \alpha_\theta$

**4. Check advantage signs and magnitudes**
- Advantages should be centered around 0 (some positive, some negative)
- If all positive or all negative → problem with baseline
- If too large → gradient explosion → normalize advantages

**5. Add or increase entropy bonus**
- Might be stuck oscillating between local optima
- Entropy smooths the optimization landscape

**6. Use batching**
- Single-step updates are very noisy
- Try n-step (n=5) or batch updates for more stable gradients

**Debugging checklist**:
- [ ] Critic loss decreasing?
- [ ] Advantages centered around 0?
- [ ] `detach()` on advantage for actor?
- [ ] $\alpha_\phi > \alpha_\theta$?
- [ ] Entropy coefficient reasonable?

**Common pitfall**: Blaming the actor when the real problem is the critic. A bad critic makes actor training impossible.
</details>

<details markdown="1">
<summary><strong>Q6 (Conceptual):</strong> Should actor and critic share network layers? What are the tradeoffs?</summary>

**Answer**: Both approaches are valid with different tradeoffs.

**Shared layers:**
```
State → [Shared MLP] → shared_features
                          ↓         ↓
                      Actor head  Critic head
                      (policy)    (value)
```

**Separate networks:**
```
State → [Actor MLP] → policy
State → [Critic MLP] → value
```

**Tradeoffs:**

| Aspect | Shared | Separate |
|--------|--------|----------|
| Parameters | Fewer | More |
| Gradient interference | Risk of conflict | No interference |
| Feature learning | Shared representations | Task-specific features |
| Stability | Can be unstable | More stable |
| Common usage | A3C original paper | Modern implementations (PPO) |

**When to share:**
- Simple environments where same features useful for both
- Compute-constrained settings
- When you want faster feature learning

**When to separate:**
- Complex environments
- When experiencing training instability
- When actor and critic need different representations

**Mitigation for shared networks:**
- Gradient scaling: Weight critic gradient lower
- Stop gradient: Don't let critic gradient affect shared layers
- Separate optimizers with different learning rates

**Common pitfall**: Assuming sharing is always better for efficiency. The instability cost often outweighs parameter savings.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 13.5
- **Mnih et al. (2016)**, Asynchronous Methods for Deep RL (A3C)
- **Konda & Tsitsiklis (2000)**, Actor-Critic Algorithms
- **Schulman et al. (2016)**, High-Dimensional Continuous Control Using GAE

**What to memorize for interviews**: Actor-critic architecture, TD error as advantage (with proof), A2C vs A3C differences, entropy regularization, why critic reduces variance, shared vs separate networks tradeoff.

**Code example**: [actor_critic.py](../../../rl_examples/algorithms/actor_critic.py)
