# Actor-Critic Methods

## Interview Summary

**Actor-Critic** methods combine policy gradients (actor) with value function learning (critic). The **actor** updates the policy using policy gradients, while the **critic** estimates value functions to reduce variance. This is the foundation for A2C, A3C, and PPO. Key insight: use TD error as a low-variance estimate of advantage, enabling per-step updates instead of waiting for episode end.

**What to memorize**: Actor-critic architecture, TD error as advantage, A2C update rules, A3C's asynchronous training.

---

## Core Definitions

### Actor-Critic Architecture

**Actor**: Policy \(\pi_\theta(a|s)\) — decides what actions to take
**Critic**: Value function \(V_\phi(s)\) or \(Q_\phi(s,a)\) — evaluates how good states/actions are

### Why Combine?

| Pure Policy Gradient | Pure Value-Based |
|---------------------|------------------|
| High variance | Lower variance |
| Can learn stochastic policies | Deterministic policies |
| On-policy only | Can be off-policy |
| Works for continuous actions | Requires max (discrete) |

Actor-Critic gets benefits of both: lower variance from critic, stochastic policies from actor.

### TD Error as Advantage Estimate

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

This is an unbiased estimate of the advantage \(A^\pi(s_t, a_t)\) when averaged over trajectories.

---

## Math and Derivations

### Actor Update (Policy Gradient with Critic)

$$\nabla_\theta J \approx \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s, a) \right]$$

where \(\hat{A}\) is the advantage estimate:
- **Monte Carlo**: \(\hat{A} = G_t - V_\phi(s_t)\)
- **TD(0)**: \(\hat{A} = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\)
- **GAE**: Weighted combination (see GAE chapter)

### Critic Update (TD Learning)

$$\phi \leftarrow \phi + \alpha_c \cdot \delta_t \cdot \nabla_\phi V_\phi(s_t)$$

Or with MSE loss:

$$L(\phi) = \mathbb{E}\left[ (V_\phi(s) - V^{target})^2 \right]$$

where \(V^{target} = r + \gamma V_\phi(s')\) or \(G_t\) (Monte Carlo).

### Why TD Error Estimates Advantage

$$\mathbb{E}[\delta_t | s_t, a_t] = \mathbb{E}[r_t + \gamma V(s_{t+1}) | s_t, a_t] - V(s_t)$$

$$= Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

So TD error is an unbiased (but noisy) estimate of advantage!

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
         δ = r - V_φ(s)
     Else:
         δ = r + γ V_φ(s') - V_φ(s)

     # Critic update
     φ ← φ + α_φ · δ · ∇_φ V_φ(s)

     # Actor update
     θ ← θ + α_θ · δ · ∇_θ log π_θ(a|s)

     # Optional: entropy bonus for exploration
     θ ← θ + β · ∇_θ H(π_θ(·|s))

     s ← s'
```

### A3C (Asynchronous Advantage Actor-Critic)

```
Algorithm: A3C (High-level)

1. Global parameters: θ, φ (shared)
2. Launch N parallel workers
3. Each worker:
     Local copy of θ, φ
     Interact with own environment
     Collect T steps of experience
     Compute gradients locally
     Apply gradients to global θ, φ (async)
     Sync local ← global
```

**Key benefit**: Parallel workers provide diverse experience, natural exploration.

---

## Common Pitfalls

1. **Critic learns too slowly**: If critic is bad, actor gets noisy gradients. Critic often needs higher learning rate.

2. **Shared network issues**: Sharing layers between actor/critic can cause interference. Use separate networks or careful architecture.

3. **Entropy collapse**: Policy becomes deterministic too fast. Add entropy bonus: \(L = -\log \pi(a|s) \cdot A - \beta H(\pi)\).

4. **Critic target issues**: Should critic target include new or old value estimate? Be consistent.

5. **Not handling terminal states**: \(V(s_{terminal}) = 0\), don't bootstrap from it.

---

## Mini Example

**CartPole Actor-Critic:**

```python
# Networks
actor = MLP(4, [32], 2)   # policy logits
critic = MLP(4, [32], 1)  # value

for episode in range(1000):
    state = env.reset()
    while not done:
        # Forward pass
        logits = actor(state)
        probs = softmax(logits)
        value = critic(state)

        # Sample action
        action = sample(probs)
        next_state, reward, done = env.step(action)

        # TD error
        next_value = 0 if done else critic(next_state)
        td_error = reward + gamma * next_value - value

        # Critic update
        critic_loss = td_error ** 2
        critic_loss.backward()

        # Actor update
        log_prob = log_softmax(logits)[action]
        actor_loss = -log_prob * td_error.detach()  # stop gradient through td_error
        actor_loss.backward()

        optimizer.step()
        state = next_state
```

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why use actor-critic instead of pure REINFORCE?</summary>

**Answer**: Lower variance from the critic.

**Explanation**: REINFORCE uses full episode returns \(G_t\), which have high variance (sum of many random rewards). Actor-critic uses TD error \(\delta_t = r + \gamma V(s') - V(s)\), which involves only one reward plus a learned estimate.

**Trade-off**: TD error introduces bias (if V is wrong), but the variance reduction usually wins.

**Common pitfall**: Thinking actor-critic is always better. For short episodes with sparse rewards, REINFORCE might be competitive.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> What's the role of entropy regularization in actor-critic?</summary>

**Answer**: Prevents premature convergence to deterministic policies, encouraging exploration.

**Explanation**: Without entropy bonus, policy gradients push toward deterministic policies (one action gets all probability). Adding \(\beta H(\pi)\) to the objective rewards uncertainty.

**Implementation**: \(\nabla_\theta [\beta H(\pi)] = -\beta \sum_a \pi(a|s) \log \pi(a|s)\)

**Common pitfall**: Setting \(\beta\) too high makes the policy too random. Too low allows entropy collapse.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Show that E[δ_t | s_t, a_t] = A^π(s_t, a_t).</summary>

**Answer**:

$$\mathbb{E}[\delta_t | s_t, a_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) | s_t, a_t] - V^\pi(s_t)$$

$$= R(s_t, a_t) + \gamma \sum_{s'} P(s'|s_t, a_t) V^\pi(s') - V^\pi(s_t)$$

$$= Q^\pi(s_t, a_t) - V^\pi(s_t)$$

$$= A^\pi(s_t, a_t)$$

**Key insight**: TD error is an unbiased estimator of advantage!

**Common pitfall**: This requires \(V^\pi\) to be correct. With learned \(V_\phi\), we get biased estimates.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> How does A3C achieve exploration without explicit exploration bonus?</summary>

**Answer**: Parallel workers with different random seeds naturally explore different parts of state space.

**Explanation**: Each worker runs independently, making different random action choices. The global network aggregates these diverse experiences. This provides implicit exploration through diversity.

**Additional mechanism**: Entropy bonus is still typically used in A3C.

**Common pitfall**: Thinking parallelism is only for speed. It genuinely improves exploration and stability.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your actor-critic isn't learning. Actor loss is oscillating. What to debug?</summary>

**Answer**: Debugging steps:
1. **Check critic first**: If critic is bad, actor can't learn. Train critic longer, use higher α_φ.
2. **Reduce α_θ**: Actor learning rate too high causes oscillation.
3. **Detach TD error**: Don't backprop actor gradient through critic.
4. **Add entropy bonus**: Might be stuck in local optimum.
5. **Check advantage signs**: Positive for good actions, negative for bad.
6. **Use batching**: Single-step updates are noisy; try n-step or batch updates.

**Explanation**: Most actor-critic failures are due to critic quality. A bad critic gives noisy gradients.

**Common pitfall**: Blaming the actor when the problem is the critic.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapter 13.5
- **Mnih et al. (2016)**, Asynchronous Methods for Deep RL (A3C)
- **Konda & Tsitsiklis (2000)**, Actor-Critic Algorithms

**What to memorize for interviews**: Actor-critic architecture, TD error as advantage, A2C vs A3C, entropy regularization purpose.

**Code example**: [actor_critic.py](../../../rl_examples/algorithms/actor_critic.py)
