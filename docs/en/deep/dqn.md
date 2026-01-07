# Deep Q-Network (DQN)

## Interview Summary

**DQN** combines Q-learning with deep neural networks to solve high-dimensional problems (e.g., Atari from pixels). Two key innovations: **experience replay** (stores and samples past transitions) and **target networks** (stabilizes training by using frozen Q for targets). First algorithm to achieve human-level performance on Atari games. Foundation for modern value-based deep RL.

**What to memorize**: DQN loss function, replay buffer purpose, target network purpose, the two key tricks.

---

## Core Definitions

### DQN Architecture

**Input**: State \(s\) (e.g., 84×84×4 stacked frames)
**Output**: \(Q(s, a)\) for all actions \(a \in A\)

```
State → Conv layers → FC layers → Q-values for all actions
```

### Experience Replay Buffer

Store transitions \((s, a, r, s', \text{done})\) in buffer \(D\).
Sample minibatches uniformly for training.

**Purpose**:
1. Break temporal correlation
2. Reuse data (sample efficiency)
3. Smooth out non-stationarity

### Target Network

Maintain a separate network \(Q(s, a; \theta^-)\) for computing targets.
Update periodically: \(\theta^- \leftarrow \theta\) every \(C\) steps.

**Purpose**: Stable targets for regression (not moving while we train).

---

## Math and Derivations

### DQN Loss Function

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\left[ \left( y^{DQN} - Q(s, a; \theta) \right)^2 \right]$$

where:

$$y^{DQN} = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

### Gradient

$$\nabla_\theta L = \mathbb{E}\left[ -2 \left( y^{DQN} - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

**Note**: \(y^{DQN}\) treated as constant (no gradient through \(\theta^-\)).

### Huber Loss Variant

To reduce sensitivity to outliers:

$$L_\delta(x) = \begin{cases} \frac{1}{2}x^2 & |x| \leq \delta \\ \delta(|x| - \frac{\delta}{2}) & \text{otherwise} \end{cases}$$

Smooth near zero, linear for large errors.

---

## Algorithm Sketch

```
Algorithm: DQN

Hyperparameters: γ, ε, buffer size N, batch size B, target update C

1. Initialize Q-network θ, target network θ^- = θ
2. Initialize replay buffer D (capacity N)
3. For episode = 1 to M:
     s = env.reset()
     For t = 1 to T:
         # Action selection
         With probability ε: a = random action
         Otherwise: a = argmax_a Q(s, a; θ)

         # Environment step
         s', r, done = env.step(a)
         Store (s, a, r, s', done) in D
         s = s'

         # Learning step
         Sample minibatch of B transitions from D
         For each (s_j, a_j, r_j, s'_j, done_j):
             If done_j:
                 y_j = r_j
             Else:
                 y_j = r_j + γ max_a' Q(s'_j, a'; θ^-)

         Compute loss: L = (1/B) Σ (y_j - Q(s_j, a_j; θ))²
         Gradient step: θ ← θ - α∇_θ L

         # Target update
         Every C steps: θ^- ← θ

         If done: break
```

### Key Hyperparameters (Atari)

| Hyperparameter | Value |
|----------------|-------|
| Replay buffer size | 1M transitions |
| Batch size | 32 |
| Target update | Every 10K steps |
| Learning rate | 2.5e-4 |
| ε decay | 1.0 → 0.1 over 1M steps |
| γ | 0.99 |

---

## Common Pitfalls

1. **Buffer too small**: Need diversity; 1M for Atari, 10K+ for simple envs.

2. **Target network updated too often**: Updates every step negates benefit. Try every 1000-10000 steps.

3. **ε decay too fast**: Not enough exploration early on.

4. **Reward clipping forgotten**: Atari clips to {-1, 0, +1}. Large rewards destabilize.

5. **Frame stacking missing**: For Atari, stack 4 frames to capture motion.

6. **Ignoring done flag**: Terminal states should have \(y = r\), not \(r + \gamma Q(s')\).

---

## Mini Example

**CartPole DQN:**

```python
# Simplified pseudocode
network = MLP(4, [128, 128], 2)  # 4 state dims, 2 actions
target_net = copy(network)
buffer = ReplayBuffer(10000)

for episode in range(500):
    state = env.reset()
    while not done:
        # ε-greedy
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(network(state))

        next_state, reward, done = env.step(action)
        buffer.add(state, action, reward, next_state, done)

        # Train
        batch = buffer.sample(32)
        targets = rewards + gamma * target_net(next_states).max(1) * (1-dones)
        loss = MSE(network(states)[actions], targets)
        optimizer.step()

        # Update target
        if steps % 100 == 0:
            target_net = copy(network)
```

**Expected**: Solves CartPole in ~100-200 episodes.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why is experience replay necessary for DQN?</summary>

**Answer**: Two reasons:
1. **Decorrelation**: Sequential transitions are correlated, violating SGD's i.i.d. assumption
2. **Data efficiency**: Each transition is used multiple times

**Explanation**: Without replay, gradients are biased by recent experience. The network might "forget" earlier patterns. Replay maintains a diverse dataset and allows uniform sampling.

**Key insight**: Replay transforms online RL into something closer to supervised learning.

**Common pitfall**: Using a too-small buffer. Need enough diversity to cover the state space.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> What happens if we don't use a target network?</summary>

**Answer**: Training becomes unstable — the target we're regressing toward changes with every update, leading to oscillation or divergence.

**Explanation**: The loss is \((y - Q(s,a;\theta))^2\) where \(y = r + \gamma \max Q(s',a';\theta)\). If \(\theta\) is used for targets, every gradient step changes the target. It's like chasing a moving goal.

**Key insight**: Target networks make DQN like supervised learning with fixed labels.

**Common pitfall**: Thinking target network is just for efficiency. It's essential for stability.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Write the DQN loss function and explain each term.</summary>

**Answer**:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

**Terms**:
- \(r + \gamma \max_{a'} Q(s', a'; \theta^-)\): TD target using target network
- \(Q(s, a; \theta)\): Current Q-network prediction
- \(D\): Replay buffer (uniform sampling)
- The difference is squared to form MSE loss

**Key point**: Gradient only through \(Q(s,a;\theta)\), not through target.

**Common pitfall**: Including gradient through target — this changes the algorithm.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> What is Double DQN and how does it fix overestimation?</summary>

**Answer**: Double DQN uses current network to select action, target network to evaluate:

$$y^{DDQN} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

**Explanation**: Standard DQN uses \(\max_{a'} Q(s', a'; \theta^-)\), which both selects and evaluates with the same network. This leads to overestimation. Double DQN decouples: \(\theta\) selects, \(\theta^-\) evaluates.

**Key insight**: Even if selection is overoptimistic, evaluation uses different parameters, reducing bias.

**Common pitfall**: Thinking you need two separate networks. Just use \(\theta\) and \(\theta^-\) differently.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your DQN learns initially but then performance drops. What's happening?</summary>

**Answer**: Possible causes:
1. **Overfitting**: Replay buffer becomes stale as policy changes
2. **Catastrophic forgetting**: Network forgets earlier experience
3. **ε decayed too much**: Stuck exploiting suboptimal policy
4. **Target network lag**: Target too outdated

**Debugging**:
- Monitor Q-value magnitudes (should stabilize, not explode)
- Check replay buffer diversity
- Try slower ε decay
- Larger replay buffer

**Common pitfall**: Not monitoring training curves. Plot Q-values, loss, episode rewards.
</details>

---

## References

- **Mnih et al. (2015)**, Human-level control through deep reinforcement learning (Nature)
- **Van Hasselt et al. (2016)**, Deep RL with Double Q-learning
- **Schaul et al. (2016)**, Prioritized Experience Replay
- **Wang et al. (2016)**, Dueling Network Architectures

**What to memorize for interviews**: DQN loss, experience replay purpose, target network purpose, Double DQN, key hyperparameters.

**Code example**: [dqn.py](../../../rl_examples/algorithms/dqn.py)
