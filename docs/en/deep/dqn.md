# Deep Q-Network (DQN)

## Interview Summary

**DQN** combines Q-learning with deep neural networks to solve high-dimensional problems (e.g., Atari from pixels). Two key innovations: **experience replay** (stores and samples past transitions) and **target networks** (stabilizes training by using frozen Q for targets). First algorithm to achieve human-level performance on Atari games. Foundation for modern value-based deep RL.

**What to memorize**: DQN loss function, replay buffer purpose, target network purpose, the deadly triad, why both tricks are essential.

---

## Design Motivation: Why DQN Exists

### The Dream: Q-Learning + Neural Networks

Q-learning works great for small problems. Can we just use a neural network to represent Q?

$$Q(s, a) \approx Q(s, a; \theta)$$

**Naive approach**: Run Q-learning, but use a neural network instead of a table.

```python
# Naive Deep Q-Learning (DOESN'T WORK!)
target = r + gamma * max(Q(next_state; theta))
loss = (target - Q(state, action; theta))^2
theta = theta - lr * gradient(loss)
```

### Why Naive Deep Q-Learning Fails

**Problem 1: Correlated samples**
- Q-learning updates use consecutive transitions: $(s_t, a_t, r_t, s_{t+1})$
- Consecutive samples are highly correlated
- Neural networks + correlated data = unstable training
- SGD assumes i.i.d. samples!

**Problem 2: Moving targets**
- Target: $y = r + \gamma \max_{a'} Q(s', a'; \theta)$
- Every gradient step changes $\theta$
- Which changes the target $y$
- We're chasing a moving goal!

```
Update θ → Q changes → Target changes → Update θ → ...
          (This feedback loop causes divergence!)
```

**Problem 3: The Deadly Triad**
Sutton & Barto identified that combining these three causes instability:
1. **Function approximation** (neural networks)
2. **Bootstrapping** (using Q estimates as targets)
3. **Off-policy learning** (learning from replay buffer)

DQN has all three! So how does it work?

### DQN's Solution: Two Simple Tricks

| Problem | Solution |
|---------|----------|
| Correlated samples | **Experience Replay**: Store transitions, sample randomly |
| Moving targets | **Target Network**: Freeze target Q, update periodically |

These two tricks transform unstable deep Q-learning into a stable algorithm.

---

## Core Definitions

### DQN Architecture

**Input**: State $s$ (e.g., 84×84×4 stacked frames for Atari)
**Output**: $Q(s, a; \theta)$ for all actions $a \in A$

```
State (pixels) → Conv layers → FC layers → Q-value for each action
                     ↓
              [Q(s,left), Q(s,right), Q(s,up), Q(s,down)]
```

**Key insight**: Output Q-values for ALL actions at once. One forward pass, then take argmax.

### Experience Replay Buffer

Store transitions $(s, a, r, s', \text{done})$ in a buffer $D$ of fixed size.
Sample minibatches **uniformly at random** for training.

**Why it works:**

| Without Replay | With Replay |
|----------------|-------------|
| Train on $(s_1, s_2, s_3, ...)$ | Train on random $(s_{42}, s_{1337}, s_{7}, ...)$ |
| Samples correlated | Samples approximately i.i.d. |
| Forget old experience | Reuse experience many times |
| High variance | Lower variance |

**Analogy**: Like studying for an exam by reviewing random flashcards instead of re-reading the textbook in order.

### Target Network

Maintain TWO networks:
- **Online network** $Q(s, a; \theta)$: Updated every step
- **Target network** $Q(s, a; \theta^-)$: Frozen, updated every $C$ steps

$$\theta^- \leftarrow \theta \quad \text{every } C \text{ steps}$$

**Why it works:**

| Without Target Network | With Target Network |
|------------------------|---------------------|
| Target = $r + \gamma \max Q(s'; \theta)$ | Target = $r + \gamma \max Q(s'; \theta^-)$ |
| Target changes every step | Target stable for $C$ steps |
| Chasing moving goal | Stable regression problem |
| Oscillation/divergence | Convergence |

**Analogy**: Like hitting a target that stays still for a while, then moves. Much easier than hitting a constantly moving target.

---

## Math and Derivations

### DQN Loss Function

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\left[ \left( y^{DQN} - Q(s, a; \theta) \right)^2 \right]$$

where the target is:

$$y^{DQN} = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

**Important**: $y^{DQN}$ is treated as a **constant** (no gradient flows through $\theta^-$).

### Understanding the Gradient

$$\nabla_\theta L = \mathbb{E}\left[ -2 \left( y^{DQN} - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

**Note**: Gradient only through the prediction $Q(s,a;\theta)$, NOT through the target!

### Why Experience Replay Helps Mathematically

Standard Q-learning gradient:
$$\nabla_\theta L = (y - Q(s_t, a_t; \theta)) \nabla_\theta Q(s_t, a_t; \theta)$$

With replay, we average over many samples:
$$\nabla_\theta L = \frac{1}{B} \sum_{i=1}^{B} (y_i - Q(s_i, a_i; \theta)) \nabla_\theta Q(s_i, a_i; \theta)$$

By sampling uniformly from buffer, samples are approximately i.i.d., so this is a valid Monte Carlo estimate of the expected gradient.

### Huber Loss (Smooth L1)

MSE is sensitive to outliers. Huber loss is more robust:

$$L_\delta(x) = \begin{cases} \frac{1}{2}x^2 & |x| \leq \delta \\ \delta(|x| - \frac{\delta}{2}) & \text{otherwise} \end{cases}$$

- Quadratic (like MSE) near zero → smooth gradients
- Linear for large errors → doesn't explode from outliers

---

## Algorithm Sketch

```
Algorithm: DQN

Hyperparameters: γ (discount), ε (exploration), N (buffer size),
                 B (batch size), C (target update frequency)

1. Initialize Q-network θ randomly
2. Initialize target network θ^- = θ
3. Initialize replay buffer D (capacity N)

4. For episode = 1 to M:
     s = env.reset()
     For t = 1 to T:
         # ε-greedy action selection
         With probability ε: a = random action
         Otherwise: a = argmax_a Q(s, a; θ)

         # Environment step
         s', r, done = env.step(a)

         # Store transition (EXPERIENCE REPLAY)
         Store (s, a, r, s', done) in D
         s = s'

         # Learning step
         Sample random minibatch of B transitions from D

         For each (s_j, a_j, r_j, s'_j, done_j):
             If done_j:
                 y_j = r_j                          # Terminal: no future
             Else:
                 y_j = r_j + γ max_a' Q(s'_j, a'; θ^-)   # TARGET NETWORK

         # Gradient descent
         L = (1/B) Σ (y_j - Q(s_j, a_j; θ))²
         θ ← θ - α∇_θ L

         # Target network update (every C steps)
         Every C steps: θ^- ← θ

         If done: break
```

### Key Design Choices Explained

| Design Choice | Why? |
|--------------|------|
| Output Q for all actions | Single forward pass, efficient argmax |
| Uniform replay sampling | Approximately i.i.d. samples |
| Frozen target network | Stable regression targets |
| ε-greedy exploration | Simple, effective for discrete actions |
| Frame stacking (Atari) | Capture motion, velocity information |
| Reward clipping (Atari) | Prevent gradient explosion from large rewards |

### Key Hyperparameters

| Hyperparameter | Typical Value | Why This Value |
|----------------|---------------|----------------|
| Replay buffer size | 1M | Enough diversity, not too much memory |
| Batch size | 32 | Standard mini-batch size |
| Target update (C) | 10K steps | Stable enough, not too stale |
| Learning rate | 2.5e-4 | Low to prevent instability |
| ε decay | 1.0 → 0.1 over 1M | Explore early, exploit later |
| γ | 0.99 | Value long-term rewards |

---

## DQN Extensions: Evolution of Ideas

```
DQN (2015)
  ↓ Problem: Overestimates Q-values
Double DQN (2016)
  ↓ Problem: Not all experiences equally important
Prioritized Experience Replay (2016)
  ↓ Problem: Value and advantage conflated
Dueling DQN (2016)
  ↓ Combine all improvements
Rainbow (2017) - All of the above + more
```

### Double DQN

**Problem**: $\max_{a'} Q(s', a'; \theta^-)$ overestimates because max of noisy estimates is biased upward.

**Solution**: Decouple action selection from evaluation:

$$y^{DDQN} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

- **Select** best action with online network $\theta$
- **Evaluate** that action with target network $\theta^-$

### Prioritized Experience Replay

**Problem**: Uniform sampling wastes time on "easy" transitions.

**Solution**: Sample proportional to TD error:
$$P(i) \propto |\delta_i|^\alpha$$

Transitions with high surprise are sampled more often.

### Dueling DQN

**Problem**: Q-value = how good is action + how good is state. These are conflated.

**Solution**: Decompose Q into value and advantage:
$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')$$

---

## Common Pitfalls

1. **Buffer too small**: Need diversity. 1M for Atari, 10K+ for simple envs. Small buffer = overfitting to recent experience.

2. **Target network updated too often**: If you update every step, you lose the stability benefit. Try every 1000-10000 steps.

3. **ε decayed too fast**: Need exploration early. If ε hits 0.1 too soon, you might miss good strategies.

4. **Reward clipping forgotten**: Atari clips rewards to {-1, 0, +1}. Without this, gradients explode.

5. **Frame stacking missing**: For visual inputs, single frame doesn't show velocity/direction. Stack 4 frames.

6. **Ignoring done flag**: Terminal states must have $y = r$, NOT $r + \gamma Q(s')$. Otherwise you bootstrap from non-existent states!

7. **Gradient through target**: Make sure no gradient flows through $\theta^-$. Use `.detach()` or `stop_gradient()`.

---

## Mini Example

**CartPole DQN:**

```python
# Networks
q_network = MLP(state_dim=4, hidden=[128, 128], output_dim=2)
target_network = copy(q_network)
buffer = ReplayBuffer(capacity=10000)

for episode in range(500):
    state = env.reset()
    while not done:
        # ε-greedy action selection
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(q_network(state))

        next_state, reward, done = env.step(action)
        buffer.add(state, action, reward, next_state, done)

        # Train (if buffer has enough samples)
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size=32)

            # Compute targets (NO GRADIENT through target_network!)
            with no_grad():
                next_q = target_network(batch.next_states).max(dim=1)
                targets = batch.rewards + gamma * next_q * (1 - batch.dones)

            # Compute predictions
            predictions = q_network(batch.states)[batch.actions]

            # MSE loss and gradient step
            loss = MSE(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network periodically
        if steps % 100 == 0:
            target_network.load_state_dict(q_network.state_dict())

        state = next_state
```

**Expected**: Solves CartPole in ~100-200 episodes.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What is the "deadly triad" and how does DQN survive it?</summary>

**Answer**: The deadly triad is the combination of:
1. **Function approximation** (neural networks)
2. **Bootstrapping** (using Q estimates in targets)
3. **Off-policy learning** (learning from replay buffer)

This combination often causes divergence. DQN survives by:
- **Experience replay**: Reduces correlation, makes data more i.i.d.
- **Target network**: Stabilizes the bootstrap targets

**Key insight**: DQN doesn't eliminate the deadly triad—it manages it through these two tricks.

**Common pitfall**: Thinking either trick alone is sufficient. You need BOTH.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why does experience replay break correlation, and why does that matter?</summary>

**Answer**:
- Without replay: Train on $(s_1, s_2, s_3, ...)$ — consecutive, correlated
- With replay: Train on random $(s_{42}, s_{1337}, s_7, ...)$ — approximately i.i.d.

**Why it matters**: SGD theory assumes i.i.d. samples. Correlated samples cause the gradient to be biased in a particular direction, leading to:
- Overfitting to recent experience
- Forgetting earlier experience
- Unstable training

**Analogy**: Studying only the last chapter before an exam vs. randomly reviewing all chapters.

**Common pitfall**: Using a too-small buffer. If buffer only holds 100 transitions, samples are still correlated.
</details>

<details markdown="1">
<summary><strong>Q3 (Conceptual):</strong> Explain why we need a target network with an analogy.</summary>

**Answer**: Imagine learning archery where the target moves every time you shoot.

**Without target network**:
- You aim at position X, shoot
- Before your arrow lands, target moves to Y
- You adjust aim to Y, shoot
- Target moves to Z...
- You never converge because you're always chasing

**With target network**:
- Target stays at X for 100 shots
- You learn to hit X consistently
- Then target moves to X'
- You adjust and learn X'
- Gradual improvement!

**Key insight**: Temporary stability allows learning.

**Common pitfall**: Updating target too frequently negates the benefit.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> Write the DQN loss and explain why gradient doesn't flow through the target.</summary>

**Answer**:

$$L(\theta) = \mathbb{E}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

**Why no gradient through target**:
- $\theta^-$ is a frozen copy, not a function of $\theta$
- In code: `target = r + gamma * target_network(s').max()` uses `target_network` which has separate parameters
- We only want to move $Q(s,a;\theta)$ toward the target, not move the target toward $Q$

**If we included gradient through target**: Both would move, creating a feedback loop → instability.

**Common pitfall**: Forgetting `.detach()` or `stop_gradient()` in implementation.
</details>

<details markdown="1">
<summary><strong>Q5 (Math):</strong> What is Double DQN and why does standard DQN overestimate?</summary>

**Answer**: Standard DQN uses $\max_{a'} Q(s', a'; \theta^-)$ which overestimates.

**Why overestimation**:
- Q-values are noisy estimates
- $\max$ of noisy values > true max (Jensen's inequality for convex functions)
- Error accumulates through bootstrapping

**Double DQN solution**:
$$y^{DDQN} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

- **Select** action with online network $\theta$
- **Evaluate** with target network $\theta^-$
- If selection is overoptimistic, evaluation with different parameters corrects it

**Common pitfall**: Thinking you need two separate networks. Just use $\theta$ and $\theta^-$ differently.
</details>

<details markdown="1">
<summary><strong>Q6 (Practical):</strong> Your DQN training curve shows initial learning then performance collapse. Diagnose.</summary>

**Answer**: Possible causes:

1. **Replay buffer overflow**: Old good experiences pushed out, only recent (possibly bad) experiences remain
   - Fix: Larger buffer, or prioritized replay

2. **Target network too stale**: Target is based on very old policy
   - Fix: Update target more frequently (but not every step)

3. **ε decayed too much**: Stuck exploiting a suboptimal policy
   - Fix: Slower ε decay, or periodic exploration boosts

4. **Q-value explosion**: Check if Q-values are growing unboundedly
   - Fix: Gradient clipping, reward scaling, Huber loss

**Debugging tips**:
- Plot Q-values over time (should stabilize, not explode)
- Plot replay buffer statistics
- Check ε schedule

**Common pitfall**: Not logging enough metrics. Always track Q-values, loss, buffer size.
</details>

---

## References

- **Mnih et al. (2015)**, Human-level control through deep reinforcement learning (Nature DQN paper)
- **Van Hasselt et al. (2016)**, Deep RL with Double Q-learning
- **Schaul et al. (2016)**, Prioritized Experience Replay
- **Wang et al. (2016)**, Dueling Network Architectures
- **Hessel et al. (2017)**, Rainbow: Combining Improvements in Deep RL

**What to memorize for interviews**: DQN loss, why experience replay (i.i.d.), why target network (stable targets), deadly triad, Double DQN, key hyperparameters.

**Code example**: [dqn.py](../../../rl_examples/algorithms/dqn.py)
