# Common Interview Questions

## Interview Summary

This page compiles frequently asked RL interview questions with concise answers. Topics cover fundamentals, algorithms, deep RL, and practical considerations. Use this as a quick reference before interviews.

---

## Fundamentals

<details markdown="1">
<summary><strong>Q: What is the difference between model-based and model-free RL?</strong></summary>

**Answer**:
- **Model-based**: Learns/uses a model of environment dynamics P(s'|s,a) and R(s,a). Plans by simulating trajectories.
- **Model-free**: Learns policy/value directly from experience without modeling dynamics.

**Trade-offs**: Model-based is more sample-efficient but requires accurate models. Model-free is simpler but needs more data.
</details>

<details markdown="1">
<summary><strong>Q: Explain the Bellman equation.</strong></summary>

**Answer**: The Bellman equation expresses value recursively:

$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

It says: value = immediate reward + discounted future value. Foundation of DP, TD, and Q-learning.
</details>

<details markdown="1">
<summary><strong>Q: What is the difference between on-policy and off-policy?</strong></summary>

**Answer**:
- **On-policy**: Learns about the policy being used to collect data (e.g., SARSA)
- **Off-policy**: Learns about a different policy than the one collecting data (e.g., Q-learning)

Off-policy allows learning from any data but has stability issues. On-policy is more stable but less sample-efficient.
</details>

<details markdown="1">
<summary><strong>Q: Why do we use a discount factor γ?</strong></summary>

**Answer**:
1. **Mathematical**: Ensures finite returns for infinite horizons
2. **Behavioral**: Models preference for immediate rewards
3. **Practical**: Reduces variance in value estimates

Common values: 0.99 (far-sighted), 0.9 (medium), 0.5 (myopic).
</details>

<details markdown="1">
<summary><strong>Q: What is the Markov property?</strong></summary>

**Answer**: The future is independent of the past given the present:

$$P(S_{t+1}|S_t, A_t, S_{t-1}, ..., S_0) = P(S_{t+1}|S_t, A_t)$$

State contains all relevant information. Violated in partially observable environments.
</details>

---

## Value-Based Methods

<details markdown="1">
<summary><strong>Q: Explain Q-learning and its update rule.</strong></summary>

**Answer**: Q-learning is off-policy TD control:

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

Uses max to learn optimal Q regardless of exploration policy. Converges to Q* under standard conditions.
</details>

<details markdown="1">
<summary><strong>Q: What is the difference between SARSA and Q-learning?</strong></summary>

**Answer**:
- **SARSA**: \(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]\) — uses action actually taken
- **Q-learning**: \(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]\) — uses max

SARSA is on-policy (learns Q^π), Q-learning is off-policy (learns Q*).
</details>

<details markdown="1">
<summary><strong>Q: What is the deadly triad?</strong></summary>

**Answer**: Three elements that together can cause divergence:
1. Function approximation
2. Bootstrapping
3. Off-policy learning

Each alone is fine. Together, errors can compound and values diverge. Solutions: target networks, experience replay.
</details>

<details markdown="1">
<summary><strong>Q: How does DQN stabilize training?</strong></summary>

**Answer**: Two key innovations:
1. **Experience replay**: Stores and samples transitions randomly, breaking correlation
2. **Target network**: Uses frozen Q for targets, preventing moving-target problem

Also: reward clipping, frame stacking, Huber loss.
</details>

<details markdown="1">
<summary><strong>Q: What is Double DQN and why is it needed?</strong></summary>

**Answer**: Double DQN addresses overestimation bias:

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s',a';\theta); \theta^-)$$

Standard Q-learning's max over noisy estimates tends to overestimate. Double DQN uses θ to select action, θ⁻ to evaluate.
</details>

---

## Policy-Based Methods

<details markdown="1">
<summary><strong>Q: What is the policy gradient theorem?</strong></summary>

**Answer**:

$$\nabla J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$$

Gradient of expected return equals expected score function times Q-value. Foundation for REINFORCE, PPO.
</details>

<details markdown="1">
<summary><strong>Q: Why use a baseline in policy gradients?</strong></summary>

**Answer**: Reduces variance without adding bias.

$$\nabla J = \mathbb{E}[\nabla \log \pi(a|s)(Q(s,a) - b(s))]$$

Common baseline: V(s). Then advantage A(s,a) = Q(s,a) - V(s) centers rewards around zero.
</details>

<details markdown="1">
<summary><strong>Q: What is the advantage function?</strong></summary>

**Answer**:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

Measures how much better action a is compared to average. Positive = better than average, negative = worse.
</details>

<details markdown="1">
<summary><strong>Q: Explain actor-critic methods.</strong></summary>

**Answer**: Combines policy gradient (actor) with value function (critic):
- **Actor**: Updates policy using advantage estimates
- **Critic**: Estimates V(s) to compute advantages

Lower variance than REINFORCE (uses TD instead of MC).
</details>

<details markdown="1">
<summary><strong>Q: How does PPO work?</strong></summary>

**Answer**: PPO limits policy updates via clipping:

$$L = \min(r(\theta) A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A)$$

where r(θ) = π_θ(a|s)/π_old(a|s). Prevents too-large policy changes. Simple, stable, widely used.
</details>

---

## Deep RL

<details markdown="1">
<summary><strong>Q: Why is experience replay important?</strong></summary>

**Answer**:
1. **Decorrelation**: Breaks temporal correlation in sequential data
2. **Efficiency**: Each transition used multiple times
3. **Stability**: Smoother learning from diverse data

Required for stable DQN training.
</details>

<details markdown="1">
<summary><strong>Q: Explain target networks.</strong></summary>

**Answer**: A frozen copy of the Q-network used to compute TD targets:

$$y = r + \gamma \max_a Q(s', a; \theta^-)$$

Updated periodically (every C steps). Prevents chasing moving target.
</details>

<details markdown="1">
<summary><strong>Q: What is GAE?</strong></summary>

**Answer**: Generalized Advantage Estimation:

$$\hat{A}^{GAE}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

Interpolates between TD (λ=0, low variance) and MC (λ=1, low bias). Typically λ=0.95.
</details>

<details markdown="1">
<summary><strong>Q: How do you handle continuous action spaces?</strong></summary>

**Answer**: Options:
1. **Policy gradient**: Output Gaussian parameters μ, σ; sample actions
2. **DDPG/TD3**: Deterministic policy + noise for exploration
3. **SAC**: Maximum entropy framework with soft updates
4. **Discretization**: Coarse-grain actions (loses resolution)
</details>

<details markdown="1">
<summary><strong>Q: What is entropy regularization?</strong></summary>

**Answer**: Add entropy bonus to objective:

$$J(\theta) = \mathbb{E}[R] + \beta H(\pi_\theta)$$

Encourages exploration by penalizing deterministic policies. Common in PPO, SAC.
</details>

---

## Exploration

<details markdown="1">
<summary><strong>Q: Compare exploration strategies.</strong></summary>

**Answer**:
- **ε-greedy**: Random action with prob ε. Simple, doesn't prioritize.
- **Softmax**: Actions weighted by exp(Q/τ). Better-valued actions more likely.
- **UCB**: Bonus for uncertainty. Principled, requires counts.
- **Entropy bonus**: Rewards randomness in policy.
- **Intrinsic motivation**: Curiosity-driven exploration for sparse rewards.
</details>

<details markdown="1">
<summary><strong>Q: What is the exploration-exploitation tradeoff?</strong></summary>

**Answer**: Balance between:
- **Exploitation**: Use known good actions (short-term optimal)
- **Exploration**: Try uncertain actions (discover better strategies)

Early training: explore more. Later: exploit more.
</details>

---

## Practical

<details markdown="1">
<summary><strong>Q: How do you debug an RL agent that isn't learning?</strong></summary>

**Answer**: Debugging hierarchy:
1. **Rewards**: Print and verify reward signal
2. **Environment**: Render, manually interact
3. **Gradients**: Are they non-zero and reasonable?
4. **Exploration**: Is agent visiting diverse states?
5. **Hyperparameters**: Start from known-working settings
6. **Simple test**: Can it memorize a single transition?
</details>

<details markdown="1">
<summary><strong>Q: What hyperparameters are most important to tune?</strong></summary>

**Answer**: Priority order:
1. **Learning rate**: Most sensitive
2. **Network architecture**: Size and structure
3. **Batch size**: Affects gradient variance
4. **Discount γ**: Planning horizon
5. **Exploration parameters**: ε schedule, entropy coefficient
</details>

<details markdown="1">
<summary><strong>Q: How do you ensure reproducibility?</strong></summary>

**Answer**:
1. Set random seeds (numpy, torch, env)
2. Use deterministic operations
3. Log all hyperparameters
4. Version control code and configs
5. Run multiple seeds (3-5 minimum)
6. Report mean ± std error
</details>

<details markdown="1">
<summary><strong>Q: What is reward shaping?</strong></summary>

**Answer**: Adding intermediate rewards to guide learning:

$$r' = r + F(s, s')$$

Can accelerate learning but may change optimal policy if done wrong. Potential-based shaping preserves optimality.
</details>

<details markdown="1">
<summary><strong>Q: How do you handle sparse rewards?</strong></summary>

**Answer**: Strategies:
1. **Reward shaping**: Add intermediate rewards
2. **Curriculum learning**: Start with easier tasks
3. **Hindsight Experience Replay**: Relabel goals
4. **Intrinsic motivation**: Curiosity, count-based
5. **Demonstration/imitation**: Learn from expert
</details>

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> You're interviewing and asked "What's your favorite RL algorithm and why?" How do you answer?</summary>

**Answer**: Good structure:
1. Name the algorithm (e.g., PPO)
2. Explain why: "Stable, simple to implement, works across many domains"
3. Show understanding of tradeoffs: "Not as sample-efficient as SAC, but more robust"
4. Mention when you'd choose something else

**Example**: "I often use PPO because it's stable and works well for continuous control without careful hyperparameter tuning. For sample efficiency with continuous actions, I'd consider SAC. For discrete actions with high-dimensional observations, DQN variants."
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> "Walk me through training a DQN from scratch."</summary>

**Answer**: Structure your response:
1. **Environment**: State/action spaces, reward structure
2. **Network**: CNN for images, MLP for vectors → Q-values for all actions
3. **Replay buffer**: Store (s, a, r, s', done) tuples
4. **Target network**: Frozen copy, update every C steps
5. **Training loop**: ε-greedy action → step → store → sample batch → compute targets → gradient step
6. **Evaluation**: Periodically run greedy policy

Key points to mention: replay buffer breaks correlation, target network stabilizes training.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> "Derive the policy gradient."</summary>

**Answer**: Show the key steps:
1. \(J(\theta) = \mathbb{E}_\tau[R(\tau)]\)
2. \(\nabla J = \nabla \mathbb{E}[R] = \sum_\tau R(\tau) \nabla P(\tau|\theta)\)
3. Log-derivative: \(\nabla P = P \nabla \log P\)
4. \(\nabla J = \mathbb{E}[\nabla \log P(\tau|\theta) R(\tau)]\)
5. Only π depends on θ: \(\nabla \log P(\tau) = \sum_t \nabla \log \pi(a_t|s_t)\)

Final: \(\nabla J = \mathbb{E}[\sum_t \nabla \log \pi(a_t|s_t) G_t]\)
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> "Explain the bias-variance tradeoff in TD vs MC."</summary>

**Answer**:
- **MC**: Uses actual returns \(G_t\). Zero bias (by definition), high variance (sum of many random rewards)
- **TD**: Uses \(r + \gamma V(s')\). Biased (if V wrong), low variance (one reward)

GAE interpolates: λ controls the tradeoff. λ=0 is TD, λ=1 is MC.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> "How would you approach a new RL problem?"</summary>

**Answer**: Framework:
1. **Understand the problem**: State/action spaces, reward, episode structure
2. **Start simple**: Tabular if possible, random baseline
3. **Choose algorithm**: DQN for discrete, PPO/SAC for continuous
4. **Known hyperparameters**: Start from paper/working settings
5. **Monitor everything**: Returns, Q-values, gradients, entropy
6. **Debug systematically**: One change at a time
7. **Scale up**: More compute, hyperparameter tuning
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction
- **Spinning Up in Deep RL** (OpenAI)
- **RL Course by David Silver** (DeepMind)
- **Berkeley Deep RL Course** (CS 285)

**Interview tip**: Be honest about what you don't know. It's better to say "I haven't implemented that" than to fake understanding.
