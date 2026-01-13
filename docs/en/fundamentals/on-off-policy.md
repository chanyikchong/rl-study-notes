# On-Policy vs Off-Policy Learning

## Interview Summary

**On-policy** methods learn about the policy being used to generate data (behavior policy = target policy). **Off-policy** methods learn about a different policy than the one generating data (behavior policy ≠ target policy). SARSA is on-policy (learns about ε-greedy), Q-learning is off-policy (learns optimal policy while exploring). Off-policy enables experience replay and learning from demonstrations, but requires importance sampling corrections for convergence guarantees.

**What to memorize**: Definitions of behavior/target policy, SARSA vs Q-learning distinction, why off-policy enables replay buffers, importance sampling for off-policy MC.

---

## Design Motivation: Why Two Types?

### The Exploration-Exploitation Dilemma

To learn well, an agent must:
1. **Explore**: Try new actions to discover their effects
2. **Exploit**: Use current knowledge to maximize reward

**The problem**: If we always exploit (greedy), we never discover better actions. If we always explore (random), we never use what we learned.

**Common solution**: Use ε-greedy — mostly greedy, sometimes random.

### But What Should We Learn About?

Here's where on-policy vs off-policy differs:

**On-policy thinking**: "I'm behaving with ε-greedy, so I should learn the value of ε-greedy behavior."

**Off-policy thinking**: "I'm behaving with ε-greedy for exploration, but I want to learn the value of the greedy (optimal) policy."

```
On-policy:  Learn V^π where π = behavior policy (ε-greedy)
Off-policy: Learn V^π* where π* = optimal policy (greedy)
            while behaving according to π_b (ε-greedy)
```

---

## Core Definitions

### Behavior Policy vs Target Policy

| Term | Symbol | Definition |
|------|--------|------------|
| **Behavior Policy** | $\pi_b$ or $b$ | The policy used to generate experience (select actions) |
| **Target Policy** | $\pi$ | The policy we want to evaluate or improve |

### On-Policy

$$\pi_b = \pi$$

- We learn about the same policy we're using
- Data comes from the policy we're evaluating
- Simpler, more stable

**Examples**: SARSA, REINFORCE, A2C, PPO

### Off-Policy

$$\pi_b \neq \pi$$

- We learn about a different policy than we're using
- Can learn optimal policy while exploring
- Can reuse old data (experience replay)
- More sample efficient, but more complex

**Examples**: Q-learning, DQN, SAC, off-policy actor-critic

---

## Why Does This Distinction Matter?

### Example: The Cliff Walking Problem

```
[S][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][G]
[C][C][C][C][C][C][C][C][C][C][C][C]

S = Start, G = Goal, C = Cliff (reward = -100, reset to S)
Normal step reward = -1
```

**ε-greedy with ε = 0.1**:

**SARSA (On-policy)**:
- Learns value of ε-greedy policy
- Knows it will sometimes fall off cliff due to ε
- Learns to take the safe path (away from cliff edge)
- Actual behavior matches learned policy

**Q-learning (Off-policy)**:
- Learns value of optimal (greedy) policy
- Optimal policy walks right along the cliff edge (shortest path)
- But agent is ε-greedy, so it sometimes falls!
- Learned Q-values don't account for exploration randomness

```
SARSA path:    S → → → → → → → → → → → G  (safe, longer)
                ↑ avoids cliff edge

Q-learning:    S → → → → → → → → → → → G  (optimal but risky)
               walks along cliff, falls 10% of time
```

**Key insight**: On-policy is "safer" because it learns about actual behavior. Off-policy is more "optimistic" — it learns the best possible policy.

---

## The Math: Why Off-Policy Needs Correction

### On-Policy: No Correction Needed

When $\pi_b = \pi$, the expectation is straightforward:

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

We sample trajectories from $\pi$, compute returns, average them. Done.

### Off-Policy: Distribution Mismatch

When $\pi_b \neq \pi$, we have a problem:

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

But our data comes from $\pi_b$, not $\pi$! The state-action distribution is wrong.

### Importance Sampling

**Solution**: Reweight samples to correct for the distribution mismatch.

$$V^\pi(s) = \mathbb{E}_{\pi_b}\left[ \rho_t G_t | S_t = s \right]$$

where the **importance sampling ratio** is:

$$\rho_t = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{\pi_b(A_k|S_k)}$$

**Intuition**: If target policy $\pi$ would take action $a$ more often than behavior policy $\pi_b$, we upweight that sample.

### Problem: High Variance

The product of ratios can explode:

$$\rho_t = \frac{\pi(a_t)}{\pi_b(a_t)} \times \frac{\pi(a_{t+1})}{\pi_b(a_{t+1})} \times ... \times \frac{\pi(a_{T-1})}{\pi_b(a_{T-1})}$$

If any $\pi_b(a) \approx 0$ while $\pi(a) > 0$, the ratio becomes huge!

**Solutions**:
- Weighted importance sampling (lower variance, some bias)
- Per-decision importance sampling (only correct for one step)
- Use TD methods (naturally handle one-step updates)

---

## Why Q-Learning Doesn't Need Importance Sampling

Here's the magic of Q-learning:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

The $\max_{a'}$ operator implicitly selects the greedy action. We don't need to know the probability of taking that action under $\pi_b$.

**Key insight**: Q-learning is "off-policy for free" because:
1. We observe $(s, a, r, s')$ — doesn't matter which policy chose $a$
2. The target $r + \gamma \max_{a'} Q(s', a')$ is for the greedy policy
3. We only need one-step updates, not full trajectory corrections

**But**: This only works for one-step TD. Multi-step off-policy Q-learning does need corrections.

---

## Methods Comparison

### On-Policy Methods

| Method | Type | Key Feature |
|--------|------|-------------|
| **SARSA** | TD | Updates Q(s,a) toward r + γQ(s',a') where a' is from behavior |
| **Expected SARSA** | TD | Updates toward expected value under behavior policy |
| **REINFORCE** | Policy Gradient | Uses returns from behavior policy |
| **A2C/A3C** | Actor-Critic | Critic evaluates behavior policy |
| **PPO** | Policy Gradient | Limits policy change per update |

### Off-Policy Methods

| Method | Type | Key Feature |
|--------|------|-------------|
| **Q-learning** | TD | Uses max_a' Q(s',a') — greedy target |
| **DQN** | Deep TD | Q-learning + neural nets + replay buffer |
| **DDPG** | Actor-Critic | Off-policy with deterministic policy |
| **SAC** | Actor-Critic | Maximum entropy off-policy |
| **TD3** | Actor-Critic | Improved DDPG |

---

## SARSA vs Q-Learning: Side by Side

### Update Rules

**SARSA** (State-Action-Reward-State-Action):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

where $a'$ is the **actual next action** taken by the behavior policy.

**Q-learning**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

where $\max_{a'}$ selects the **best action** regardless of what was actually taken.

### Algorithm Comparison

```
SARSA:
1. In state s, choose a using ε-greedy on Q
2. Take action a, observe r, s'
3. Choose a' using ε-greedy on Q        ← uses actual next action
4. Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
5. s ← s', a ← a'

Q-learning:
1. In state s, choose a using ε-greedy on Q
2. Take action a, observe r, s'
3. Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]  ← uses max
4. s ← s'
```

### Numerical Example

```
State s, two actions: LEFT, RIGHT
Current: Q(s, LEFT) = 10, Q(s, RIGHT) = 5
Agent takes LEFT (ε-greedy chose greedy)
Reward r = 2, reaches s' where Q(s', LEFT) = 8, Q(s', RIGHT) = 12
Next action a' = RIGHT (ε-greedy chose greedy)
α = 0.1, γ = 0.9

SARSA update:
  Target = r + γ Q(s', a') = 2 + 0.9 × 12 = 12.8
  Q(s, LEFT) ← 10 + 0.1 × (12.8 - 10) = 10.28

Q-learning update:
  Target = r + γ max_a' Q(s', a') = 2 + 0.9 × max(8, 12) = 12.8
  Q(s, LEFT) ← 10 + 0.1 × (12.8 - 10) = 10.28

Same result here! But if a' was random (ε exploration):

If ε caused a' = LEFT (random):
  SARSA: Target = 2 + 0.9 × 8 = 9.2
         Q(s, LEFT) ← 10 + 0.1 × (9.2 - 10) = 9.92

  Q-learning: Target = 2 + 0.9 × 12 = 12.8  (still uses max!)
              Q(s, LEFT) ← 10 + 0.1 × (12.8 - 10) = 10.28

Now they differ! Q-learning ignores the suboptimal exploratory action.
```

---

## Why Off-Policy Enables Experience Replay

### The Replay Buffer

Off-policy methods can store past transitions and relearn from them:

```
Replay Buffer: [(s₁,a₁,r₁,s₁'), (s₂,a₂,r₂,s₂'), ..., (sₙ,aₙ,rₙ,sₙ')]
```

**Why this works for Q-learning**:
- Q-learning updates don't depend on which policy chose the action
- Old transitions are still valid training data
- Can sample uniformly for better learning

**Why this DOESN'T work for SARSA**:
- SARSA uses $Q(s', a')$ where $a'$ was the actual next action
- Old data was generated by an old policy $\pi_{old}$
- Current policy $\pi_{new}$ would choose different $a'$
- Using old $a'$ gives wrong updates!

### Sample Efficiency

```
On-policy:  Use data once, discard
Off-policy: Store data, reuse many times

Data efficiency: Off-policy >> On-policy
```

This is why DQN (off-policy) is much more sample efficient than REINFORCE (on-policy).

---

## Common Pitfalls

1. **Thinking "off-policy = bad"**: Off-policy enables replay buffers and learning from demonstrations. It's often more sample efficient.

2. **Forgetting SARSA needs a'**: SARSA requires observing the next action before updating. Q-learning doesn't.

3. **Using replay buffer with on-policy methods**: This breaks convergence guarantees! On-policy methods need fresh data from the current policy.

4. **Ignoring the cliff walking lesson**: On-policy learns safer policies when exploration is dangerous. Off-policy learns optimal but may not account for exploration risks.

5. **Thinking importance sampling is always needed**: Q-learning avoids it for one-step TD. Multi-step methods need corrections.

6. **Confusing "exploration" with "off-policy"**: ε-greedy is an exploration strategy. Whether you're on/off-policy depends on what you're learning about, not how you explore.

---

## Mini Example: When Each Shines

### On-Policy Preferred: Robot Learning

A robot learning to walk should use on-policy because:
- Safety matters — learn about actual behavior
- Falling is costly — SARSA learns to avoid risky states
- Policy changes gradually — no sudden dangerous behavior

### Off-Policy Preferred: Game Playing

An Atari game agent should use off-policy because:
- Can replay interesting experiences many times
- Learning from human demonstrations is off-policy
- Sample efficiency matters — can't play millions of games
- No real-world safety concerns

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> What is the key difference between on-policy and off-policy learning?</summary>

**Answer**: On-policy learns about the same policy generating data (behavior = target). Off-policy learns about a different policy than the one generating data (behavior ≠ target).

**Explanation**:
- On-policy: $\pi_b = \pi$ — evaluate/improve the policy you're using
- Off-policy: $\pi_b \neq \pi$ — evaluate/improve a target policy while using a different behavior policy

**Key equation**:
- SARSA (on-policy): Updates toward $Q(s', a')$ where $a'$ is from behavior policy
- Q-learning (off-policy): Updates toward $\max_{a'} Q(s', a')$ — optimal policy

**Common pitfall**: Thinking exploration makes something off-policy. ε-greedy with SARSA is still on-policy!
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why can Q-learning use experience replay but SARSA cannot?</summary>

**Answer**: Q-learning's update ($\max_{a'}$) is independent of which policy generated the data. SARSA's update uses the actual next action $a'$, which depends on the behavior policy.

**Explanation**:
- Q-learning: $r + \gamma \max_{a'} Q(s', a')$ — same target regardless of how $(s,a,r,s')$ was collected
- SARSA: $r + \gamma Q(s', a')$ — requires knowing what action the current policy would take

Old data from policy $\pi_{old}$ has $a'$ chosen by $\pi_{old}$. Using this for SARSA with current $\pi_{new}$ gives biased updates.

**Common pitfall**: Trying to use replay buffers with on-policy methods. This breaks the on-policy assumption.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> What is importance sampling and when is it needed?</summary>

**Answer**: Importance sampling reweights samples from behavior policy $\pi_b$ to estimate expectations under target policy $\pi$:

$$\mathbb{E}_\pi[f(x)] = \mathbb{E}_{\pi_b}\left[ \frac{\pi(x)}{\pi_b(x)} f(x) \right]$$

**When needed**:
- Off-policy Monte Carlo methods (full trajectory corrections)
- Off-policy multi-step TD methods
- Policy gradient methods using old data

**When NOT needed**:
- One-step Q-learning (max operator handles it)
- One-step off-policy TD with function approximation (semi-gradient methods)

**Key equation**: $\rho_t = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{\pi_b(A_k|S_k)}$

**Common pitfall**: Importance ratios can have very high variance when policies differ significantly.
</details>

<details markdown="1">
<summary><strong>Q4 (Conceptual):</strong> In the cliff walking example, why does SARSA find a different path than Q-learning?</summary>

**Answer**: SARSA learns the value of the ε-greedy policy (accounts for random exploration), while Q-learning learns the optimal policy (assumes greedy execution).

**Explanation**:
- Q-learning: Values reflect optimal path along cliff edge
- But agent is ε-greedy, so it sometimes falls
- SARSA: Values account for the 10% chance of random action
- Learns that cliff-edge states are risky with ε-greedy
- Takes longer but safer path

**Key insight**: On-policy methods are "realistic" about actual behavior. Off-policy methods are "optimistic" about optimal execution.

**Common pitfall**: Assuming off-policy always gives better policies. In safety-critical applications, on-policy may be preferred.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your DQN agent is learning slowly. You switch from uniform replay sampling to on-policy (using only most recent experience). What happens?</summary>

**Answer**: Performance likely degrades due to correlated samples and loss of diverse experience.

**Explanation**:
1. **Correlated samples**: Sequential experiences are highly correlated, causing unstable gradients
2. **Catastrophic forgetting**: Without replay, agent forgets how to handle old states
3. **Loss of diversity**: Missing rare but important transitions
4. **Breaks DQN's design**: DQN assumes i.i.d. samples from replay buffer

This is why replay buffers are crucial for DQN's stability.

**Common pitfall**: Thinking "fresh data is better." For off-policy methods, diverse old data is often more valuable than correlated new data.
</details>

<details markdown="1">
<summary><strong>Q6 (Conceptual):</strong> Is PPO on-policy or off-policy? Why?</summary>

**Answer**: PPO is on-policy, but it reuses data within a "trust region" by clipping the importance ratio.

**Explanation**:
- PPO collects data with current policy $\pi_\theta$
- Updates policy while constraining how much it changes
- The clipping mechanism limits $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ to $[1-\epsilon, 1+\epsilon]$
- After update, old data is discarded (can't reuse across epochs)

**Why not truly off-policy**:
- Data is only from recent policy versions
- Importance ratios are bounded to stay "close" to on-policy
- Cannot use arbitrary old data like Q-learning

**Common pitfall**: Thinking PPO's multiple epochs make it off-policy. It's on-policy with controlled reuse.
</details>

---

## Algorithm Sketch

### On-Policy TD (SARSA)

```
Initialize Q(s,a) arbitrarily
For each episode:
    s = initial state
    a = ε-greedy(Q, s)          # choose first action

    While not terminal:
        Take action a, observe r, s'
        a' = ε-greedy(Q, s')    # choose next action NOW
        Q(s,a) += α[r + γQ(s',a') - Q(s,a)]
        s, a = s', a'           # carry forward
```

### Off-Policy TD (Q-Learning)

```
Initialize Q(s,a) arbitrarily
For each episode:
    s = initial state

    While not terminal:
        a = ε-greedy(Q, s)
        Take action a, observe r, s'
        Q(s,a) += α[r + γ max_a' Q(s',a') - Q(s,a)]  # max, not actual
        s = s'
```

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapters 5-6
- **Precup et al. (2000)**, Eligibility Traces for Off-Policy Policy Evaluation
- **Munos et al. (2016)**, Safe and Efficient Off-Policy Reinforcement Learning

**What to memorize for interviews**: On/off-policy definitions, SARSA vs Q-learning update rules, why Q-learning enables replay buffers, importance sampling basics, cliff walking example.
