# Practical Training Pipeline

## Interview Summary

Real-world RL training requires more than algorithms: **seeding** for reproducibility, **logging** for debugging, **evaluation** separate from training, **hyperparameter tuning**, and extensive **debugging skills**. Common issues: reward bugs, environment errors, wrong hyperparameters, implementation bugs. Most RL failures are engineering problems, not algorithmic ones.

**What to memorize**: Reproducibility practices, what to log, common debugging patterns, hyperparameter tuning strategies.

---

## Core Definitions

### Training vs Evaluation

- **Training**: Agent learns (explores, updates)
- **Evaluation**: Measure true performance (greedy, no updates)

**Why separate?** Training performance includes exploration (ε-greedy), which lowers returns. Evaluation shows actual learned behavior.

### Reproducibility

Achieving same results with same code and seed:
1. Random seeds (numpy, torch, env)
2. Deterministic operations
3. Version control for code and data
4. Logging all hyperparameters

### Key Metrics to Track

| Metric | What It Shows |
|--------|--------------|
| Episode return | Agent performance |
| Episode length | Learning to survive |
| Q-value magnitude | Stability (shouldn't explode) |
| Policy entropy | Exploration level |
| Loss values | Learning progress |
| Gradient norms | Training stability |

---

## Math and Derivations

### Seeding for Reproducibility

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (slower):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Environment seeding
env = gym.make("CartPole-v1")
env.seed(seed)
env.action_space.seed(seed)
```

### Evaluation Protocol

```python
def evaluate(agent, env, n_episodes=10, seed=42):
    """Evaluate agent without exploration or learning."""
    env.seed(seed)
    returns = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_return = 0
        done = False

        while not done:
            action = agent.get_action(state, greedy=True)  # No exploration
            state, reward, done, info = env.step(action)
            episode_return += reward

        returns.append(episode_return)

    return np.mean(returns), np.std(returns)
```

### Learning Curves

Plot with confidence intervals:
```
Mean return ± std/sqrt(n_seeds) across multiple seeds
```

Minimum 3 seeds, preferably 5-10 for publishing.

---

## Algorithm Sketch

### Standard Training Loop

```python
# Setup
set_seed(args.seed)
env = make_env(args.env)
agent = make_agent(args)
logger = setup_logging(args)

# Training
for step in range(args.total_steps):
    # Collect experience
    action = agent.get_action(state, explore=True)
    next_state, reward, done, info = env.step(action)
    agent.buffer.add(state, action, reward, next_state, done)

    # Update agent
    if step > args.warmup_steps:
        metrics = agent.update()
        logger.log(metrics)

    # Periodic evaluation
    if step % args.eval_freq == 0:
        eval_return, eval_std = evaluate(agent, eval_env)
        logger.log({"eval_return": eval_return, "eval_std": eval_std})
        logger.save_checkpoint(agent, step)

    # Reset if done
    if done:
        state = env.reset()
    else:
        state = next_state
```

### Hyperparameter Tuning Order

1. **Learning rate**: Most sensitive. Try 1e-5, 1e-4, 1e-3
2. **Network size**: Start small (2 layers, 64-256 units)
3. **Batch size**: 32-256 typical
4. **Discount γ**: 0.99 default, 0.9-0.999 for tuning
5. **Exploration**: ε schedule, entropy coefficient
6. **Algorithm-specific**: Replay buffer, target update, etc.

### Debugging Hierarchy

```
1. Check reward signal
   → Print rewards, verify they match expectations
   → Common bug: wrong sign, missing terminal reward

2. Check environment
   → Render and interact manually
   → Verify state space, action effects

3. Check learning basics
   → Is loss decreasing? Are gradients flowing?
   → Can agent memorize a single transition?

4. Check exploration
   → Is agent exploring enough states?
   → Print state visitation statistics

5. Check hyperparameters
   → Start with known-working settings
   → Tune one parameter at a time
```

---

## Common Pitfalls

1. **Not setting seeds**: Results vary wildly between runs. Always seed.

2. **Evaluating during training**: Training returns are noisy (exploration). Use separate greedy evaluation.

3. **Wrong reward function**: Most common bug. Verify rewards manually.

4. **Terminal state handling**: Forgetting to set V(terminal)=0 breaks value learning.

5. **Hyperparameter cargo culting**: Paper hyperparameters may not work for your setup. Tune for your environment.

6. **Not logging enough**: When things fail, you need data. Log everything.

7. **Changing multiple things at once**: Debug one change at a time.

---

## Mini Example

**CartPole Training Example:**

```python
# Verified working hyperparameters for CartPole-v1
config = {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 10000,
    "buffer_size": 10000,
    "target_update": 100,
    "hidden_size": 128,
    "total_steps": 50000,
    "eval_freq": 1000,
}

# Expected results:
# - Solves (return > 195) in ~10K-20K steps
# - Q-values stabilize around 20-50
# - Epsilon decays from 1.0 to 0.01 over 10K steps
```

**When it doesn't work, check:**
1. Is reward being received? (Print to verify)
2. Is exploration happening? (Print epsilon, action distribution)
3. Are Q-values reasonable? (Should be ~20-50 for solved CartPole)

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why must evaluation be separate from training?</summary>

**Answer**: Training includes exploration (random actions) which hurts measured performance. Evaluation should use greedy policy to measure true capability.

**Explanation**: With ε-greedy at ε=0.1, 10% of actions are random → lower return. Evaluation with ε=0 shows what the learned policy actually achieves.

**Also**: Training updates the network, creating non-stationarity. Evaluation is a snapshot of current policy.

**Common pitfall**: Reporting training returns as results. Always report greedy evaluation.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why is RL debugging harder than supervised learning?</summary>

**Answer**: Several reasons:
1. **No ground truth**: Can't compare predictions to labels
2. **Delayed signal**: Reward may come much later than action
3. **Non-stationary**: Policy changes → data distribution changes
4. **Exploration confounds**: Bad results could be exploration, not bad policy
5. **Many failure modes**: Env bugs, reward bugs, algorithm bugs, hyperparameters

**Explanation**: In supervised learning, if loss doesn't decrease, something is wrong. In RL, loss might decrease but policy might be bad (overfitting to buffer) or increase but learning is happening.

**Common pitfall**: Assuming loss behavior indicates learning.
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> How do you know if Q-values are reasonable?</summary>

**Answer**: Sanity check with discounted reward sum:

For CartPole (max 500 steps, reward=1/step, γ=0.99):

$$Q_{max} \approx \frac{1}{1-\gamma} = \frac{1}{0.01} = 100$$

For episodic task ending at T steps:

$$Q_{max} \approx \sum_{t=0}^{T} \gamma^t r_{max}$$

**If Q >> expected max**: Overestimation, possible divergence
**If Q ≈ 0 always**: Not learning (check rewards, gradients)

**Common pitfall**: Not checking Q-value magnitudes. Easy diagnostic often missed.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> How many seeds should you run for reliable results?</summary>

**Answer**: Minimum 3, preferably 5-10 for publication-quality results.

**Explanation**: RL has high variance across seeds due to:
- Random initialization
- Exploration randomness
- Environment stochasticity

Standard error decreases with √n seeds. With 5 seeds, you can estimate mean ± 0.45σ (at 95% confidence).

**Best practice**: Report mean ± stderr, or show all individual runs in plots.

**Common pitfall**: Reporting single-seed results (cherry-picking best seed).
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your agent learns in one environment but fails in another. What to try?</summary>

**Answer**: Transfer checklist:

1. **Scale differences**: Normalize observations and rewards
   ```python
   obs = (obs - obs_mean) / obs_std
   reward = np.clip(reward, -10, 10)
   ```

2. **Action space**: Different ranges need different scaling

3. **Episode length**: Longer episodes need smaller learning rate, larger buffer

4. **Reward sparsity**: Sparse rewards need more exploration (curiosity, higher ε)

5. **State complexity**: More complex states need bigger networks

6. **Hyperparameter sensitivity**: Re-tune for new environment

**Explanation**: RL algorithms are surprisingly environment-specific. What works for CartPole may fail for Atari.

**Common pitfall**: Assuming one set of hyperparameters works everywhere.
</details>

---

## References

- **Henderson et al. (2018)**, Deep RL That Matters
- **Engstrom et al. (2020)**, Implementation Matters in Deep RL
- **Andrychowicz et al. (2020)**, What Matters in On-Policy RL
- **Islam et al. (2017)**, Reproducibility of Benchmarked Deep RL Tasks

**What to memorize for interviews**: Seeding practices, evaluation vs training, debugging hierarchy, hyperparameter tuning order, common issues.
