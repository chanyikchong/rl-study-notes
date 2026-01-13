# Reinforcement Learning Study Notes

Welcome to the **Reinforcement Learning Study Notes** — a comprehensive, interview-ready resource for mastering RL concepts, mathematics, and practical implementations.

## What You'll Find Here

- **Rigorous Mathematical Foundations**: Clear derivations of key equations with intuitive explanations
- **Interview-Focused Content**: Each topic highlights what you need to memorize and common interview questions
- **Practical Code Examples**: Runnable implementations of major algorithms
- **Interactive Quizzes**: Test your understanding with click-to-reveal Q&A

---

## Learning Roadmap

Follow this path for the best learning experience:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REINFORCEMENT LEARNING                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: FOUNDATIONS                                                        │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │ MDP Basics  │→ │ Policy & Value  │→ │   Bellman   │→ │ Computing V/Q  │  │
│  └─────────────┘  └─────────────────┘  └─────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│  STEP 2A: MODEL-BASED (DP)      │   │  STEP 2B: MODEL-FREE                │
│  ┌───────────────────────────┐  │   │  ┌───────────────────────────────┐  │
│  │ Policy Eval → Policy Iter │  │   │  │ Monte Carlo (full episodes)   │  │
│  │ → Value Iteration         │  │   │  │ TD Learning (bootstrapping)   │  │
│  └───────────────────────────┘  │   │  │ SARSA / Q-Learning / Exp.SARSA│  │
│  (requires transition model)    │   │  └───────────────────────────────┘  │
└─────────────────────────────────┘   │  (learns from experience only)      │
                                      │                                     │
                                      │  ┌───────────────────────────────┐  │
                                      │  │ On-Policy vs Off-Policy       │  │
                                      │  └───────────────────────────────┘  │
                                      └─────────────────────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: SCALING UP (Function Approximation)                                │
│  ┌──────────────────┐     ┌──────────────────┐                              │
│  │  Linear Methods  │ →   │ Neural Networks  │                              │
│  └──────────────────┘     └──────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│  STEP 4A: VALUE-BASED DEEP RL   │   │  STEP 4B: POLICY-BASED DEEP RL      │
│  ┌───────────────────────────┐  │   │  ┌───────────────────────────────┐  │
│  │ DQN                       │  │   │  │ Policy Gradients (REINFORCE)  │  │
│  │ + Experience Replay       │  │   │  │ ↓                             │  │
│  │ + Target Network          │  │   │  │ Actor-Critic (A2C/A3C)        │  │
│  └───────────────────────────┘  │   │  │ ↓                             │  │
│  (off-policy, discrete actions) │   │  │ PPO (stable, practical)       │  │
└─────────────────────────────────┘   │  └───────────────────────────────┘  │
                                      │  (on-policy, continuous actions)    │
                                      └─────────────────────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: ADVANCED TOPICS                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Exploration │  │    GAE      │  │  Stability  │  │ Practical Training  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Topics by Section

### 1. Fundamentals (Start Here!)

| Topic | What You'll Learn | Key Concepts |
|-------|-------------------|--------------|
| [MDP Basics](fundamentals/mdp.md) | The mathematical framework for RL | States, actions, rewards, transitions, return |
| [Policy and Value Functions](fundamentals/policy-value.md) | How to evaluate and compare behaviors | $\pi(a|s)$, $V^\pi(s)$, $Q^\pi(s,a)$ |
| [Bellman Equations](fundamentals/bellman.md) | The recursive structure of value | Bellman expectation, Bellman optimality |
| [Computing Value Functions](fundamentals/computing-values.md) | **How RL actually computes V, Q, A** | DP vs MC vs TD, Advantage function |
| [On-Policy vs Off-Policy](fundamentals/on-off-policy.md) | Two paradigms of learning | Behavior policy, target policy, importance sampling |

### 2. Dynamic Programming (Model-Based)

When you have the transition model $P(s'|s,a)$:

| Topic | What You'll Learn | Key Concepts |
|-------|-------------------|--------------|
| [Policy Evaluation](dp/policy-evaluation.md) | Compute $V^\pi$ for a fixed policy | Iterative update, convergence |
| [Policy Iteration](dp/policy-iteration.md) | Find optimal policy by alternating | Evaluate → Improve → Repeat |
| [Value Iteration](dp/value-iteration.md) | Direct computation of $V^*$ | Bellman optimality backup |

### 3. Model-Free Tabular Methods

When you only have experience (no model):

| Topic | What You'll Learn | Key Concepts |
|-------|-------------------|--------------|
| [Monte Carlo Methods](mc/monte-carlo.md) | Learn from complete episodes | First-visit MC, exploring starts |
| [SARSA](td/sarsa.md) | On-policy TD control | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$ |
| [Q-Learning](td/q-learning.md) | Off-policy TD control | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| [Expected SARSA](td/expected-sarsa.md) | Variance reduction | Uses expected value over next actions |

### 4. Function Approximation

Scaling beyond tabular (large/continuous state spaces):

| Topic | What You'll Learn | Key Concepts |
|-------|-------------------|--------------|
| [Linear Methods](fa/linear.md) | Feature-based approximation | $\hat{V}(s) = \mathbf{w}^\top \phi(s)$ |
| [Neural Networks](fa/neural.md) | Deep function approximation | Non-linear features, training challenges |

### 5. Deep Reinforcement Learning

Modern RL algorithms:

| Topic | What You'll Learn | Approach |
|-------|-------------------|----------|
| [DQN](deep/dqn.md) | Deep Q-Networks | **Value-based**, off-policy, discrete actions |
| [Policy Gradients](deep/policy-gradients.md) | REINFORCE algorithm | **Policy-based**, on-policy, high variance |
| [Actor-Critic](deep/actor-critic.md) | A2C, A3C architectures | **Hybrid**: policy (actor) + value (critic) |
| [PPO](deep/ppo.md) | Proximal Policy Optimization | **State-of-the-art** for many tasks |

**Progression in Deep RL:**
```
REINFORCE (high variance)
    → Actor-Critic (reduce variance with baseline)
        → A2C/A3C (advantage + parallel)
            → PPO (stable updates via clipping)
```

### 6. Advanced Topics

| Topic | What You'll Learn | When You Need It |
|-------|-------------------|------------------|
| [Exploration Strategies](advanced/exploration.md) | ε-greedy, UCB, entropy bonus | Balancing exploration vs exploitation |
| [Advantage Estimation (GAE)](advanced/gae.md) | Bias-variance tradeoff | Improving policy gradient methods |
| [Stability Issues](advanced/stability.md) | Deadly triad, divergence | Debugging training failures |
| [Practical Training](advanced/practical.md) | Debugging, seeding, logging | Real-world implementation |

### 7. Interview Preparation

| Topic | What You'll Find |
|-------|------------------|
| [Common Interview Questions](interview/questions.md) | Quick reference with concise answers |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Serve documentation locally
mkdocs serve

# Run algorithm examples
python -m rl_examples.run --algo q_learning
python -m rl_examples.run --algo dqn
python -m rl_examples.run --algo ppo
```

---

## How to Use This Resource

| Goal | Recommended Approach |
|------|---------------------|
| **Learning RL from scratch** | Follow the roadmap top-to-bottom. Read each section, work through the math, run the code. |
| **Interview preparation** | Focus on "Interview Summary" and "What to Memorize" sections. Review the quiz questions. |
| **Quick reference** | Use the search bar or navigate directly to the topic you need. |
| **Understanding specific algorithm** | Read the concept page, then run the corresponding code example. |

---

## Code Examples Available

All algorithms have runnable implementations in `rl_examples/`:

| Algorithm | Command |
|-----------|---------|
| Policy Iteration | `python -m rl_examples.run --algo policy_iteration` |
| Value Iteration | `python -m rl_examples.run --algo value_iteration` |
| Monte Carlo Control | `python -m rl_examples.run --algo mc_control` |
| SARSA | `python -m rl_examples.run --algo sarsa` |
| Q-Learning | `python -m rl_examples.run --algo q_learning` |
| Expected SARSA | `python -m rl_examples.run --algo expected_sarsa` |
| DQN | `python -m rl_examples.run --algo dqn` |
| REINFORCE | `python -m rl_examples.run --algo reinforce` |
| PPO | `python -m rl_examples.run --algo ppo` |

---

*Switch to [中文版本](/zh/) using the language toggle in the header.*
