# Reinforcement Learning Study Notes

Welcome to the **Reinforcement Learning Study Notes** — a comprehensive, interview-ready resource for mastering RL concepts, mathematics, and practical implementations.

## What You'll Find Here

- **Rigorous Mathematical Foundations**: Clear derivations of key equations with intuitive explanations
- **Interview-Focused Content**: Each topic highlights what you need to memorize and common interview questions
- **Practical Code Examples**: Runnable implementations of major algorithms
- **Interactive Quizzes**: Test your understanding with click-to-reveal Q&A

## Topics Covered

### Fundamentals
- [MDP Basics](fundamentals/mdp.md) — States, actions, rewards, transitions
- [Policy and Value Functions](fundamentals/policy-value.md) — \(\pi(a|s)\), \(V^\pi(s)\), \(Q^\pi(s,a)\)
- [Bellman Equations](fundamentals/bellman.md) — The foundation of RL algorithms

### Dynamic Programming
- [Policy Evaluation](dp/policy-evaluation.md) — Computing \(V^\pi\)
- [Policy Iteration](dp/policy-iteration.md) — Alternating evaluation and improvement
- [Value Iteration](dp/value-iteration.md) — Direct optimal value computation

### Model-Free Methods
- [Monte Carlo Methods](mc/monte-carlo.md) — Learning from complete episodes
- [SARSA](td/sarsa.md) — On-policy TD control
- [Q-Learning](td/q-learning.md) — Off-policy TD control
- [Expected SARSA](td/expected-sarsa.md) — Variance reduction in TD

### Function Approximation
- [Linear Methods](fa/linear.md) — Feature-based value approximation
- [Neural Networks](fa/neural.md) — Deep function approximation

### Deep Reinforcement Learning
- [DQN](deep/dqn.md) — Deep Q-Networks with experience replay and target networks
- [Policy Gradients](deep/policy-gradients.md) — REINFORCE and the policy gradient theorem
- [Actor-Critic](deep/actor-critic.md) — Combining value and policy methods
- [PPO](deep/ppo.md) — Proximal Policy Optimization

### Advanced Topics
- [Exploration Strategies](advanced/exploration.md) — ε-greedy, UCB, entropy bonus
- [Advantage Estimation (GAE)](advanced/gae.md) — Bias-variance tradeoff in advantage
- [Stability Issues](advanced/stability.md) — Deadly triad, divergence
- [Practical Training](advanced/practical.md) — Debugging, seeding, logging

### Interview Prep
- [Common Interview Questions](interview/questions.md) — Quick reference with answers

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Serve documentation locally
mkdocs serve

# Run algorithm examples
python -m rl_examples.run --algo q_learning
```

## How to Use This Resource

1. **For Learning**: Read topics in order, work through the math, run the code examples
2. **For Interview Prep**: Focus on "Interview Summary" and "What to Memorize" sections
3. **For Reference**: Use search and cross-links to quickly find specific concepts
4. **For Practice**: Complete the quiz at the end of each topic

---

*Switch to [中文版本](/zh/) using the language toggle in the header.*
