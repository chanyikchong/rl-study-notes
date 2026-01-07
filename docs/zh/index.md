# 强化学习学习笔记

欢迎使用**强化学习学习笔记** — 一份全面、面试就绪的RL概念、数学和实践实现资源。

## 内容概览

- **严谨的数学基础**: 关键方程的清晰推导与直观解释
- **面试导向内容**: 每个主题突出需要记忆的要点和常见面试问题
- **实用代码示例**: 主要算法的可运行实现
- **互动测验**: 通过点击显示的问答测试你的理解

## 主题覆盖

### 基础知识
- [MDP基础](fundamentals/mdp.md) — 状态、动作、奖励、转移
- [策略与价值函数](fundamentals/policy-value.md) — \(\pi(a|s)\), \(V^\pi(s)\), \(Q^\pi(s,a)\)
- [贝尔曼方程](fundamentals/bellman.md) — RL算法的基础

### 动态规划
- [策略评估](dp/policy-evaluation.md) — 计算 \(V^\pi\)
- [策略迭代](dp/policy-iteration.md) — 交替评估与改进
- [值迭代](dp/value-iteration.md) — 直接计算最优价值

### 无模型方法
- [蒙特卡洛方法](mc/monte-carlo.md) — 从完整轨迹学习
- [SARSA](td/sarsa.md) — 在策略TD控制
- [Q学习](td/q-learning.md) — 离策略TD控制
- [期望SARSA](td/expected-sarsa.md) — TD方差减少

### 函数逼近
- [线性方法](fa/linear.md) — 基于特征的价值逼近
- [神经网络](fa/neural.md) — 深度函数逼近

### 深度强化学习
- [DQN](deep/dqn.md) — 带经验回放和目标网络的深度Q网络
- [策略梯度](deep/policy-gradients.md) — REINFORCE和策略梯度定理
- [Actor-Critic](deep/actor-critic.md) — 结合价值和策略方法
- [PPO](deep/ppo.md) — 近端策略优化

### 高级主题
- [探索策略](advanced/exploration.md) — ε-贪婪, UCB, 熵奖励
- [优势估计 (GAE)](advanced/gae.md) — 优势中的偏差-方差权衡
- [稳定性问题](advanced/stability.md) — 致命三角, 发散
- [实践训练](advanced/practical.md) — 调试、种子、日志

### 面试准备
- [常见面试问题](interview/questions.md) — 快速参考与答案

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 本地启动文档
mkdocs serve

# 运行算法示例
python -m rl_examples.run --algo q_learning
```

## 如何使用本资源

1. **学习**: 按顺序阅读主题，理解数学，运行代码示例
2. **面试准备**: 重点关注"面试摘要"和"需要记忆的内容"部分
3. **参考**: 使用搜索和交叉链接快速找到特定概念
4. **练习**: 完成每个主题末尾的测验

---

*使用页眉的语言切换器切换到 [English Version](/en/)*
