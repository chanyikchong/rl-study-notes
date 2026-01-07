# 常见面试问题

## 面试摘要

本页汇编了常见的RL面试问题及简洁答案。主题涵盖基础、算法、深度RL和实践考虑。面试前用作快速参考。

---

## 基础

<details markdown="1">
<summary><strong>问：基于模型和无模型RL有什么区别？</strong></summary>

**答案**：
- **基于模型**：学习/使用环境动态模型 P(s'|s,a) 和 R(s,a)。通过模拟轨迹规划。
- **无模型**：直接从经验学习策略/价值，不建模动态。

**权衡**：基于模型更样本高效但需要准确模型。无模型更简单但需要更多数据。
</details>

<details markdown="1">
<summary><strong>问：解释贝尔曼方程。</strong></summary>

**答案**：贝尔曼方程递归表达价值：

$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

它说：价值 = 即时奖励 + 折扣未来价值。DP、TD和Q学习的基础。
</details>

<details markdown="1">
<summary><strong>问：同策略和异策略有什么区别？</strong></summary>

**答案**：
- **同策略**：学习正在用于收集数据的策略（例如SARSA）
- **异策略**：学习与收集数据不同的策略（例如Q学习）

异策略允许从任何数据学习但有稳定性问题。同策略更稳定但样本效率更低。
</details>

<details markdown="1">
<summary><strong>问：为什么使用折扣因子γ？</strong></summary>

**答案**：
1. **数学**：确保无限时间范围内回报有限
2. **行为**：建模对即时奖励的偏好
3. **实践**：减少价值估计的方差

常见值：0.99（远视），0.9（中等），0.5（短视）。
</details>

<details markdown="1">
<summary><strong>问：什么是马尔可夫性质？</strong></summary>

**答案**：给定当前状态，未来与过去独立：

$$P(S_{t+1}|S_t, A_t, S_{t-1}, ..., S_0) = P(S_{t+1}|S_t, A_t)$$

状态包含所有相关信息。在部分可观测环境中被违反。
</details>

---

## 基于价值的方法

<details markdown="1">
<summary><strong>问：解释Q学习及其更新规则。</strong></summary>

**答案**：Q学习是异策略TD控制：

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

使用max学习最优Q不管探索策略。标准条件下收敛到Q*。
</details>

<details markdown="1">
<summary><strong>问：SARSA和Q学习有什么区别？</strong></summary>

**答案**：
- **SARSA**：\(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]\) — 使用实际采取的动作
- **Q学习**：\(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]\) — 使用max

SARSA是同策略（学习Q^π），Q学习是异策略（学习Q*）。
</details>

<details markdown="1">
<summary><strong>问：什么是致命三角？</strong></summary>

**答案**：三个元素一起可能导致发散：
1. 函数逼近
2. 自举
3. 异策略学习

每个单独没问题。一起，误差可以复合，价值发散。解决方案：目标网络，经验回放。
</details>

<details markdown="1">
<summary><strong>问：DQN如何稳定训练？</strong></summary>

**答案**：两个关键创新：
1. **经验回放**：存储和随机采样转移，打破相关性
2. **目标网络**：使用冻结的Q作为目标，防止移动目标问题

还有：奖励裁剪，帧堆叠，Huber损失。
</details>

<details markdown="1">
<summary><strong>问：什么是Double DQN，为什么需要它？</strong></summary>

**答案**：Double DQN解决过估计偏差：

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s',a';\theta); \theta^-)$$

标准Q学习对噪声估计取max倾向于过估计。Double DQN用θ选择动作，θ⁻评估。
</details>

---

## 基于策略的方法

<details markdown="1">
<summary><strong>问：什么是策略梯度定理？</strong></summary>

**答案**：

$$\nabla J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$$

期望回报的梯度等于期望得分函数乘以Q值。REINFORCE、PPO的基础。
</details>

<details markdown="1">
<summary><strong>问：为什么在策略梯度中使用基线？</strong></summary>

**答案**：减少方差而不增加偏差。

$$\nabla J = \mathbb{E}[\nabla \log \pi(a|s)(Q(s,a) - b(s))]$$

常见基线：V(s)。然后优势 A(s,a) = Q(s,a) - V(s) 将奖励中心化到零附近。
</details>

<details markdown="1">
<summary><strong>问：什么是优势函数？</strong></summary>

**答案**：

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

衡量动作a比平均好多少。正 = 比平均好，负 = 比平均差。
</details>

<details markdown="1">
<summary><strong>问：解释actor-critic方法。</strong></summary>

**答案**：结合策略梯度（actor）和价值函数（critic）：
- **Actor**：使用优势估计更新策略
- **Critic**：估计V(s)来计算优势

比REINFORCE方差低（使用TD而不是MC）。
</details>

<details markdown="1">
<summary><strong>问：PPO如何工作？</strong></summary>

**答案**：PPO通过裁剪限制策略更新：

$$L = \min(r(\theta) A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A)$$

其中r(θ) = π_θ(a|s)/π_old(a|s)。防止过大的策略变化。简单、稳定、广泛使用。
</details>

---

## 深度RL

<details markdown="1">
<summary><strong>问：为什么经验回放重要？</strong></summary>

**答案**：
1. **去相关**：打破顺序数据中的时间相关
2. **效率**：每个转移使用多次
3. **稳定性**：从多样数据更平滑学习

DQN稳定训练所必需。
</details>

<details markdown="1">
<summary><strong>问：解释目标网络。</strong></summary>

**答案**：Q网络的冻结副本用于计算TD目标：

$$y = r + \gamma \max_a Q(s', a; \theta^-)$$

周期性更新（每C步）。防止追逐移动目标。
</details>

<details markdown="1">
<summary><strong>问：什么是GAE？</strong></summary>

**答案**：广义优势估计：

$$\hat{A}^{GAE}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

在TD（λ=0，低方差）和MC（λ=1，低偏差）之间插值。典型λ=0.95。
</details>

<details markdown="1">
<summary><strong>问：如何处理连续动作空间？</strong></summary>

**答案**：选项：
1. **策略梯度**：输出高斯参数μ, σ；采样动作
2. **DDPG/TD3**：确定性策略+探索噪声
3. **SAC**：最大熵框架带软更新
4. **离散化**：粗粒度动作（损失分辨率）
</details>

<details markdown="1">
<summary><strong>问：什么是熵正则化？</strong></summary>

**答案**：将熵奖励添加到目标：

$$J(\theta) = \mathbb{E}[R] + \beta H(\pi_\theta)$$

通过惩罚确定性策略来鼓励探索。PPO、SAC中常见。
</details>

---

## 探索

<details markdown="1">
<summary><strong>问：比较探索策略。</strong></summary>

**答案**：
- **ε-贪婪**：以概率ε随机动作。简单，不优先考虑。
- **Softmax**：动作按exp(Q/τ)加权。更高价值动作更可能。
- **UCB**：不确定性奖励。有原则，需要计数。
- **熵奖励**：奖励策略中的随机性。
- **内在动机**：好奇心驱动的稀疏奖励探索。
</details>

<details markdown="1">
<summary><strong>问：什么是探索-利用权衡？</strong></summary>

**答案**：平衡：
- **利用**：使用已知好的动作（短期最优）
- **探索**：尝试不确定的动作（发现更好策略）

训练早期：更多探索。后期：更多利用。
</details>

---

## 实践

<details markdown="1">
<summary><strong>问：如何调试不学习的RL智能体？</strong></summary>

**答案**：调试层级：
1. **奖励**：打印并验证奖励信号
2. **环境**：渲染，手动交互
3. **梯度**：它们非零且合理吗？
4. **探索**：智能体访问多样状态吗？
5. **超参数**：从已知有效设置开始
6. **简单测试**：能记住单个转移吗？
</details>

<details markdown="1">
<summary><strong>问：最重要调优的超参数是什么？</strong></summary>

**答案**：优先级顺序：
1. **学习率**：最敏感
2. **网络架构**：大小和结构
3. **批量大小**：影响梯度方差
4. **折扣γ**：规划视野
5. **探索参数**：ε计划，熵系数
</details>

<details markdown="1">
<summary><strong>问：如何确保可重复性？</strong></summary>

**答案**：
1. 设置随机种子（numpy, torch, env）
2. 使用确定性操作
3. 记录所有超参数
4. 代码和配置的版本控制
5. 运行多个种子（最少3-5个）
6. 报告均值±标准误差
</details>

<details markdown="1">
<summary><strong>问：什么是奖励塑形？</strong></summary>

**答案**：添加中间奖励引导学习：

$$r' = r + F(s, s')$$

可以加速学习但如果做错可能改变最优策略。基于势的塑形保持最优性。
</details>

<details markdown="1">
<summary><strong>问：如何处理稀疏奖励？</strong></summary>

**答案**：策略：
1. **奖励塑形**：添加中间奖励
2. **课程学习**：从简单任务开始
3. **后见经验回放**：重新标记目标
4. **内在动机**：好奇心，基于计数
5. **演示/模仿**：从专家学习
</details>

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 面试时被问"你最喜欢的RL算法是什么，为什么？"如何回答？</summary>

**答案**：好的结构：
1. 命名算法（例如PPO）
2. 解释为什么："稳定，实现简单，跨多领域有效"
3. 展示对权衡的理解："不如SAC样本高效，但更稳健"
4. 提及何时选择其他

**例子**："我经常使用PPO因为它稳定，不需要仔细超参数调优就能在连续控制上工作良好。对于连续动作的样本效率，我会考虑SAC。对于高维观测的离散动作，DQN变体。"
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> "从头走我过训练DQN的过程。"</summary>

**答案**：结构化回答：
1. **环境**：状态/动作空间，奖励结构
2. **网络**：图像用CNN，向量用MLP → 所有动作的Q值
3. **回放缓冲区**：存储(s, a, r, s', done)元组
4. **目标网络**：冻结副本，每C步更新
5. **训练循环**：ε-贪婪动作 → step → 存储 → 采样批量 → 计算目标 → 梯度步
6. **评估**：周期性运行贪婪策略

关键点要提：回放缓冲区打破相关性，目标网络稳定训练。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> "推导策略梯度。"</summary>

**答案**：展示关键步骤：
1. \(J(\theta) = \mathbb{E}_\tau[R(\tau)]\)
2. \(\nabla J = \nabla \mathbb{E}[R] = \sum_\tau R(\tau) \nabla P(\tau|\theta)\)
3. 对数导数技巧：\(\nabla P = P \nabla \log P\)
4. \(\nabla J = \mathbb{E}[\nabla \log P(\tau|\theta) R(\tau)]\)
5. 只有π依赖θ：\(\nabla \log P(\tau) = \sum_t \nabla \log \pi(a_t|s_t)\)

最终：\(\nabla J = \mathbb{E}[\sum_t \nabla \log \pi(a_t|s_t) G_t]\)
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> "解释TD vs MC的偏差-方差权衡。"</summary>

**答案**：
- **MC**：使用实际回报 \(G_t\)。零偏差（定义上），高方差（许多随机奖励的总和）
- **TD**：使用 \(r + \gamma V(s')\)。有偏（如果V错误），低方差（一个奖励）

GAE插值：λ控制权衡。λ=0是TD，λ=1是MC。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> "你如何处理新的RL问题？"</summary>

**答案**：框架：
1. **理解问题**：状态/动作空间，奖励，回合结构
2. **从简单开始**：如果可能用表格，随机基线
3. **选择算法**：离散用DQN，连续用PPO/SAC
4. **已知超参数**：从论文/有效设置开始
5. **监控所有东西**：回报，Q值，梯度，熵
6. **系统调试**：一次改一个
7. **扩展**：更多计算，超参数调优
</details>

---

## 参考文献

- **Sutton & Barto**, 强化学习：导论
- **Spinning Up in Deep RL**（OpenAI）
- **David Silver RL课程**（DeepMind）
- **Berkeley深度RL课程**（CS 285）

**面试技巧**：对你不知道的诚实。说"我没实现过那个"比假装理解好。
