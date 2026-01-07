# 深度Q网络（DQN）

## 面试摘要

**DQN** 将Q学习与深度神经网络结合，解决高维问题（例如从像素玩Atari）。两个关键创新：**经验回放**（存储和采样过去的转移）和**目标网络**（通过使用冻结的Q作为目标来稳定训练）。第一个在Atari游戏上达到人类水平性能的算法。现代基于价值的深度RL的基础。

**需要记忆的**：DQN损失函数，回放缓冲区目的，目标网络目的，两个关键技巧。

---

## 核心定义

### DQN架构

**输入**：状态 \(s\)（例如84×84×4堆叠帧）
**输出**：所有动作 \(a \in A\) 的 \(Q(s, a)\)

```
状态 → 卷积层 → 全连接层 → 所有动作的Q值
```

### 经验回放缓冲区

存储转移 \((s, a, r, s', \text{done})\) 在缓冲区 \(D\) 中。
均匀采样小批量用于训练。

**目的**：
1. 打破时间相关性
2. 重用数据（样本效率）
3. 平滑非平稳性

### 目标网络

维护一个单独的网络 \(Q(s, a; \theta^-)\) 用于计算目标。
周期性更新：每 \(C\) 步 \(\theta^- \leftarrow \theta\)。

**目的**：稳定的回归目标（训练时不移动）。

---

## 数学与推导

### DQN损失函数

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\left[ \left( y^{DQN} - Q(s, a; \theta) \right)^2 \right]$$

其中：

$$y^{DQN} = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

### 梯度

$$\nabla_\theta L = \mathbb{E}\left[ -2 \left( y^{DQN} - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

**注意**：\(y^{DQN}\) 被视为常数（对 \(\theta^-\) 没有梯度）。

### Huber损失变体

为了减少对异常值的敏感性：

$$L_\delta(x) = \begin{cases} \frac{1}{2}x^2 & |x| \leq \delta \\ \delta(|x| - \frac{\delta}{2}) & \text{否则} \end{cases}$$

接近零时平滑，大误差时线性。

---

## 算法概述

```
算法：DQN

超参数：γ, ε, 缓冲区大小 N, 批量大小 B, 目标更新 C

1. 初始化 Q网络 θ, 目标网络 θ^- = θ
2. 初始化回放缓冲区 D（容量 N）
3. 对 episode = 1 到 M：
     s = env.reset()
     对 t = 1 到 T：
         # 动作选择
         以概率 ε：a = 随机动作
         否则：a = argmax_a Q(s, a; θ)

         # 环境步骤
         s', r, done = env.step(a)
         存储 (s, a, r, s', done) 到 D
         s = s'

         # 学习步骤
         从 D 采样 B 个转移的小批量
         对每个 (s_j, a_j, r_j, s'_j, done_j)：
             如果 done_j：
                 y_j = r_j
             否则：
                 y_j = r_j + γ max_a' Q(s'_j, a'; θ^-)

         计算损失：L = (1/B) Σ (y_j - Q(s_j, a_j; θ))²
         梯度步骤：θ ← θ - α∇_θ L

         # 目标更新
         每 C 步：θ^- ← θ

         如果 done：break
```

### 关键超参数（Atari）

| 超参数 | 值 |
|----------------|-------|
| 回放缓冲区大小 | 1M 转移 |
| 批量大小 | 32 |
| 目标更新 | 每 10K 步 |
| 学习率 | 2.5e-4 |
| ε 衰减 | 1.0 → 0.1 在 1M 步内 |
| γ | 0.99 |

---

## 常见陷阱

1. **缓冲区太小**：需要多样性；Atari用1M，简单环境用10K+。

2. **目标网络更新太频繁**：每步更新会抵消好处。尝试每1000-10000步。

3. **ε衰减太快**：早期探索不足。

4. **忘记奖励裁剪**：Atari裁剪到{-1, 0, +1}。大奖励会破坏稳定性。

5. **缺少帧堆叠**：对于Atari，堆叠4帧以捕获运动。

6. **忽略done标志**：终止状态应该有 \(y = r\)，不是 \(r + \gamma Q(s')\)。

---

## 小例子

**CartPole DQN：**

```python
# 简化伪代码
network = MLP(4, [128, 128], 2)  # 4个状态维度，2个动作
target_net = copy(network)
buffer = ReplayBuffer(10000)

for episode in range(500):
    state = env.reset()
    while not done:
        # ε-贪婪
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(network(state))

        next_state, reward, done = env.step(action)
        buffer.add(state, action, reward, next_state, done)

        # 训练
        batch = buffer.sample(32)
        targets = rewards + gamma * target_net(next_states).max(1) * (1-dones)
        loss = MSE(network(states)[actions], targets)
        optimizer.step()

        # 更新目标
        if steps % 100 == 0:
            target_net = copy(network)
```

**预期**：在约100-200个回合内解决CartPole。

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 为什么经验回放对DQN是必要的？</summary>

**答案**：两个原因：
1. **去相关**：顺序转移是相关的，违反SGD的i.i.d.假设
2. **数据效率**：每个转移被使用多次

**解释**：没有回放，梯度被最近经验偏置。网络可能"忘记"早期模式。回放维护多样数据集并允许均匀采样。

**关键洞见**：回放将在线RL转变为更接近监督学习。

**常见陷阱**：使用太小的缓冲区。需要足够多样性来覆盖状态空间。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> 如果我们不使用目标网络会发生什么？</summary>

**答案**：训练变得不稳定——我们回归的目标每次更新都变化，导致振荡或发散。

**解释**：损失是 \((y - Q(s,a;\theta))^2\) 其中 \(y = r + \gamma \max Q(s',a';\theta)\)。如果目标用 \(\theta\)，每次梯度步都改变目标。就像追逐移动的目标。

**关键洞见**：目标网络使DQN像带固定标签的监督学习。

**常见陷阱**：认为目标网络只是为了效率。它对稳定性至关重要。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 写出DQN损失函数并解释每个项。</summary>

**答案**：

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

**各项**：
- \(r + \gamma \max_{a'} Q(s', a'; \theta^-)\)：使用目标网络的TD目标
- \(Q(s, a; \theta)\)：当前Q网络预测
- \(D\)：回放缓冲区（均匀采样）
- 差的平方形成MSE损失

**关键点**：梯度只通过 \(Q(s,a;\theta)\)，不通过目标。

**常见陷阱**：包含通过目标的梯度——这改变了算法。
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> 什么是Double DQN，它如何修复过估计？</summary>

**答案**：Double DQN用当前网络选择动作，目标网络评估：

$$y^{DDQN} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

**解释**：标准DQN使用 \(\max_{a'} Q(s', a'; \theta^-)\)，用同一网络选择和评估。这导致过估计。Double DQN解耦：\(\theta\) 选择，\(\theta^-\) 评估。

**关键洞见**：即使选择过于乐观，评估使用不同参数，减少偏差。

**常见陷阱**：认为需要两个独立网络。只需不同地使用 \(\theta\) 和 \(\theta^-\)。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> 你的DQN最初学习但然后性能下降。发生了什么？</summary>

**答案**：可能的原因：
1. **过拟合**：回放缓冲区在策略变化时变得陈旧
2. **灾难性遗忘**：网络忘记早期经验
3. **ε衰减太多**：卡在利用次优策略
4. **目标网络滞后**：目标太过时

**调试**：
- 监控Q值大小（应该稳定，不爆炸）
- 检查回放缓冲区多样性
- 尝试更慢的ε衰减
- 更大的回放缓冲区

**常见陷阱**：不监控训练曲线。绘制Q值、损失、回合奖励。
</details>

---

## 参考文献

- **Mnih et al. (2015)**, 通过深度强化学习实现人类水平控制（Nature）
- **Van Hasselt et al. (2016)**, 带双Q学习的深度RL
- **Schaul et al. (2016)**, 优先经验回放
- **Wang et al. (2016)**, 对决网络架构

**面试需要记忆的**：DQN损失，经验回放目的，目标网络目的，Double DQN，关键超参数。

**代码示例**：[dqn.py](../../../rl_examples/algorithms/dqn.py)
