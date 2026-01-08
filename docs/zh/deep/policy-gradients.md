# 策略梯度

## 面试摘要

**策略梯度方法** 通过对期望回报的梯度上升直接优化策略 $\pi_\theta(a|s)$。关键结果是**策略梯度定理**：$\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$。**REINFORCE** 是最简单的算法——用采样回报作为Q估计。高方差是主要挑战；基线可以减少它。

**需要记忆的**：策略梯度定理，REINFORCE更新，对数导数技巧，为什么基线减少方差而不增加偏差。

---

## 设计动机：为什么策略梯度存在

### 基于价值方法的局限性

Q学习和DQN学习价值函数，然后贪婪行动：

$$\pi(s) = \arg\max_a Q(s, a)$$

**问题1：连续动作**
- 如果动作空间是连续的（例如机器人关节力矩）怎么办？
- 在无限动作上取 $\arg\max$ 是难以处理的！

**问题2：随机策略**
- 有时最优行为是随机的（例如石头剪刀布）
- 基于价值的方法给出确定性策略

**问题3：间接优化**
- 我们关心的是累积奖励，但我们在学习Q
- 没有保证更好的Q → 更好的策略

### 策略梯度的思想

> **关键洞见**：为什么不直接优化策略？

不是：
```
学习 Q(s,a) → 从Q导出策略
```

而是：
```
参数化策略 π(a|s; θ) → 直接为奖励优化 θ
```

**优势：**
- 自然地处理连续动作
- 可以表示随机策略
- 直接优化我们关心的东西

### 挑战：如何计算梯度？

目标是：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

**问题**：期望是对从 $\pi_\theta$ 采样的轨迹取的。如何对这个采样过程求导？

**解决方案**：**策略梯度定理**给我们一个方法！

---

## 核心定义

### 策略参数化

随机策略：$\pi_\theta(a|s) = P(A=a|S=s; \theta)$

**常见形式：**

| 动作类型 | 策略形式 | 输出 |
|-------------|-------------|--------|
| 离散 | Softmax | $\pi(a|s) = \frac{e^{f_\theta(s,a)}}{\sum_{a'} e^{f_\theta(s,a')}}$ |
| 连续 | 高斯 | $\pi(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$ |

### 目标：期望回报

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

**目标**：找到最大化 $J(\theta)$ 的 $\theta$。

### 策略梯度定理

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a) \right]$$

**用语言说**：期望回报的梯度等于：
- "得分函数" $\nabla \log \pi$（哪个方向增加动作概率）
- 乘以动作的价值 $Q(s,a)$（这个动作有多好）

**直觉**：增加好动作的概率，减少坏动作的概率。

---

## 数学与推导

### 推导策略梯度（重要！）

这个推导在面试中经常被问到。

**步骤1**：将目标写成期望

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_\tau P(\tau|\theta) R(\tau)$$

**步骤2**：取梯度

$$\nabla_\theta J = \sum_\tau \nabla_\theta P(\tau|\theta) \cdot R(\tau)$$

**步骤3**：对数导数技巧

$$\nabla P = P \cdot \nabla \log P$$

因此：

$$\nabla_\theta J = \sum_\tau P(\tau|\theta) \cdot \nabla_\theta \log P(\tau|\theta) \cdot R(\tau)$$

$$= \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau|\theta) \cdot R(\tau)]$$

**步骤4**：展开轨迹的对数概率

$$\log P(\tau|\theta) = \log p(s_0) + \sum_{t=0}^{T-1} \left[ \log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t) \right]$$

只有 $\pi_\theta$ 依赖于 $\theta$：

$$\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**步骤5**：最终形式（带因果性）

$$\nabla_\theta J = \mathbb{E}_\tau\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$ 是**从t开始的回报**（不是总回报！）。

**为什么是从t开始的回报？** 时刻t的动作只影响从t开始的奖励（因果性）。

### 对数导数技巧解释

$$\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \cdot \nabla_\theta \log P(\tau|\theta)$$

**为什么有效：**

$$\nabla \log f = \frac{\nabla f}{f}$$

$$\therefore \nabla f = f \cdot \nabla \log f$$

**为什么有用**：将概率的梯度转换为对数概率的梯度，我们可以为 $\pi_\theta$ 计算这个！

### 基线减法（方差减少）

$$\nabla_\theta J = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot (Q^{\pi}(s,a) - b(s)) \right]$$

任何不依赖于 $a$ 的基线 $b(s)$ 都可以减去，**而不改变期望梯度**。

**证明基线不增加偏差：**

$$\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$$

$$= b(s) \sum_a \nabla_\theta \pi_\theta(a|s) = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0$$

**最佳基线**：$b(s) = V^\pi(s)$，给出**优势**：

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

**直觉**：优势告诉我们"在状态 $s$ 中，动作 $a$ 比平均动作好多少？"

---

## 算法概述

### REINFORCE（蒙特卡洛策略梯度）

```
算法：REINFORCE

输入：学习率 α，折扣 γ
初始化：策略参数 θ

对每个回合：
    # 收集轨迹
    τ = []
    s = env.reset()
    while not done:
        a ~ π_θ(·|s)                    # 采样动作
        s', r, done = env.step(a)
        τ.append((s, a, r))
        s = s'

    # 计算从t开始的回报
    G = 0
    returns = []
    for (s, a, r) in reversed(τ):
        G = r + γ * G
        returns.insert(0, G)

    # 策略梯度更新
    for (s, a, r), G in zip(τ, returns):
        θ ← θ + α · G · ∇_θ log π_θ(a|s)

返回 θ
```

### 带基线的REINFORCE

```
算法：带基线的REINFORCE

初始化：θ（策略），φ（价值函数）

对每个回合：
    使用 π_θ 收集轨迹 τ
    计算回报 G_t

    对每个时间步 t：
        # 优势 = 回报 - 基线
        A_t = G_t - V_φ(s_t)

        # 更新价值函数（基线）
        φ ← φ + α_φ · A_t · ∇_φ V_φ(s_t)

        # 更新策略
        θ ← θ + α_θ · A_t · ∇_θ log π_θ(a_t|s_t)

返回 θ
```

### 关键设计选择

| 选择 | 目的 |
|--------|---------|
| 从t开始的回报 $G_t$ | 因果性——只有未来奖励重要 |
| 基线 $V(s)$ | 不增加偏差的方差减少 |
| 归一化回报 | 稳定性——$(G - \mu) / \sigma$ |
| 熵奖励 | 防止过早收敛 |

---

## 基于价值 vs 基于策略：何时使用哪个？

| 方面 | 基于价值（DQN） | 基于策略（PG） |
|--------|-------------------|-------------------|
| 动作空间 | 仅离散 | 任何（连续！） |
| 策略类型 | 确定性 | 随机 |
| 样本效率 | 高（回放） | 低（同策略） |
| 稳定性 | 可能发散 | 更稳定 |
| 收敛 | 到Q的局部最优 | 到J的局部最优 |
| 最适合 | 离散，复杂价值 | 连续，简单价值 |

**经验法则：**
- 离散动作，需要样本效率 → DQN
- 连续动作，或需要随机策略 → 策略梯度

---

## 常见陷阱

1. **高方差**：不带基线的REINFORCE方差非常高。总是使用基线！

2. **使用总回报而不是从t开始的回报**：在梯度中包含过去奖励是无意义的方差。

3. **样本效率低**：每个样本使用一次。这是同策略方法的根本特点。

4. **学习率太高**：策略梯度很敏感。从1e-4或更低开始。

5. **熵崩溃**：策略太快变得确定性。添加熵奖励：$L = L_{PG} + \beta H(\pi)$。

6. **奖励缩放**：大奖励 → 大梯度 → 不稳定。归一化回报！

7. **忘记采样动作**：必须从 $\pi_\theta$ 采样，不是取argmax！

---

## 小例子

**用REINFORCE的CartPole：**

```python
# 策略网络：状态 → 动作概率
policy = MLP(input_dim=4, hidden=[32], output_dim=2)
optimizer = Adam(policy.parameters(), lr=1e-3)

for episode in range(1000):
    states, actions, rewards = [], [], []

    # 收集回合
    state = env.reset()
    done = False
    while not done:
        # 从策略采样动作
        probs = softmax(policy(state))
        action = sample_categorical(probs)

        next_state, reward, done = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # 计算从t开始的回报
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    # 归一化回报（方差减少）
    returns = (returns - mean(returns)) / (std(returns) + 1e-8)

    # 策略梯度更新
    loss = 0
    for s, a, G in zip(states, actions, returns):
        log_prob = log(softmax(policy(s))[a])
        loss -= log_prob * G  # 负号因为优化器最小化

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 为什么使用策略梯度而不是基于价值的方法？</summary>

**答案**：策略梯度有关键优势：
1. **连续动作**：不需要解决 $\arg\max_a Q(s,a)$
2. **随机策略**：可以表示混合策略
3. **直接优化**：直接优化期望回报
4. **简单性**：没有目标网络，没有回放缓冲区

**何时偏好基于价值的**：需要样本效率的离散动作（带回放的DQN）。

**关键洞见**：策略梯度是不需要离散化的连续控制的唯一选择。

**常见陷阱**：对DQN更高效的简单离散问题使用策略梯度。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> 解释对数导数技巧以及为什么它是必要的。</summary>

**答案**：对数导数技巧转换：

$$\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] \text{ (困难)} \rightarrow \mathbb{E}_{x \sim p_\theta}[\nabla_\theta \log p_\theta(x) \cdot f(x)] \text{ (容易)}$$

**为什么必要：**
- 我们不能对采样过程求导
- 但我们可以为任何采样的动作计算 $\nabla \log \pi_\theta(a|s)$
- 允许梯度的蒙特卡洛估计

**技巧：**

$$\nabla p = p \cdot \nabla \log p$$

**常见陷阱**：忘记这是策略梯度有效的核心。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 证明减去状态相关的基线不增加偏差。</summary>

**答案**：我们需要证明 $\mathbb{E}_a[\nabla \log \pi(a|s) \cdot b(s)] = 0$：

$$\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)]$$

$$= b(s) \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$$

$$= b(s) \sum_a \pi_\theta(a|s) \cdot \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

$$= b(s) \sum_a \nabla_\theta \pi_\theta(a|s)$$

$$= b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s)$$

$$= b(s) \cdot \nabla_\theta 1 = 0$$

**关键洞见**：基线因子提出并乘以零，因为概率和为1。

**常见陷阱**：使用依赖于 $a$ 的基线——这确实增加偏差！
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> 什么是优势函数，为什么它优于Q？</summary>

**答案**：$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$

**为什么优于Q：**

| 使用Q | 使用优势 |
|---------|--------------------|
| 所有Q值可能是大正数 | 以零为中心 |
| 好动作：正，坏动作：也是正！ | 好：正，坏：负 |
| 高方差 | 更低方差 |

**直觉**："这个动作比平均动作好多少？"
- $A > 0$：比平均好 → 增加概率
- $A < 0$：比平均差 → 减少概率
- $A = 0$：平均 → 不变

**常见陷阱**：不使用基线。原始Q值有效但方差高得多。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> 你的策略梯度训练方差非常高。怎么办？</summary>

**答案**：方差减少技术：

1. **添加基线**：使用 $V(s)$ 计算优势。最重要！
2. **归一化回报**：每批 $(G - \mu) / \sigma$
3. **更大批量**：对更多轨迹取平均
4. **使用从t开始的回报**：不是总回合回报
5. **GAE**：广义优势估计（后续主题）
6. **熵正则化**：防止策略崩溃
7. **降低学习率**：更稳定的更新

**优先顺序**：基线 > 归一化 > 批量大小 > 其他。

**常见陷阱**：仅通过降低学习率来减少方差。解决根源！
</details>

<details markdown="1">
<summary><strong>问题6（概念）：</strong> 为什么REINFORCE是同策略的？这对样本效率意味着什么？</summary>

**答案**：**同策略** 意味着我们只能从当前策略 $\pi_\theta$ 收集的数据中学习。

**为什么是同策略：**
- 策略梯度是 $\mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta \cdot G]$
- 期望是在 $\pi_\theta$ 下的
- 来自旧 $\pi_{\theta_{old}}$ 的数据不能给当前 $\pi_\theta$ 有效的梯度

**样本效率影响：**
- 每个转移用于一次梯度更新，然后丢弃
- 不能使用经验回放（数据会是异策略的）
- 比DQN样本效率低得多

**解决方案**：Actor-critic（减少方差），PPO/TRPO（带修正重用样本）。

**常见陷阱**：尝试给REINFORCE添加回放缓冲区——这会破坏算法！
</details>

---

## 参考文献

- **Sutton & Barto**, 强化学习：导论，第13章
- **Williams (1992)**, 简单统计梯度跟随算法（REINFORCE）
- **Sutton et al. (2000)**, 带函数逼近的RL的策略梯度方法

**面试需要记忆的**：策略梯度定理，对数导数技巧推导，基线无偏性证明，优势函数，同策略vs异策略。

**代码示例**：[reinforce.py](../../../rl_examples/algorithms/reinforce.py)
