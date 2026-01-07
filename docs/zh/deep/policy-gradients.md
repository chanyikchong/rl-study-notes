# 策略梯度

## 面试摘要

**策略梯度方法** 通过对期望回报的梯度上升直接优化策略 \(\pi_\theta(a|s)\)。关键结果是**策略梯度定理**：\(\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]\)。**REINFORCE** 是最简单的算法——用采样回报作为Q估计。高方差是主要挑战；基线可以减少它。

**需要记忆的**：策略梯度定理，REINFORCE更新，对数导数技巧，为什么基线减少方差而不增加偏差。

---

## 核心定义

### 策略参数化

随机策略：\(\pi_\theta(a|s) = P(A=a|S=s; \theta)\)

常见形式：
- **离散**：动作logits上的softmax
- **连续**：高斯 \(\mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)\)

### 目标

最大化期望回报：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

### 策略梯度定理

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a) \right]$$

**含义**：目标的梯度等于期望的"得分"乘以动作价值。

---

## 数学与推导

### 推导策略梯度

从以下开始：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_\tau P(\tau|\theta) R(\tau)$$

取梯度：

$$\nabla J = \sum_\tau \nabla P(\tau|\theta) R(\tau)$$

使用对数导数技巧：\(\nabla P = P \nabla \log P\)：

$$= \sum_\tau P(\tau|\theta) \nabla \log P(\tau|\theta) R(\tau)$$

$$= \mathbb{E}_\tau[\nabla \log P(\tau|\theta) \cdot R(\tau)]$$

展开 \(\log P(\tau|\theta)\)：

$$\log P(\tau|\theta) = \log p(s_0) + \sum_{t=0}^{T} \left[ \log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t) \right]$$

只有 \(\pi_\theta\) 依赖 \(\theta\)：

$$\nabla \log P(\tau|\theta) = \sum_{t=0}^{T} \nabla \log \pi_\theta(a_t|s_t)$$

**最终形式**：

$$\nabla J = \mathbb{E}_\tau\left[ \sum_{t=0}^{T} \nabla \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

其中 \(G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k\) 是从t开始的回报（不是总回报——因果性！）。

### 基线减法

$$\nabla J = \mathbb{E}\left[ \nabla \log \pi_\theta(a|s) \cdot (Q^{\pi}(s,a) - b(s)) \right]$$

任何不依赖 \(a\) 的基线 \(b(s)\) 都可以减去。

**为什么无偏？**

$$\mathbb{E}_a[\nabla \log \pi(a|s) \cdot b(s)] = b(s) \cdot \nabla \sum_a \pi(a|s) = b(s) \cdot \nabla 1 = 0$$

**常见基线**：\(b(s) = V^\pi(s)\)，给出优势：\(A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)\)。

---

## 算法概述

### REINFORCE

```
算法：REINFORCE（蒙特卡洛策略梯度）

输入：学习率 α
输出：策略参数 θ

1. 任意初始化 θ
2. 对每个回合：
     生成轨迹 τ = (s_0, a_0, r_1, s_1, ..., s_T) 使用 π_θ
     对 t = 0 到 T-1：
         G_t = Σ_{k=t}^{T-1} γ^{k-t} r_{k+1}  # 从 t 开始的回报
         θ ← θ + α · γ^t · G_t · ∇_θ log π_θ(a_t|s_t)
3. 返回 θ
```

### 带基线的REINFORCE

```
算法：带基线的REINFORCE

1. 初始化 θ（策略），w（价值函数）
2. 对每个回合：
     使用 π_θ 生成轨迹
     对 t = 0 到 T-1：
         G_t = Σ_{k=t}^{T-1} γ^{k-t} r_{k+1}
         δ = G_t - V(s_t; w)           # 优势估计
         w ← w + α_w · δ · ∇_w V(s_t; w)  # 价值更新
         θ ← θ + α_θ · γ^t · δ · ∇_θ log π_θ(a_t|s_t)
3. 返回 θ
```

---

## 常见陷阱

1. **高方差**：REINFORCE因为使用完整回报有高方差。使用基线！

2. **样本效率低**：同策略——每个样本只用一次。比异策略效率低得多。

3. **奖励缩放**：大奖励 → 大梯度 → 不稳定。归一化回报。

4. **熵崩溃**：策略太快变得太确定性。添加熵奖励。

5. **学习率敏感性**：策略梯度对α非常敏感。从小开始（1e-4）。

6. **忘记使用从t开始的回报**：使用总回合回报而不是 \(G_t\) 增加方差。

---

## 小例子

**带REINFORCE的CartPole：**

```python
# 策略：神经网络 → softmax → 动作概率
policy = MLP(4, [32], 2)

for episode in range(1000):
    states, actions, rewards = [], [], []

    # 收集回合
    state = env.reset()
    while not done:
        probs = softmax(policy(state))
        action = sample(probs)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # 计算回报
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    # 策略更新
    for s, a, G in zip(states, actions, returns):
        log_prob = log(softmax(policy(s))[a])
        loss = -log_prob * G  # 负号用于梯度上升
        loss.backward()
        optimizer.step()
```

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 为什么直接优化策略而不是学习Q然后贪婪行动？</summary>

**答案**：策略梯度有优势：
1. **连续动作**：可以自然输出连续动作分布
2. **随机策略**：可以表示混合策略
3. **直接优化**：优化我们关心的（期望回报）
4. **更好收敛**：避免Q学习中max算子的问题

**解释**：基于价值的方法需要对动作取max，这对连续/大动作空间有问题。策略梯度绕过这个。

**常见陷阱**：假设策略梯度总是更好。基于价值的方法对离散动作样本效率更高。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> 为什么基线不给梯度估计增加偏差？</summary>

**答案**：因为 \(\mathbb{E}_a[\nabla \log \pi(a|s) \cdot b(s)] = 0\) 对任何与 \(a\) 无关的 \(b(s)\)。

**解释**：

$$\mathbb{E}_a[\nabla \log \pi(a|s) \cdot b(s)] = b(s) \sum_a \nabla \pi(a|s) = b(s) \nabla \sum_a \pi(a|s) = b(s) \nabla 1 = 0$$

基线因子提出并乘以零。

**关键洞见**：基线移动回报但不改变期望梯度方向。

**常见陷阱**：使用依赖 \(a\) 的基线，这确实增加偏差。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 从头推导策略梯度定理。</summary>

**答案**：见上面数学与推导部分。关键步骤：
1. \(J(\theta) = \mathbb{E}_\tau[R(\tau)]\)
2. 对数导数技巧：\(\nabla P = P \nabla \log P\)
3. 展开 \(\log P(\tau|\theta)\)，梯度只通过 \(\pi_\theta\)
4. 使用因果性：时刻 \(t\) 的动作只影响 \(t\) 之后的奖励

**关键方程**：

$$\nabla J = \mathbb{E}\left[\sum_t \nabla \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

**常见陷阱**：使用总回报 \(R(\tau)\) 而不是从t开始的回报 \(G_t\) 增加方差。
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> 什么是优势函数，为什么它优于Q？</summary>

**答案**：\(A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)\)

**为什么优于Q**：
- 以零为中心（坏动作负，好动作正）
- 比原始Q值方差低
- 将梯度聚焦于相对动作质量

**解释**：直接使用Q，即使是常数高奖励也给所有动作正梯度。优势只强化比平均好的动作。

**常见陷阱**：不使用基线。原始回报有高方差。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> 你的策略梯度训练非常不稳定。尝试什么？</summary>

**答案**：稳定化技术：
1. **降低学习率**：从1e-4或更低开始
2. **添加基线**：减去状态价值以减少方差
3. **归一化回报**：批内 \((G - \mu) / \sigma\)
4. **熵正则化**：添加 \(\beta H(\pi)\) 到目标以防止过早收敛
5. **梯度裁剪**：裁剪梯度范数
6. **更大批量**：对更多轨迹取平均

**解释**：策略梯度本质上是高方差的。所有这些技术都减少有效方差。

**常见陷阱**：每次更新样本不够。批量大小比监督学习更重要。
</details>

---

## 参考文献

- **Sutton & Barto**, 强化学习：导论，第13章
- **Williams (1992)**, 简单统计梯度跟随算法（REINFORCE）
- **Sutton et al. (2000)**, 带函数逼近的RL的策略梯度方法

**面试需要记忆的**：策略梯度定理，对数导数技巧，REINFORCE更新，基线无偏性证明，优势函数。

**代码示例**：[reinforce.py](../../../rl_examples/algorithms/reinforce.py)
