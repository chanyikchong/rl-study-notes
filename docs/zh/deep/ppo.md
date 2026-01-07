# 近端策略优化（PPO）

## 面试摘要

**PPO** 是最广泛使用的深度RL算法。它通过将策略更新限制在"信任区域"内来改进普通策略梯度——防止过大的步骤破坏训练稳定性。两种变体：**PPO-Clip**（裁剪目标）和 **PPO-Penalty**（KL惩罚）。PPO简单、稳定，在许多领域都工作良好。大多数实际应用的默认选择。

**需要记忆的**：PPO-Clip目标，为什么裁剪有帮助，比率 \(r_t(\theta)\)，典型超参数。

---

## 核心定义

### 普通策略梯度的问题

大梯度步可能：
1. 策略变化太大 → 破坏训练稳定性
2. 移动到难以恢复的坏策略
3. 使旧样本无效（数据是同策略的）

### 概率比率

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**含义**：新策略下动作 \(a_t\) 比旧策略更可能多少？

- \(r = 1\)：相同概率
- \(r > 1\)：新策略更喜欢这个动作
- \(r < 1\)：新策略更不喜欢这个动作

### PPO-Clip目标

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中 \(\epsilon \approx 0.2\) 是裁剪范围。

**直觉**：如果策略变化太大（|r-1| > ε），我们裁剪目标，所以梯度不会继续往那个方向推。

---

## 数学与推导

### 为什么裁剪有效

考虑min的情况：

**情况1：\(\hat{A}_t > 0\)**（好动作）
- 想要增加 \(\pi(a_t|s_t)\)，所以 \(r\) 增加
- 但如果 \(r > 1 + \epsilon\)，裁剪项 = \((1+\epsilon) \hat{A}\)，是常数
- 梯度变为零——停止推

**情况2：\(\hat{A}_t < 0\)**（坏动作）
- 想要减少 \(\pi(a_t|s_t)\)，所以 \(r\) 减少
- 但如果 \(r < 1 - \epsilon\)，裁剪项 = \((1-\epsilon) \hat{A}\)，常数
- 梯度变为零——停止推

裁剪在旧策略周围创建了一个"信任区域"。

### PPO完整目标

$$L(\theta) = \mathbb{E}_t\left[ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

其中：
- \(L^{VF}\)：价值函数损失（critic）
- \(S[\pi]\)：熵奖励（探索）
- \(c_1 \approx 0.5\)，\(c_2 \approx 0.01\)

### 广义优势估计（GAE）

PPO通常使用GAE进行优势估计：

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\)。

这平衡了偏差和方差（见GAE章节）。

---

## 算法概述

```
算法：PPO-Clip

超参数：ε（裁剪），γ，λ（GAE），K（epoch数），M（小批量大小）

1. 初始化 θ（策略），φ（价值函数）
2. 对 iteration = 1, 2, ...：
     # 收集轨迹
     运行策略 π_θ T 个时间步跨 N 个并行actor
     使用 V_φ 计算优势 Â_t 使用GAE
     计算回报：R_t = Â_t + V_φ(s_t)

     # 存储旧策略概率
     对所有 (s_t, a_t)：π_old(a_t|s_t) = π_θ(a_t|s_t)

     # 多轮更新
     对 k = 1 到 K：
         对每个大小为 M 的小批量：
             计算比率：r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)

             # 裁剪的代理目标
             L_clip = min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t)

             # 价值损失
             L_vf = (V_φ(s_t) - R_t)²

             # 熵奖励
             L_ent = -Σ π_θ(a|s) log π_θ(a|s)

             # 总损失
             L = -L_clip + c1·L_vf - c2·L_ent

             对 L 做梯度步
```

### 关键超参数

| 超参数 | 典型值 |
|----------------|---------------|
| 裁剪范围 ε | 0.1 - 0.3 |
| GAE λ | 0.95 |
| 折扣 γ | 0.99 |
| 学习率 | 3e-4 |
| Epoch数 K | 3 - 10 |
| 小批量大小 | 64 - 256 |
| 并行actor数 | 8 - 64 |

---

## 常见陷阱

1. **裁剪范围太大**：策略可能变化太大，导致不稳定。

2. **epoch太多**：对当前批次过拟合；策略移动太远。

3. **价值函数拟合不好**：优势变得有噪声；使用更多价值更新。

4. **忘记归一化优势**：大优势导致大梯度。每批归一化 \(\hat{A}\)。

5. **学习率太高**：PPO对学习率敏感；从低开始。

6. **不使用GAE**：原始TD误差或MC回报可以工作但GAE通常更好。

---

## 小例子

**带PPO的CartPole：**

```python
# 收集批次
for _ in range(batch_size):
    action, log_prob = sample_action(policy, state)
    next_state, reward, done = env.step(action)
    buffer.store(state, action, reward, log_prob, done)
    state = next_state

# 计算优势（GAE）
advantages = compute_gae(buffer.rewards, buffer.values, gamma, lam)
returns = advantages + buffer.values

# PPO更新（多轮）
for epoch in range(num_epochs):
    for batch in buffer.iterate_minibatches(minibatch_size):
        # 计算新的log概率
        new_log_probs = policy.log_prob(batch.states, batch.actions)

        # 比率
        ratio = exp(new_log_probs - batch.old_log_probs)

        # 裁剪目标
        surr1 = ratio * batch.advantages
        surr2 = clip(ratio, 1-eps, 1+eps) * batch.advantages
        policy_loss = -min(surr1, surr2).mean()

        # 价值损失
        value_loss = MSE(critic(batch.states), batch.returns)

        # 熵奖励
        entropy = policy.entropy(batch.states).mean()

        # 总损失
        loss = policy_loss + c1 * value_loss - c2 * entropy
        loss.backward()
        optimizer.step()
```

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 为什么PPO比普通策略梯度更稳定？</summary>

**答案**：PPO通过裁剪限制每次更新策略可以变化多少。大变化被阻止，保持更新在"信任区域"内。

**解释**：在普通PG中，大梯度步可以剧烈改变策略，使旧样本无效并可能移动到坏策略。PPO的裁剪确保小的、保守的更新。

**关键洞见**：裁剪在目标中创建一个平坦区域当策略移动太远时，停止梯度。

**常见陷阱**：认为裁剪只是为了数值稳定性。它根本上是关于限制策略变化。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> 比率 r_t(θ) 代表什么？</summary>

**答案**：新策略下动作概率与旧策略的比率。

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**解释**：
- \(r = 1\)：策略对这个动作相同
- \(r > 1\)：新策略更可能采取这个动作
- \(r < 1\)：新策略更不可能

**关键洞见**：用 \(r\) 乘以优势给出重要性加权的策略梯度。

**常见陷阱**：忘记在开始更新前存储旧策略概率。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 解释裁剪机制对正负优势如何工作。</summary>

**答案**：

**正优势**（\(\hat{A} > 0\)）：我们想增加动作概率。
- 目标是 \(\min(r \cdot \hat{A}, \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot \hat{A})\)
- 如果 \(r > 1 + \epsilon\)：裁剪项胜出，梯度 = 0
- 如果 \(r < 1 + \epsilon\)：未裁剪项胜出，梯度推 \(r\) 上升
- 效果：策略增加足够后停止推

**负优势**（\(\hat{A} < 0\)）：我们想减少动作概率。
- 如果 \(r < 1 - \epsilon\)：裁剪项胜出（更不负），梯度 = 0
- 如果 \(r > 1 - \epsilon\)：未裁剪项胜出，梯度推 \(r\) 下降
- 效果：策略减少足够后停止推

**关键方程**：\(L^{CLIP} = \min(r \hat{A}, \text{clip}(r) \hat{A})\)

**常见陷阱**：搞错min/max逻辑。画出来！
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> 为什么PPO可以重用样本进行多次梯度步？</summary>

**答案**：因为比率 \(r_t\) 修正了旧策略和新策略之间的分布不匹配。

**解释**：样本用 \(\pi_{old}\) 收集。用 \(\pi_\theta\) 使用它们需要重要性加权：

$$\mathbb{E}_{a \sim \pi_{old}}[f(a) \frac{\pi_\theta(a)}{\pi_{old}(a)}] = \mathbb{E}_{a \sim \pi_\theta}[f(a)]$$

比率提供这个修正。裁剪防止权重变得太极端。

**关键洞见**：这使PPO比普通PG更样本高效。

**常见陷阱**：重用样本太多epoch——修正变得不准确。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> PPO学习效果不好。首先调什么超参数？</summary>

**答案**：调参优先级：
1. **学习率**：尝试1e-4到3e-4；太高导致不稳定
2. **裁剪范围ε**：尝试0.1、0.2、0.3
3. **epoch数K**：如果过拟合（策略变化太多）减少epoch
4. **GAE λ**：较低λ（0.9）减少方差，较高（0.99）减少偏差
5. **批量大小**：更大批次给更稳定梯度
6. **熵系数**：如果策略崩溃则增加

**解释**：PPO对学习率和裁剪范围敏感。从论文默认值开始，然后调整。

**常见陷阱**：同时调太多东西。一次改一个超参数。
</details>

---

## 参考文献

- **Schulman et al. (2017)**, 近端策略优化算法
- **Schulman et al. (2016)**, 使用GAE的高维连续控制
- **Engstrom et al. (2020)**, 深度RL中实现很重要

**面试需要记忆的**：PPO-Clip目标，比率定义，正负优势的裁剪逻辑，典型超参数，为什么它稳定。

**代码示例**：[ppo.py](../../../rl_examples/algorithms/ppo.py)
