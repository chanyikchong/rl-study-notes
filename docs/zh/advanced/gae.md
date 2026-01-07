# 广义优势估计（GAE）

## 面试摘要

**GAE** 是一族优势估计器，在高偏差（TD误差）和高方差（蒙特卡洛）估计之间插值。由参数 \(\lambda \in [0,1]\) 控制。GAE公式：\(\hat{A}^{GAE}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}\)。当 \(\lambda=0\) 时，GAE等于TD误差。当 \(\lambda=1\) 时，GAE等于MC回报减去基线。用于PPO、A3C和大多数现代策略梯度方法。

**需要记忆的**：GAE公式，偏差-方差权衡，λ的效果，典型λ=0.95。

---

## 核心定义

### 优势估计问题

我们想要估计：

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

但我们不精确知道 \(Q^\pi\) 或 \(V^\pi\)。不同的估计器有不同的偏差-方差性质。

### TD误差

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- **偏差**：如果 \(V\) 错误，估计有偏
- **方差**：低（只有一个随机奖励）

### 蒙特卡洛优势

$$\hat{A}^{MC}_t = G_t - V(s_t) = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)$$

- **偏差**：零（根据 \(V^\pi\) 定义）
- **方差**：高（许多奖励的总和）

### GAE定义

$$\hat{A}^{GAE(\gamma,\lambda)}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\)。

---

## 数学与推导

### 展开GAE

$$\hat{A}^{GAE}_t = \delta_t + \gamma\lambda \delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \cdots$$

代入 \(\delta_l\)：

$$= (r_t + \gamma V(s_{t+1}) - V(s_t))$$

$$+ \gamma\lambda(r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1}))$$

$$+ (\gamma\lambda)^2(r_{t+2} + \gamma V(s_{t+3}) - V(s_{t+2}))$$

$$+ \cdots$$

### 特殊情况

**λ = 0**：

$$\hat{A}^{GAE(\gamma,0)}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

就是TD误差——低方差，可能高偏差。

**λ = 1**：

$$\hat{A}^{GAE(\gamma,1)}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}$$

这望远镜展开为：

$$= \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$

蒙特卡洛优势——零偏差，高方差。

### 偏差-方差权衡

| λ | 偏差 | 方差 | 行为 |
|---|------|----------|----------|
| 0 | 高（如果V错误）| 低 | 只有TD误差 |
| 0.5 | 中 | 中 | 平衡 |
| 0.95 | 低 | 中-高 | 接近MC但有些偏差减少 |
| 1 | 零（期望中）| 高 | 完整MC |

### 实际计算

对于有限时间范围回合（长度T）：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t-1}\delta_{T-1}$$

递归向后计算：

$$\hat{A}_{T-1} = \delta_{T-1}$$

$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$

---

## 算法概述

```
算法：计算GAE优势

输入：奖励 r[], 价值 V[], γ, λ, done[]
输出：优势 A[]

T = 回合长度
A[T] = 0  # 终止处自举

对 t = T-1 向下到 0：
    如果 done[t+1]：
        delta = r[t] - V[t]  # 终止处不自举
        A[t+1] = 0
    否则：
        delta = r[t] + γ * V[t+1] - V[t]

    A[t] = delta + γ * λ * A[t+1]

返回 A[0:T]
```

### 与PPO集成

```python
# 在PPO中，收集轨迹后：
values = critic(states)  # 所有状态的 V(s)
next_values = critic(next_states)

# 计算TD误差
deltas = rewards + gamma * next_values * (1 - dones) - values

# 计算GAE（向后传递）
advantages = np.zeros_like(rewards)
gae = 0
for t in reversed(range(len(rewards))):
    gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
    advantages[t] = gae

# 归一化优势（常见做法）
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## 常见陷阱

1. **计算方向错误**：GAE必须从回合末尾向后计算。

2. **不处理终止状态**：当done=True时，不要自举——\(\delta_t = r_t - V(s_t)\)。

3. **总是使用λ=1**：给出MC方差。λ=0.95-0.97通常更好。

4. **不归一化优势**：大优势导致大梯度。每批归一化。

5. **与TD(λ)混淆**：GAE用于优势；TD(λ)用于价值目标。相关但不同。

---

## 小例子

**短回合**：奖励 = [0, 0, 1]，价值 = [0.5, 0.6, 0.7]，γ=0.99，λ=0.95

**TD误差**：
- δ₂ = 1 + 0 - 0.7 = 0.3（步骤2后终止）
- δ₁ = 0 + 0.99×0.7 - 0.6 = 0.093
- δ₀ = 0 + 0.99×0.6 - 0.5 = 0.094

**GAE**（向后）：
- Â₂ = δ₂ = 0.3
- Â₁ = δ₁ + 0.99×0.95×Â₂ = 0.093 + 0.94×0.3 = 0.375
- Â₀ = δ₀ + 0.99×0.95×Â₁ = 0.094 + 0.94×0.375 = 0.447

**比较MC优势**（λ=1）：
- G₀ - V(s₀) = (0 + 0.99×0 + 0.99²×1) - 0.5 = 0.98 - 0.5 = 0.48

GAE给出0.447，接近但方差稍低。

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 优势估计中的偏差-方差权衡是什么？</summary>

**答案**：
- **高λ**（→1）：低偏差（接近真实优势），高方差（使用许多奖励）
- **低λ**（→0）：高偏差（依赖价值函数准确性），低方差（使用一个TD步骤）

**解释**：TD误差使用可能错误的V(s')——这是偏差。MC使用实际回报但对许多随机变量求和——这是方差。GAE在两者之间插值。

**关键洞见**：λ控制估计器的有效"视野"。

**常见陷阱**：假设λ=1总是最好因为它无偏。方差也很重要！
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> 为什么PPO中使用GAE而不是只用TD误差？</summary>

**答案**：PPO需要好的优势估计用于策略更新。纯TD误差偏差太大（特别是早期V不好时）。纯MC方差太高。GAE提供可调节的平衡。

**解释**：λ=0.95-0.97保留大部分TD的方差减少，同时获得足够低的偏差。这给出稳定的学习。

**常见陷阱**：在PPO中使用λ=0——策略梯度变得非常噪声。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 证明λ=1的GAE等于蒙特卡洛优势。</summary>

**答案**：

$$\hat{A}^{GAE(\gamma,1)}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}$$

$$= \sum_{l=0}^{\infty} \gamma^l (r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l}))$$

展开望远镜：

$$= \sum_{l=0}^{\infty} \gamma^l r_{t+l} + \sum_{l=0}^{\infty} \gamma^{l+1} V(s_{t+l+1}) - \sum_{l=0}^{\infty} \gamma^l V(s_{t+l})$$

V项望远镜：

$$= \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$

**关键洞见**：λ=1使GAE对所有TD误差求和，恢复完整回报。

**常见陷阱**：看不到望远镜展开模式。
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> 推导计算GAE的递归公式。</summary>

**答案**：

$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$

**推导**：

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

$$= \delta_t + \sum_{l=1}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

$$= \delta_t + \gamma\lambda \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+1+l}$$

$$= \delta_t + \gamma\lambda \hat{A}_{t+1}$$

**关键洞见**：这允许通过向后扫描O(T)计算。

**常见陷阱**：向前计算（错误）而不是向后（正确）。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> 如何为新问题选择λ？</summary>

**答案**：从λ=0.95开始（好的默认值）。根据以下调整：
- **噪声奖励/长回合**：较低λ（0.9-0.95）减少方差
- **准确的critic**：可以使用更高λ（0.97-0.99）
- **稀疏奖励**：需要更高λ来传播信号

**实践方法**：
1. 从λ=0.95开始
2. 如果训练不稳定，降低λ
3. 如果学习太慢，提高λ
4. 监控优势方差和策略熵

**常见陷阱**：不调λ——默认不适用于所有问题。
</details>

---

## 参考文献

- **Schulman et al. (2016)**, 使用GAE的高维连续控制
- **Sutton & Barto**, 强化学习：导论（TD(λ)章节）
- **Schulman et al. (2017)**, 近端策略优化（使用GAE）

**面试需要记忆的**：GAE公式，递归计算，λ=0是TD，λ=1是MC，典型λ=0.95，偏差-方差权衡。
