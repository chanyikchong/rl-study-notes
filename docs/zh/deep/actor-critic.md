# Actor-Critic方法

## 面试摘要

**Actor-Critic** 方法结合策略梯度（actor）和价值函数学习（critic）。**Actor** 使用策略梯度更新策略，而 **critic** 估计价值函数以减少方差。这是A2C、A3C和PPO的基础。关键洞见：使用TD误差作为优势的低方差估计，允许每步更新而不是等待回合结束。

**需要记忆的**：Actor-Critic架构，TD误差作为优势，A2C更新规则，A3C的异步训练。

---

## 设计动机：为什么Actor-Critic存在

### REINFORCE的问题

REINFORCE使用完整蒙特卡洛回报：

$$\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是许多随机奖励的总和。

**问题：高方差**
- $G_t$ 是从 $t$ 到回合结束的所有奖励的总和
- 每个奖励都有噪声 → 总和方差很大
- 需要许多样本才能得到可靠的梯度估计

**问题：必须等到回合结束**
- 需要完整轨迹来计算 $G_t$
- 对长回合效率低下
- 不能做在线学习

### Actor-Critic的思想

> **关键洞见**：不用等待观察所有未来奖励，用学习的估计 $V(s)$ 来预测它们！

```
REINFORCE：使用 G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...  （实际奖励）
Actor-Critic：使用 r_t + γV(s_{t+1})                    （一个奖励 + 估计）
```

**权衡：**

| 方面 | REINFORCE | Actor-Critic |
|------|-----------|--------------|
| 方差 | 高（许多随机项的总和） | 低（一个奖励 + 估计） |
| 偏差 | 无偏 | 有偏（如果V错误） |
| 更新 | 每回合 | 每步 |
| 样本效率 | 更低 | 更高 |

方差减少的收益通常大于偏差的代价！

### 为什么称为"Actor-Critic"？

- **Actor**：策略 $\pi_\theta(a|s)$ — 决定采取什么动作（"演员"）
- **Critic**：价值函数 $V_\phi(s)$ — 评估状态有多好（"评论家"）

演员表演，评论家评分。评论家的反馈帮助演员改进！

---

## 核心定义

### Actor-Critic架构

```
        状态 s
           │
    ┌──────┴──────┐
    ▼             ▼
┌───────┐   ┌───────┐
│ Actor │   │Critic │
│  π_θ  │   │  V_φ  │
└───┬───┘   └───┬───┘
    │           │
    ▼           ▼
  动作 a     价值 V(s)
```

**两个独立的学习者：**
1. Actor学习策略（用策略梯度）
2. Critic学习价值函数（用TD学习）

### TD误差作为优势估计

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

**关键结果**：

$$\mathbb{E}[\delta_t | s_t, a_t] = A^\pi(s_t, a_t)$$

TD误差是优势的**无偏估计**！（见下面的证明）

### 为什么结合？

| 纯策略梯度 | 纯基于价值 | Actor-Critic |
|------------|------------|--------------|
| 高方差 | 更低方差 | 低方差（从critic） |
| 可以随机策略 | 确定性策略 | 可以随机策略 |
| 只能同策略 | 可以异策略 | 灵活 |
| 适用连续动作 | 需要max（离散）| 适用连续动作 |

Actor-Critic获得两方面的好处！

---

## 数学与推导

### Actor更新（带Critic的策略梯度）

$$\nabla_\theta J \approx \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s, a) \right]$$

其中 $\hat{A}$ 是优势估计：
- **蒙特卡洛**：$\hat{A} = G_t - V_\phi(s_t)$（低偏差，高方差）
- **TD(0)**：$\hat{A} = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$（高偏差，低方差）
- **GAE**：加权组合（见GAE章节）

### Critic更新（TD学习）

使用半梯度TD：

$$\phi \leftarrow \phi + \alpha_c \cdot \delta_t \cdot \nabla_\phi V_\phi(s_t)$$

或用MSE损失：

$$L(\phi) = \mathbb{E}\left[ (V_\phi(s) - V^{target})^2 \right]$$

其中 $V^{target} = r + \gamma V_\phi(s')$（TD）或 $G_t$（蒙特卡洛）。

### 证明TD误差估计优势（重要！）

这在面试中经常被问到。

$$\mathbb{E}[\delta_t | s_t, a_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t) | s_t, a_t]$$

$$= \mathbb{E}[r_t | s_t, a_t] + \gamma \mathbb{E}[V^\pi(s_{t+1}) | s_t, a_t] - V^\pi(s_t)$$

$$= R(s_t, a_t) + \gamma \sum_{s'} P(s'|s_t, a_t) V^\pi(s') - V^\pi(s_t)$$

根据Q函数的定义：

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')$$

所以：

$$\mathbb{E}[\delta_t | s_t, a_t] = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

**关键洞见**：TD误差是优势的无偏估计器！

**注意**：这要求 $V^\pi$ 是正确的。用学习的 $V_\phi$，我们得到有偏估计。

---

## 算法概述

### 单步Actor-Critic（A2C风格）

```
算法：优势Actor-Critic（A2C）

参数：θ（actor），φ（critic）
超参数：α_θ, α_φ, γ, β（熵系数）

1. 初始化 θ, φ
2. 对每一步：
     观察状态 s
     采样动作 a ~ π_θ(·|s)
     执行 a，观察 r, s', done

     # 计算TD误差（优势估计）
     如果 done：
         δ = r - V_φ(s)
     否则：
         δ = r + γ V_φ(s') - V_φ(s)

     # Critic更新
     φ ← φ + α_φ · δ · ∇_φ V_φ(s)

     # Actor更新
     θ ← θ + α_θ · δ · ∇_θ log π_θ(a|s)

     # 可选：探索的熵奖励
     θ ← θ + β · ∇_θ H(π_θ(·|s))

     s ← s'
```

### A3C（异步优势Actor-Critic）

```
算法：A3C（高层）

1. 全局参数：θ, φ（共享）
2. 启动 N 个并行工作者

3. 每个工作者循环：
     # 同步本地参数
     θ_local ← θ_global
     φ_local ← φ_global

     # 收集 n 步经验
     对 t = 1 到 n：
         a_t ~ π(·|s_t; θ_local)
         执行 a_t，观察 r_t, s_{t+1}

     # 计算 n 步回报
     如果 terminal：R = 0
     否则：R = V(s_n; φ_local)

     对 t = n-1 到 0：
         R = r_t + γ R
         计算优势：A_t = R - V(s_t)

     # 计算梯度
     dθ = Σ ∇_θ log π(a_t|s_t) · A_t
     dφ = Σ ∇_φ (R_t - V(s_t))²

     # 异步更新全局参数
     θ_global ← θ_global + α_θ · dθ
     φ_global ← φ_global + α_φ · dφ
```

### A2C vs A3C

| 方面 | A2C（同步） | A3C（异步） |
|------|-------------|-------------|
| 更新 | 等所有工作者 | 工作者独立更新 |
| 批处理 | 是 | 否 |
| GPU效率 | 更高 | 更低 |
| 实现 | 更简单 | 更复杂 |
| 常用 | 是（更稳定） | 较少（不稳定） |

现代实践中，A2C通常优于A3C因为更好的批处理和GPU利用。

---

## 常见陷阱

1. **Critic学习太慢**：如果critic差，actor得到噪声梯度。Critic通常需要更高学习率或更多更新。

2. **共享网络问题**：actor/critic共享层可能导致干扰。使用独立网络或小心的架构。

3. **熵崩溃**：策略太快变得确定性。添加熵奖励：$L = -\log \pi(a|s) \cdot A - \beta H(\pi)$。

4. **Critic目标问题**：critic目标应该包含新的还是旧的价值估计？保持一致。

5. **不处理终止状态**：$V(s_{terminal}) = 0$，不要从它自举。

6. **TD误差梯度泄漏**：actor更新时不要让梯度流过critic。使用 `.detach()`。

---

## 小例子

**CartPole Actor-Critic：**

```python
# 网络
actor = MLP(4, [32], 2)   # 策略logits
critic = MLP(4, [32], 1)  # 价值

for episode in range(1000):
    state = env.reset()
    while not done:
        # 前向传播
        logits = actor(state)
        probs = softmax(logits)
        value = critic(state)

        # 采样动作
        action = sample(probs)
        next_state, reward, done = env.step(action)

        # TD误差
        next_value = 0 if done else critic(next_state)
        td_error = reward + gamma * next_value - value

        # Critic更新
        critic_loss = td_error ** 2
        critic_loss.backward()

        # Actor更新
        log_prob = log_softmax(logits)[action]
        actor_loss = -log_prob * td_error.detach()  # 停止通过td_error的梯度
        actor_loss.backward()

        optimizer.step()
        state = next_state
```

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 为什么使用actor-critic而不是纯REINFORCE？</summary>

**答案**：从critic获得更低方差。

**解释**：REINFORCE使用完整回合回报 $G_t$，它有高方差（许多随机奖励的总和）。Actor-critic使用TD误差 $\delta_t = r + \gamma V(s') - V(s)$，只涉及一个奖励加上学习的估计。

**权衡**：TD误差引入偏差（如果V错误），但方差减少通常获胜。

**常见陷阱**：认为actor-critic总是更好。对于有稀疏奖励的短回合，REINFORCE可能有竞争力。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> actor-critic中熵正则化的作用是什么？</summary>

**答案**：防止过早收敛到确定性策略，鼓励探索。

**解释**：没有熵奖励，策略梯度推向确定性策略（一个动作获得所有概率）。添加 $\beta H(\pi)$ 到目标奖励不确定性。

**熵公式**：$H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$

**实现**：目标变成 $L = L_{policy} + \beta H(\pi)$

**常见陷阱**：$\beta$ 设置太高使策略太随机。太低允许熵崩溃。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 证明 E[δ_t | s_t, a_t] = A^π(s_t, a_t)。</summary>

**答案**：

$$\mathbb{E}[\delta_t | s_t, a_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) | s_t, a_t] - V^\pi(s_t)$$

$$= R(s_t, a_t) + \gamma \sum_{s'} P(s'|s_t, a_t) V^\pi(s') - V^\pi(s_t)$$

$$= Q^\pi(s_t, a_t) - V^\pi(s_t)$$

$$= A^\pi(s_t, a_t)$$

**关键洞见**：TD误差是优势的无偏估计器！

**常见陷阱**：这要求 $V^\pi$ 是正确的。用学习的 $V_\phi$，我们得到有偏估计。
</details>

<details markdown="1">
<summary><strong>问题4（概念）：</strong> A3C如何在没有显式探索奖励的情况下实现探索？</summary>

**答案**：不同随机种子的并行工作者自然探索状态空间的不同部分。

**解释**：每个工作者独立运行，做出不同的随机动作选择。全局网络聚合这些多样经验。这通过多样性提供隐式探索。

**额外机制**：A3C中通常仍然使用熵奖励。

**常见陷阱**：认为并行只是为了速度。它真正改善了探索和稳定性。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> 你的actor-critic没有学习。Actor损失在振荡。调试什么？</summary>

**答案**：调试步骤：
1. **先检查critic**：如果critic差，actor无法学习。训练critic更长，使用更高的 $\alpha_\phi$。
2. **降低 $\alpha_\theta$**：Actor学习率太高导致振荡。
3. **分离TD误差**：不要反向传播actor梯度通过critic。使用 `.detach()`。
4. **添加熵奖励**：可能卡在局部最优。
5. **检查优势符号**：好动作正，坏动作负。
6. **使用批处理**：单步更新有噪声；尝试n步或批量更新。

**解释**：大多数actor-critic失败是因为critic质量。坏critic给出噪声梯度。

**常见陷阱**：当问题是critic时责怪actor。
</details>

<details markdown="1">
<summary><strong>问题6（概念）：</strong> 为什么A2C现在比A3C更常用？</summary>

**答案**：A2C（同步）比A3C（异步）有几个优势：

1. **更好的GPU利用**：同步批处理允许高效的GPU计算
2. **更稳定的训练**：没有来自陈旧梯度的不一致
3. **更简单的实现**：没有复杂的异步同步
4. **经验上相似的性能**：在大多数任务上表现相当

**解释**：A3C设计于GPU不太普遍时。现代硬件使同步批处理更高效。

**常见陷阱**：认为"异步=更快"。批处理的效率增益通常胜过并行度。
</details>

---

## 参考文献

- **Sutton & Barto**, 强化学习：导论，第13.5章
- **Mnih et al. (2016)**, 深度RL的异步方法（A3C）
- **Konda & Tsitsiklis (2000)**, Actor-Critic算法

**面试需要记忆的**：Actor-Critic架构，TD误差作为优势的证明，A2C vs A3C，熵正则化目的。

**代码示例**：[actor_critic.py](../../../rl_examples/algorithms/actor_critic.py)
