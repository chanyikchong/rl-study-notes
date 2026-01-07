# Actor-Critic方法

## 面试摘要

**Actor-Critic** 方法结合策略梯度（actor）和价值函数学习（critic）。**Actor** 使用策略梯度更新策略，而 **critic** 估计价值函数以减少方差。这是A2C、A3C和PPO的基础。关键洞见：使用TD误差作为优势的低方差估计，允许每步更新而不是等待回合结束。

**需要记忆的**：Actor-Critic架构，TD误差作为优势，A2C更新规则，A3C的异步训练。

---

## 核心定义

### Actor-Critic架构

**Actor**：策略 \(\pi_\theta(a|s)\) — 决定采取什么动作
**Critic**：价值函数 \(V_\phi(s)\) 或 \(Q_\phi(s,a)\) — 评估状态/动作有多好

### 为什么结合？

| 纯策略梯度 | 纯基于价值 |
|---------------------|------------------|
| 高方差 | 更低方差 |
| 可以学习随机策略 | 确定性策略 |
| 只能同策略 | 可以异策略 |
| 适用于连续动作 | 需要max（离散）|

Actor-Critic获得两者的好处：从critic获得更低方差，从actor获得随机策略。

### TD误差作为优势估计

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

这是优势 \(A^\pi(s_t, a_t)\) 在轨迹上平均时的无偏估计。

---

## 数学与推导

### Actor更新（带Critic的策略梯度）

$$\nabla_\theta J \approx \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s, a) \right]$$

其中 \(\hat{A}\) 是优势估计：
- **蒙特卡洛**：\(\hat{A} = G_t - V_\phi(s_t)\)
- **TD(0)**：\(\hat{A} = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\)
- **GAE**：加权组合（见GAE章节）

### Critic更新（TD学习）

$$\phi \leftarrow \phi + \alpha_c \cdot \delta_t \cdot \nabla_\phi V_\phi(s_t)$$

或用MSE损失：

$$L(\phi) = \mathbb{E}\left[ (V_\phi(s) - V^{target})^2 \right]$$

其中 \(V^{target} = r + \gamma V_\phi(s')\) 或 \(G_t\)（蒙特卡洛）。

### 为什么TD误差估计优势

$$\mathbb{E}[\delta_t | s_t, a_t] = \mathbb{E}[r_t + \gamma V(s_{t+1}) | s_t, a_t] - V(s_t)$$

$$= Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

所以TD误差是优势的无偏（但有噪声）估计！

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
3. 每个工作者：
     θ, φ 的本地副本
     与自己的环境交互
     收集 T 步经验
     本地计算梯度
     应用梯度到全局 θ, φ（异步）
     同步本地 ← 全局
```

**关键好处**：并行工作者提供多样经验，自然探索。

---

## 常见陷阱

1. **Critic学习太慢**：如果critic差，actor得到噪声梯度。Critic通常需要更高学习率。

2. **共享网络问题**：actor/critic共享层可能导致干扰。使用独立网络或小心的架构。

3. **熵崩溃**：策略太快变得确定性。添加熵奖励：\(L = -\log \pi(a|s) \cdot A - \beta H(\pi)\)。

4. **Critic目标问题**：critic目标应该包含新的还是旧的价值估计？保持一致。

5. **不处理终止状态**：\(V(s_{terminal}) = 0\)，不要从它自举。

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

**解释**：REINFORCE使用完整回合回报 \(G_t\)，它有高方差（许多随机奖励的总和）。Actor-critic使用TD误差 \(\delta_t = r + \gamma V(s') - V(s)\)，只涉及一个奖励加上学习的估计。

**权衡**：TD误差引入偏差（如果V错误），但方差减少通常获胜。

**常见陷阱**：认为actor-critic总是更好。对于有稀疏奖励的短回合，REINFORCE可能有竞争力。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> actor-critic中熵正则化的作用是什么？</summary>

**答案**：防止过早收敛到确定性策略，鼓励探索。

**解释**：没有熵奖励，策略梯度推向确定性策略（一个动作获得所有概率）。添加 \(\beta H(\pi)\) 到目标奖励不确定性。

**实现**：\(\nabla_\theta [\beta H(\pi)] = -\beta \sum_a \pi(a|s) \log \pi(a|s)\)

**常见陷阱**：\(\beta\) 设置太高使策略太随机。太低允许熵崩溃。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 证明 E[δ_t | s_t, a_t] = A^π(s_t, a_t)。</summary>

**答案**：

$$\mathbb{E}[\delta_t | s_t, a_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) | s_t, a_t] - V^\pi(s_t)$$

$$= R(s_t, a_t) + \gamma \sum_{s'} P(s'|s_t, a_t) V^\pi(s') - V^\pi(s_t)$$

$$= Q^\pi(s_t, a_t) - V^\pi(s_t)$$

$$= A^\pi(s_t, a_t)$$

**关键洞见**：TD误差是优势的无偏估计器！

**常见陷阱**：这要求 \(V^\pi\) 是正确的。用学习的 \(V_\phi\)，我们得到有偏估计。
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> A3C如何在没有显式探索奖励的情况下实现探索？</summary>

**答案**：不同随机种子的并行工作者自然探索状态空间的不同部分。

**解释**：每个工作者独立运行，做出不同的随机动作选择。全局网络聚合这些多样经验。这通过多样性提供隐式探索。

**额外机制**：A3C中通常仍然使用熵奖励。

**常见陷阱**：认为并行只是为了速度。它真正改善了探索和稳定性。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> 你的actor-critic没有学习。Actor损失在振荡。调试什么？</summary>

**答案**：调试步骤：
1. **先检查critic**：如果critic差，actor无法学习。训练critic更长，使用更高的α_φ。
2. **降低α_θ**：Actor学习率太高导致振荡。
3. **分离TD误差**：不要反向传播actor梯度通过critic。
4. **添加熵奖励**：可能卡在局部最优。
5. **检查优势符号**：好动作正，坏动作负。
6. **使用批处理**：单步更新有噪声；尝试n步或批量更新。

**解释**：大多数actor-critic失败是因为critic质量。坏critic给出噪声梯度。

**常见陷阱**：当问题是critic时责怪actor。
</details>

---

## 参考文献

- **Sutton & Barto**, 强化学习：导论，第13.5章
- **Mnih et al. (2016)**, 深度RL的异步方法（A3C）
- **Konda & Tsitsiklis (2000)**, Actor-Critic算法

**面试需要记忆的**：Actor-Critic架构，TD误差作为优势，A2C vs A3C，熵正则化目的。

**代码示例**：[actor_critic.py](../../../rl_examples/algorithms/actor_critic.py)
