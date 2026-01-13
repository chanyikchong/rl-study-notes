# 大语言模型训练的强化学习方法

## 面试摘要

LLM训练的RL方法使语言模型与人类偏好对齐。**RLHF**使用奖励模型+PPO优化响应。**TRPO**通过约束优化提供理论保证但复杂。**PPO**用截断简化TRPO。**DPO**通过直接优化偏好对消除奖励模型。**GRPO**使用基于组的优势估计，无需评论家网络。了解权衡：PPO灵活但需要奖励模型；DPO更简单但不够灵活；GRPO平衡两者。

---

## 为什么用RL训练LLM？

### 对齐问题

预训练的LLM学习预测下一个token，但这不能保证：
- 有帮助、无害、诚实的响应
- 准确遵循指令
- 避免有害内容

**监督微调（SFT）**本身是有限的，因为：
1. 收集专家演示成本高
2. 模型可能记忆而不理解意图
3. 难以为所有情况指定"好"的响应

### RLHF流程

基于人类反馈的强化学习（RLHF）解决这个问题：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RLHF 流程                                         │
└─────────────────────────────────────────────────────────────────────────────┘

步骤1: 监督微调 (SFT)
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ 基础LLM      │ →  │ 在演示数据上    │ →  │ SFT模型      │
│ (预训练)     │    │ 微调            │    │ π_SFT        │
└──────────────┘    └─────────────────┘    └──────────────┘

步骤2: 奖励模型训练
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ 收集偏好     │ →  │ 训练预测       │ →  │ 奖励模型     │
│ (y_w > y_l)  │    │ 人类偏好        │    │ r_φ(x, y)    │
└──────────────┘    └─────────────────┘    └──────────────┘

步骤3: RL优化
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ SFT模型      │ →  │ 用PPO/TRPO等   │ →  │ 对齐后的LLM  │
│ π_SFT        │    │ 优化            │    │ π_θ          │
└──────────────┘    └─────────────────┘    └──────────────┘
```

### 核心目标

所有方法都旨在最大化期望奖励，同时保持与参考策略接近：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{KL}\left(\pi_\theta \| \pi_{ref}\right)$$

其中：
- $\pi_\theta$: 正在优化的策略（LLM）
- $\pi_{ref}$: 参考策略（通常是SFT模型）
- $r_\phi(x, y)$: 奖励模型对提示$x$的响应$y$的评分
- $\beta$: KL惩罚系数（防止奖励黑客）

**为什么要KL惩罚？** 没有它，模型可以利用奖励模型的弱点，产生高奖励但无意义的输出（奖励黑客）。

---

## TRPO: 信任区域策略优化

### 基础

TRPO（Schulman等，2015）提供理论基础。它问：*如何改进策略同时保证不会变差？*

### 动机：策略改进问题

在标准策略梯度中，大更新可能灾难性地降低性能：

$$\theta_{new} = \theta_{old} + \alpha \nabla_\theta J(\theta)$$

问题：如何选择$\alpha$？太大→策略崩溃。太小→学习缓慢。

### 代理目标

TRPO优化一个**代理目标**，它是真实改进的下界：

$$L^{CPI}(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t \right] = \mathbb{E}_t \left[ r_t(\theta) A_t \right]$$

其中$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$是概率比。

### 信任区域约束

TRPO约束KL散度以确保单调改进：

$$\max_\theta \quad L^{CPI}(\theta)$$
$$\text{s.t.} \quad \mathbb{E}_t \left[ D_{KL}\left(\pi_{\theta_{old}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t)\right) \right] \leq \delta$$

**关键洞察**：在这个"信任区域"（$D_{KL} \leq \delta$）内，代理目标准确预测真实目标。

### 数学推导

**步骤1：性能差异引理**

两个策略之间的差异可以表示为：

$$J(\pi_{new}) - J(\pi_{old}) = \mathbb{E}_{s \sim d^{\pi_{new}}, a \sim \pi_{new}} \left[ A^{\pi_{old}}(s, a) \right]$$

其中$d^{\pi_{new}}$是新策略下的状态分布。

**步骤2：问题**

我们无法从$d^{\pi_{new}}$采样，因为我们还没有新策略！

**步骤3：近似**

TRPO使用旧状态分布加重要性采样：

$$L^{CPI}(\theta) = \mathbb{E}_{s \sim d^{\pi_{old}}, a \sim \pi_{old}} \left[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A^{\pi_{old}}(s, a) \right]$$

**步骤4：误差界限**

当$D_{KL}$较小时，使用错误状态分布引入的误差是有界的：

$$|J(\pi_{new}) - J(\pi_{old}) - L^{CPI}(\theta)| \leq C \cdot \sqrt{\mathbb{E}_s[D_{KL}(\pi_{old} \| \pi_{new})]}$$

### TRPO算法

```
TRPO算法:
─────────────────────────────────────────────────────────
输入: 初始策略π_θ, 约束δ

for iteration = 1, 2, ... do:
    1. 使用当前策略π_θ收集轨迹

    2. 计算所有时间步的优势A_t

    3. 计算策略梯度:
       g = ∇_θ L^CPI(θ)|_{θ=θ_old}

    4. 计算Fisher信息矩阵:
       F = ∇²_θ D_KL(π_θ_old || π_θ)|_{θ=θ_old}

    5. 计算自然梯度方向:
       d = F^{-1} g

    6. 通过线搜索计算步长:
       找到最大的β使得:
       - D_KL(π_θ_old || π_{θ_old + βd}) ≤ δ
       - L^CPI(θ_old + βd) > L^CPI(θ_old)

    7. 更新: θ ← θ_old + βd
─────────────────────────────────────────────────────────
```

### 求解约束优化

使用拉格朗日方法：

$$\mathcal{L}(\theta, \lambda) = L^{CPI}(\theta) - \lambda \left( D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) - \delta \right)$$

最优步使用**自然梯度**：

$$\theta_{new} = \theta_{old} + \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g$$

其中$F$是Fisher信息矩阵（近似KL的Hessian）。

### 复杂性问题

TRPO需要：
1. 计算Fisher矩阵$F$（对神经网络来说很昂贵）
2. 计算$F^{-1}g$（使用共轭梯度，但仍然昂贵）
3. 线搜索以满足约束（多次前向传播）

**这激发了PPO** — 我们能在没有复杂性的情况下获得类似保证吗？

---

## PPO: 近端策略优化

### 从约束到惩罚

PPO（Schulman等，2017）通过用截断替换硬约束来简化TRPO。

### 截断代理目标

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- $\epsilon$是截断参数（通常0.1-0.2）

### 截断如何工作

**情况1：$A_t > 0$（好动作）**

我们想增加$\pi_\theta(a|s)$，所以$r_t$增加。

$$L^{CLIP} = \min(r_t A_t, (1+\epsilon) A_t)$$

当$r_t > 1 + \epsilon$时，目标被截断到$(1+\epsilon)A_t$。没有进一步增加概率的激励。

**情况2：$A_t < 0$（坏动作）**

我们想减少$\pi_\theta(a|s)$，所以$r_t$减少。

$$L^{CLIP} = \min(r_t A_t, (1-\epsilon) A_t) = \max(r_t |A_t|, (1-\epsilon) |A_t|) \cdot (-1)$$

当$r_t < 1 - \epsilon$时，目标被截断。没有进一步减少概率的激励。

### 可视化理解

```
                    A > 0时的L^CLIP
           │
    (1+ε)A ┼─────────────────────────────
           │                    ╱
           │                  ╱
         A ┼────────────────╱
           │              ╱
           │            ╱
           │          ╱
           │        ╱
         0 ┼──────╱─────────────────────── r_t
           │    ╱
           │  ╱
           │╱
           0    1-ε    1    1+ε    2

           截断: 超过1+ε后无梯度

                    A < 0时的L^CLIP
           │
         0 ┼──────────────────────────────
           │╲
           │  ╲
           │    ╲
           │      ╲
    (1-ε)A ┼────────╲─────────────────────
           │          ╲
           │            ╲
         A ┼──────────────╲───────────────
           │                ╲
           0    1-ε    1    1+ε    2    r_t

           截断: 低于1-ε后无梯度
```

### LLM的PPO（RLHF）

在LLM设置中，PPO优化：

$$\mathcal{L}_{PPO-LLM} = \mathbb{E}_{x, y} \left[ \min\left( r(\theta) A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

其中：
- $x$是提示
- $y$是生成的响应
- $A = r_\phi(x, y) - b(x)$是优势（奖励减去基线）
- KL项防止偏离参考策略

### PPO-RLHF算法

```
RLHF的PPO:
─────────────────────────────────────────────────────────
输入: SFT模型π_ref, 奖励模型r_φ, 截断ε, KL系数β

初始化: π_θ ← π_ref

for iteration = 1, 2, ... do:
    1. 采样提示 x ~ D

    2. 生成响应: y ~ π_θ(·|x)

    3. 计算奖励: R = r_φ(x, y)

    4. 计算优势: A = R - baseline
       (baseline可以是运行均值或价值函数)

    5. 对每个minibatch epoch:

       计算比率: r(θ) = π_θ(y|x) / π_θ_old(y|x)

       计算截断目标:
       L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)

       计算KL惩罚:
       L^KL = β · D_KL(π_θ || π_ref)

       更新: θ ← θ + α∇_θ(L^CLIP - L^KL)
─────────────────────────────────────────────────────────
```

### 完整的PPO Actor-Critic目标函数（深入讲解）

完整的PPO目标将**三个组件**组合成一个损失函数：

$$L^{PPO}(\theta, \phi) = L^{CLIP}(\theta) - c_1 L^{VF}(\phi) + c_2 S[\pi_\theta]$$

其中：
- $L^{CLIP}(\theta)$：**Actor损失**（通过截断代理进行策略改进）
- $L^{VF}(\phi)$：**Critic损失**（价值函数准确性）
- $S[\pi_\theta]$：**熵奖励**（鼓励探索）
- $c_1, c_2$：超参数系数（通常$c_1 = 0.5$，$c_2 = 0.01$）

让我们详细分解每个组件。

---

#### 组件1：Actor损失 $L^{CLIP}(\theta)$ — 策略改进

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

**目的**：通过增加好动作（正优势）的概率和减少坏动作（负优势）的概率来改进策略，同时使用截断保证稳定性。

**组件解释**：

| 符号 | 含义 | 公式 |
|------|------|------|
| $r_t(\theta)$ | 概率比 | $\frac{\pi_\theta(a_t \| s_t)}{\pi_{\theta_{old}}(a_t \| s_t)}$ |
| $\hat{A}_t$ | 估计优势 | 通常GAE：$\sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$ |
| $\epsilon$ | 截断范围 | 通常0.1或0.2 |

**为什么是最大化（不是最小化）？** 我们想**最大化**期望优势加权回报。在代码中，我们通常取负来创建要最小化的损失：

```python
# 实践中，我们最小化负值
actor_loss = -L_CLIP  # 最小化负值 = 最大化
```

**截断机制详解**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    截断如何创建"悲观"界限                                    │
└─────────────────────────────────────────────────────────────────────────────┘

对于每个具有优势A的(状态, 动作)对：

如果 A > 0（好动作 → 想增加概率）：
┌────────────────────────────────────────────────────────────────────────────┐
│  未截断: r(θ) · A      →  随着π_θ(a|s)增加而增加                           │
│  截断:   (1+ε) · A     →  常数上限                                        │
│                                                                            │
│  min(未截断, 截断):                                                        │
│    - 当 r < 1+ε: 使用未截断（梯度推动概率上升）                             │
│    - 当 r > 1+ε: 使用截断（无梯度，停止推动）                               │
│                                                                            │
│  结果: 概率可以增加，但目标在比率超过1+ε后停止改进。防止过冲。              │
└────────────────────────────────────────────────────────────────────────────┘

如果 A < 0（坏动作 → 想减少概率）：
┌────────────────────────────────────────────────────────────────────────────┐
│  未截断: r(θ) · A      →  r·A是负的，随着π_θ(a|s)减少                      │
│                           变得"更少负"（更好）                              │
│  截断:   (1-ε) · A     →  常数下限（可能达到的最小负值）                    │
│                                                                            │
│  min(未截断, 截断):                                                        │
│    - 当 r > 1-ε: 使用未截断（梯度推动概率下降）                             │
│    - 当 r < 1-ε: 使用截断（无梯度，停止推动）                               │
│                                                                            │
│  结果: 概率可以减少，但目标在比率低于1-ε后停止改进。防止过度抑制。          │
└────────────────────────────────────────────────────────────────────────────┘
```

**数学直觉**：min()操作创建了真实目标的**悲观下界**。我们只在截断边界内获得改进的信用。

---

#### 组件2：Critic损失 $L^{VF}(\phi)$ — 价值函数准确性

$$L^{VF}(\phi) = \mathbb{E}_t \left[ \left( V_\phi(s_t) - V_t^{target} \right)^2 \right]$$

**目的**：训练价值函数（critic）准确预测期望回报，这对计算好的优势估计至关重要。

**什么是$V_t^{target}$？** 目标值来自实际经验：

$$V_t^{target} = \hat{A}_t + V_{\phi_{old}}(s_t)$$

或等效地，使用回报：

$$V_t^{target} = \hat{R}_t = \sum_{l=0}^{T-t} \gamma^l r_{t+l}$$

**为什么用平方误差？** 简单，易理解，实践中效果好。一些实现使用Huber损失来增强对异常值的鲁棒性。

**可选：截断价值损失**

一些实现也对价值函数更新进行截断以提高稳定性：

$$L^{VF-CLIP}(\phi) = \mathbb{E}_t \left[ \max\left( (V_\phi - V^{target})^2, (V^{clip} - V^{target})^2 \right) \right]$$

其中$V^{clip} = V_{\phi_{old}} + \text{clip}(V_\phi - V_{\phi_{old}}, -\epsilon, \epsilon)$

```python
# 截断价值损失实现
v_pred = critic(states)
v_pred_clipped = v_old + torch.clamp(v_pred - v_old, -clip_range, clip_range)

loss_v1 = (v_pred - returns) ** 2
loss_v2 = (v_pred_clipped - returns) ** 2

critic_loss = 0.5 * torch.mean(torch.max(loss_v1, loss_v2))
```

**为什么也要截断critic？** 防止价值函数变化太剧烈，这可能会破坏后续epoch的优势估计稳定性。

---

#### 组件3：熵奖励 $S[\pi_\theta]$ — 探索

$$S[\pi_\theta] = \mathbb{E}_t \left[ -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) \right] = \mathbb{E}_t \left[ H(\pi_\theta(\cdot|s_t)) \right]$$

**目的**：通过奖励保持不确定性的策略（不要过快变得太确定）来鼓励探索。

**为什么熵很重要**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           熵与探索                                          │
└─────────────────────────────────────────────────────────────────────────────┘

高熵（有利于探索）：
┌────────────────────────────────┐
│  π(a₁|s) = 0.25               │
│  π(a₂|s) = 0.25               │     H(π) = -4 × 0.25 × log(0.25) = 1.39
│  π(a₃|s) = 0.25               │    （4个动作的最大值）
│  π(a₄|s) = 0.25               │
└────────────────────────────────┘

低熵（利用，可能过早）：
┌────────────────────────────────┐
│  π(a₁|s) = 0.97               │
│  π(a₂|s) = 0.01               │     H(π) ≈ 0.12
│  π(a₃|s) = 0.01               │    （几乎确定性）
│  π(a₄|s) = 0.01               │
└────────────────────────────────┘

熵奖励奖励高熵，防止在充分探索之前过早收敛到确定性策略。
```

**对于连续动作**（高斯策略）：

$$S[\pi_\theta] = \mathbb{E}_t \left[ \frac{1}{2} \log(2\pi e \sigma^2) \right] = \frac{1}{2}(1 + \log(2\pi\sigma^2))$$

熵取决于标准差$\sigma$。更大的$\sigma$ → 更多探索。

**熵系数$c_2$**：
- 太高：策略保持随机，无法利用好动作
- 太低：策略过快变得确定性，可能收敛到次优行为
- 典型值：离散动作0.01，连续动作0.001

---

#### 整合：组合损失

$$L^{TOTAL}(\theta, \phi) = -L^{CLIP}(\theta) + c_1 L^{VF}(\phi) - c_2 S[\pi_\theta]$$

注意符号（假设我们**最小化**损失）：
- **负** $L^{CLIP}$：我们想最大化策略改进
- **正** $L^{VF}$：我们想最小化价值预测误差
- **负** $S$：我们想最大化熵（鼓励探索）

**包含所有组件的完整算法**：

```
完整的PPO Actor-Critic训练循环：
═══════════════════════════════════════════════════════════════════════════════

输入: Actor π_θ, Critic V_φ, 系数c₁, c₂, 截断ε, epoch数K

for iteration = 1, 2, ... do:

    ┌─ 采集阶段 ──────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  for t = 1 to T do:                                                     │
    │      采样动作: a_t ~ π_θ_old(·|s_t)                                    │
    │      存储: log π_θ_old(a_t|s_t), V_φ_old(s_t)                          │
    │      执行动作，观察 r_t, s_{t+1}                                        │
    │                                                                         │
    │  使用GAE计算优势:                                                       │
    │      δ_t = r_t + γV_φ_old(s_{t+1}) - V_φ_old(s_t)                      │
    │      Â_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}                                │
    │                                                                         │
    │  计算回报:                                                              │
    │      R̂_t = Â_t + V_φ_old(s_t)                                          │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─ 优化阶段（在相同数据上K个epoch）─────────────────────────────────────────┐
    │                                                                         │
    │  for epoch = 1 to K do:                                                 │
    │      for minibatch in shuffle(collected_data) do:                       │
    │                                                                         │
    │          ┌─ ACTOR损失 ───────────────────────────────────────────────┐  │
    │          │  log π_θ_new = actor.log_prob(actions)                    │  │
    │          │  ratio = exp(log π_θ_new - log π_θ_old)                   │  │
    │          │                                                           │  │
    │          │  surr1 = ratio × Â                                        │  │
    │          │  surr2 = clip(ratio, 1-ε, 1+ε) × Â                        │  │
    │          │  L_actor = -mean(min(surr1, surr2))                       │  │
    │          └───────────────────────────────────────────────────────────┘  │
    │                                                                         │
    │          ┌─ CRITIC损失 ──────────────────────────────────────────────┐  │
    │          │  V_pred = critic(states)                                  │  │
    │          │  L_critic = mean((V_pred - R̂)²)                          │  │
    │          │                                                           │  │
    │          │  # 可选：截断价值损失                                      │  │
    │          │  V_clipped = V_old + clip(V_pred - V_old, -ε, ε)         │  │
    │          │  L_critic = mean(max((V_pred-R̂)², (V_clipped-R̂)²))      │  │
    │          └───────────────────────────────────────────────────────────┘  │
    │                                                                         │
    │          ┌─ 熵奖励 ──────────────────────────────────────────────────┐  │
    │          │  entropy = actor.entropy(states)                          │  │
    │          │  L_entropy = -mean(entropy)  # 负号以最大化               │  │
    │          └───────────────────────────────────────────────────────────┘  │
    │                                                                         │
    │          ┌─ 总损失 ──────────────────────────────────────────────────┐  │
    │          │  L_total = L_actor + c₁ × L_critic + c₂ × L_entropy       │  │
    │          │                                                           │  │
    │          │  optimizer.zero_grad()                                    │  │
    │          │  L_total.backward()                                       │  │
    │          │  optimizer.step()                                         │  │
    │          └───────────────────────────────────────────────────────────┘  │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
```

---

#### 完整PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class PPOActorCritic(nn.Module):
    """PPO的组合Actor-Critic网络。"""

    def __init__(self, state_dim, action_dim, hidden_dim=64, continuous=False):
        super().__init__()
        self.continuous = continuous

        # 共享特征提取器（可选，可以分开）
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor头
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic头
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        value = self.critic(features)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            return mean, std, value
        else:
            logits = self.actor(features)
            return logits, value

    def get_action_and_value(self, state, action=None):
        """获取动作、log_prob、熵和价值。"""
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = Normal(mean, std)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


def compute_ppo_loss(
    model: PPOActorCritic,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    clip_value: bool = True
) -> tuple[torch.Tensor, dict]:
    """
    计算完整的PPO损失。

    返回:
        total_loss: 要最小化的组合损失
        info: 包含各个损失组件的字典
    """
    # 获取当前策略输出
    _, new_log_probs, entropy, new_values = model.get_action_and_value(states, actions)

    # ==================== ACTOR损失 ====================
    # 概率比
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 截断代理目标
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    # Actor损失（负号因为我们最小化）
    actor_loss = -torch.mean(torch.min(surr1, surr2))

    # ==================== CRITIC损失 ====================
    if clip_value:
        # 截断价值损失
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, -clip_epsilon, clip_epsilon
        )
        value_loss1 = F.mse_loss(new_values, returns, reduction='none')
        value_loss2 = F.mse_loss(value_pred_clipped, returns, reduction='none')
        critic_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))
    else:
        # 简单MSE损失
        critic_loss = 0.5 * F.mse_loss(new_values, returns)

    # ==================== 熵奖励 ====================
    entropy_loss = -torch.mean(entropy)  # 负号以最大化熵

    # ==================== 总损失 ====================
    total_loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

    # 用于日志的信息
    info = {
        'actor_loss': actor_loss.item(),
        'critic_loss': critic_loss.item(),
        'entropy': -entropy_loss.item(),  # 报告正熵
        'total_loss': total_loss.item(),
        'ratio_mean': ratio.mean().item(),
        'ratio_min': ratio.min().item(),
        'ratio_max': ratio.max().item(),
        'clip_fraction': ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()
    }

    return total_loss, info
```

---

#### 为什么每个组件都重要：消融分析

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        移除每个组件的效果                                    │
└─────────────────────────────────────────────────────────────────────────────┘

没有截断（普通策略梯度）：
┌─────────────────────────────────────────────────────────────────────────────┐
│  问题: 大的策略更新 → 性能崩溃                                               │
│  症状: 训练开始良好，然后突然崩溃                                            │
│  原因: 单次坏的更新可能把策略推到远离好区域的地方                             │
└─────────────────────────────────────────────────────────────────────────────┘

没有价值函数（无critic）：
┌─────────────────────────────────────────────────────────────────────────────┐
│  问题: 优势估计方差高                                                        │
│  症状: 训练噪声大，收敛慢                                                    │
│  原因: 必须使用蒙特卡洛回报而不是TD估计                                      │
│  这本质上是带截断的REINFORCE                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

没有熵奖励：
┌─────────────────────────────────────────────────────────────────────────────┐
│  问题: 过早收敛到次优的确定性策略                                            │
│  症状: 策略停止探索，陷入局部最优                                            │
│  原因: 没有激励来维持动作多样性                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### 超参数敏感性

| 超参数 | 典型范围 | 太低的影响 | 太高的影响 |
|--------|---------|-----------|-----------|
| $\epsilon$（截断） | 0.1 - 0.3 | 太保守，学习慢 | 不稳定，违背PPO目的 |
| $c_1$（价值系数） | 0.5 - 1.0 | 价值估计差，方差高 | Critic主导，actor欠拟合 |
| $c_2$（熵系数） | 0.001 - 0.05 | 过早收敛 | 策略保持随机 |
| K（epoch数） | 3 - 10 | 数据利用不足 | 在旧数据上过拟合 |
| Minibatch大小 | 32 - 512 | 梯度噪声大 | 更新慢，内存问题 |
| $\gamma$（折扣） | 0.95 - 0.999 | 短视行为 | 信用分配困难 |
| $\lambda$（GAE） | 0.9 - 0.99 | 偏差高 | 方差高 |

---

### Token级 vs 响应级

LLM的PPO可以在不同粒度上操作：

**响应级**：将整个响应视为一个动作
- 更简单，但方差高
- 奖励分配给整个序列

**Token级**：每个token是一个动作
- 通过适当的信用分配降低方差
- 更昂贵（每个token一个"步骤"）

$$L^{token}(\theta) = \sum_{t=1}^{T} \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)$$

其中$r_t(\theta) = \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{old}(y_t|x, y_{<t})}$。

### PPO实现FAQ：常见困惑

这些是关于PPO实现的常见问题：

#### Q1: 我们需要保存两个独立的actor模型（旧的和新的）吗？

**不需要！** 你只需要**一个actor模型**。"旧"策略$\pi_{\theta_{old}}$不是一个独立的模型——它只是在采样时计算并**存储的对数概率**。

```
┌─────────────────────────────────────────────────────────────────────────┐
│  存储的 vs 计算的                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  采样时:                                                                 │
│  ┌──────────────┐                                                       │
│  │ Actor θ_old  │ → 生成样本 → 存储 log π_old(a|s) 作为数字             │
│  └──────────────┘                                                       │
│                                                                         │
│  训练时（同一模型，更新后的权重）:                                         │
│  ┌──────────────┐                                                       │
│  │ Actor θ_new  │ → 前向传播 → 实时计算 log π_new(a|s)                  │
│  └──────────────┘                                                       │
│                                                                         │
│  ratio = exp(log π_new - log π_old)  ← 使用存储的数字！                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**内存高效**：存储标量对数概率，而不是模型副本。

#### Q2: 在第一个训练步骤，还没更新模型，$\log \pi_{\theta_{new}}$是如何计算的？

在训练的第一次前向传播时：
- $\theta_{new} = \theta_{old}$（相同参数）
- $\log \pi_{new} = \log \pi_{old}$（相同值）
- **比率$r(\theta) = 1$（精确等于1）**

当你在训练循环中执行梯度更新时，$\theta$改变，所以后续的前向传播会给出不同的$\log \pi_{new}$值。

```python
# 显示流程的伪代码
log_p_old = []  # 采样时存储

# 采样阶段
for prompt in batch:
    response = actor.generate(prompt)
    log_p_old.append(actor.log_prob(response))  # 存储这些数字

# 训练阶段
for epoch in range(K_epochs):
    for minibatch in shuffle(samples):
        # 用当前权重前向传播（θ不断变化）
        log_p_new = actor.log_prob(minibatch.responses)  # 每次新计算

        # 使用存储的log_p_old（在此迭代中永不改变）
        ratio = torch.exp(log_p_new - minibatch.log_p_old)

        # ... 计算损失并更新θ
        optimizer.step()  # θ在这里改变！
        # 下一个minibatch: log_p_new会不同，log_p_old保持不变
```

#### Q3: 多次minibatch更新后，应该使用哪个$\log \pi_{old}$？

**始终使用采样时的对数概率。** 它们在训练期间永不改变。

```
时间线:
─────────────────────────────────────────────────────────────────────────

迭代 i:
┌─────────────┐
│ Actor_i     │ ──→ 收集样本 ──→ 存储 log_p_old（来自Actor_i）
└─────────────┘
      │
      │ Minibatch 1: 更新 → Actor_{i,1}
      │ Minibatch 2: 更新 → Actor_{i,2}    全部使用相同的log_p_old
      │ Minibatch 3: 更新 → Actor_{i,3}   （来自Actor_i）
      │ ...
      │ 第K个epoch完成
      ▼
┌─────────────┐
│ Actor_{i+1} │ ──→ 收集新样本 ──→ 存储新的log_p_old
└─────────────┘

─────────────────────────────────────────────────────────────────────────
```

**关键洞察**：`log_p_old`与样本绑定，而不是当前模型。当你收集新样本时，你会得到新的`log_p_old`值。

#### Q4: 每个epoch后，我需要为所有样本重新计算$\log \pi_{old}$吗？

**不需要！** 你在采样时**只存储一次**`log_p_old`。它保持固定。

| 内容 | 何时计算 | 频率 |
|------|---------|------|
| `log_p_old` | 样本采集时 | 每次迭代**一次** |
| `log_p_new` | 训练前向传播时 | **每个minibatch** |

```python
# 正确的实现
log_p_old = collect_and_store_log_probs(actor, prompts)  # 一次！

for epoch in range(K):
    for mb in minibatches:
        log_p_new = actor.log_prob(mb.responses)  # 每次新计算
        ratio = exp(log_p_new - mb.log_p_old)     # log_p_old来自存储
        # ... 更新
```

#### Q5: 所以$\log \pi_{old}$在所有epoch中都不更新？

**正确！** 在一个PPO迭代（在同一批样本上的所有K个epoch）内，`log_p_old`是**冻结的**。

```
PPO迭代结构:
─────────────────────────────────────────────────────────────────────────

 ┌─ 用当前策略收集样本 ────────────────────────────────────────────────┐
 │  存储: log_p_old, advantages, returns                              │
 │  这些在整个迭代期间都是冻结的                                         │
 └────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ┌─ 训练循环 ─────────────────────────────────────────────────────────┐
 │                                                                    │
 │  for epoch in range(K_epochs):        # K通常为3-10               │
 │      for minibatch in shuffle(data):                               │
 │          log_p_new = forward_pass()   # 每步都变                   │
 │          ratio = exp(log_p_new - log_p_old)  # log_p_old固定       │
 │          loss = clipped_objective(ratio, advantages)               │
 │          optimizer.step()             # θ更新                      │
 │                                                                    │
 │  # K个epoch后，log_p_old可能与log_p_new非常不同                     │
 │  # 这没问题！截断防止过大的更新                                      │
 └────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ┌─ 下一次迭代 ───────────────────────────────────────────────────────┐
 │  收集新样本 → 计算新的log_p_old → 重复                              │
 └────────────────────────────────────────────────────────────────────┘
```

#### Q6: 当比率被截断并选为最小值时，梯度如何流动？

**不流动！** 当选择截断值时，梯度为**零**。

$$L^{CLIP} = \min\left( \underbrace{r(\theta) A}_{\text{有梯度}}, \underbrace{\text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A}_{\text{截断时为常数}} \right)$$

**情况分析：**

| 条件 | 选择的值 | 梯度 |
|------|---------|------|
| $1-\epsilon < r < 1+\epsilon$ | $r \cdot A$（未截断） | $\nabla_\theta r \cdot A$（非零） |
| $r > 1+\epsilon$ 且 $A > 0$ | $(1+\epsilon) \cdot A$ | **0**（常数） |
| $r < 1-\epsilon$ 且 $A < 0$ | $(1-\epsilon) \cdot A$ | **0**（常数） |

```
梯度流动可视化:
─────────────────────────────────────────────────────────────────────────

当 A > 0（好动作，想增加概率）:

                梯度流动
                     │
    Loss             ▼
      │    ┌─────────────────┐
      │    │  r(θ) · A       │ ← 当 r < 1+ε 时选择
      │    └────────┬────────┘
      │             │
      └──→ min() ───┤
                    │
           ┌────────┴────────┐
           │ (1+ε) · A       │ ← 当 r > 1+ε 时选择（无梯度！）
           └─────────────────┘
                 常数

─────────────────────────────────────────────────────────────────────────
```

**数学原因**：
$$\frac{\partial}{\partial \theta} \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) = \begin{cases} \frac{\partial r}{\partial \theta} & \text{如果 } 1-\epsilon < r < 1+\epsilon \\ 0 & \text{否则（被截断）} \end{cases}$$

#### Q7: 所以当被截断并选中时，actor对该样本不更新？

**完全正确！** 这是PPO稳定性的核心机制。

**为什么这是好的：**

1. **防止灾难性更新**：如果策略已经改变很多（$r$远离1），停止继续推动
2. **自限制优化**：好动作被强化，但不是无限地
3. **稳定性**：即使在相同数据上多个epoch，策略也不能偏移太远

```
示例场景:
─────────────────────────────────────────────────────────────────────────

初始: log_p_old = -2.0, A = +5.0（好动作）

Epoch 1, Minibatch 1:
  log_p_new = -1.8  →  r = exp(-1.8 - (-2.0)) = 1.22
  r < 1+ε (1.2)?  否, 1.22 > 1.2  →  被截断！
  梯度 = 0, 该样本不更新

发生了什么？策略已经在之前的更新中足够增加了这个动作的概率。
截断说"够了，到此为止。"

─────────────────────────────────────────────────────────────────────────
```

**这就是PPO稳定的原因**：当策略相对于采集策略已经改变"足够多"时，它会自动停止更新。

---

## DPO: 直接偏好优化

### 关键洞察

DPO（Rafailov等，2023）问：*我们能完全跳过奖励模型吗？*

**关键观察**：RLHF目标的最优策略有闭式解！

### 数学推导

**步骤1：RLHF目标**

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r(x, y) \right] - \beta D_{KL}(\pi \| \pi_{ref})$$

展开KL散度：

$$= \mathbb{E}_{x, y \sim \pi} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right]$$

**步骤2：最优解**

对导数求零，最优策略是：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

其中$Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$是配分函数。

**步骤3：重新整理得到奖励**

求解$r(x, y)$：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**关键洞察**：奖励可以用策略比率表示！

**步骤4：Bradley-Terry模型**

人类偏好遵循Bradley-Terry模型：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

其中$\sigma$是sigmoid函数，$y_w \succ y_l$表示"$y_w$优于$y_l$"。

**步骤5：代入奖励**

将我们对$r$的表达式代入：

$$P(y_w \succ y_l | x) = \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

注意：$Z(x)$项抵消了！

$$= \sigma\left( \beta \log \frac{\pi^*(y_w|x) / \pi_{ref}(y_w|x)}{\pi^*(y_l|x) / \pi_{ref}(y_l|x)} \right)$$

### DPO损失

由于我们希望策略$\pi_\theta$匹配最优策略$\pi^*$，我们直接优化：

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

简化符号：

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E} \left[ \log \sigma\left( \beta (r_\theta(x, y_w) - r_\theta(x, y_l)) \right) \right]$$

其中$r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$是**隐式奖励**。

### 理解DPO梯度

求梯度：

$$\nabla_\theta \mathcal{L}_{DPO} = -\beta \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_l - \hat{r}_w)}_{\text{权重}} \left( \underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{增加 } y_w} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{减少 } y_l} \right) \right]$$

其中$\hat{r}_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$和$\hat{r}_l = \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$。

**解释**：
- 权重$\sigma(\hat{r}_l - \hat{r}_w)$：当模型错误排序时更高
- 效果：增加偏好$y_w$的概率，减少拒绝$y_l$的概率

### DPO算法

```
DPO算法:
─────────────────────────────────────────────────────────
输入: 参考策略π_ref, 偏好数据集D, β

初始化: π_θ ← π_ref

for iteration = 1, 2, ... do:
    1. 采样批次: (x, y_w, y_l) ~ D

    2. 计算对数概率:
       log π_θ(y_w|x), log π_θ(y_l|x)
       log π_ref(y_w|x), log π_ref(y_l|x)

    3. 计算隐式奖励:
       r_w = β(log π_θ(y_w|x) - log π_ref(y_w|x))
       r_l = β(log π_θ(y_l|x) - log π_ref(y_l|x))

    4. 计算损失:
       L = -log σ(r_w - r_l)

    5. 更新: θ ← θ - α∇_θL
─────────────────────────────────────────────────────────
```

### 为什么DPO有效

**数学等价性**：DPO求解与RLHF相同的优化：
- RLHF：训练奖励模型→用RL优化
- DPO：直接用闭式解优化

**隐式奖励**：$r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$衡量$y$在$\pi_\theta$下比$\pi_{ref}$下更可能多少。

---

## GRPO: 组相对策略优化

### 动机

GRPO（Shao等，2024）解决PPO和DPO的局限性：
- **PPO**：需要单独的评论家/价值网络
- **DPO**：限于成对偏好

### 核心思想

不是使用学习的价值函数（评论家），GRPO从对同一提示的**一组响应**估计优势。

### 基于组的优势估计

对于每个提示$x$，采样一组$G$个响应：$\{y_1, y_2, ..., y_G\} \sim \pi_{\theta_{old}}(y|x)$

计算每个的奖励：$\{r_1, r_2, ..., r_G\}$，其中$r_i = r_\phi(x, y_i)$

**优势估计**：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

这是组内的**相对排名** — 不需要绝对价值函数！

### 为什么要组归一化？

1. **消除基线偏差**：用均值中心化消除提示特定的难度
2. **归一化尺度**：除以std处理奖励大小差异
3. **无需评论家**：基线来自同批次响应

### GRPO目标

$$\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{x} \mathbb{E}_{\{y_i\}_{i=1}^G \sim \pi_{old}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( r_i(\theta) \hat{A}_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]$$

加上KL正则化：

$$\mathcal{L}_{GRPO}(\theta) = \mathcal{L}_{clip}(\theta) - \beta \cdot \mathbb{E}_{x, y \sim \pi_\theta} \left[ D_{KL}(\pi_\theta(y|x) \| \pi_{ref}(y|x)) \right]$$

### GRPO算法

```
GRPO算法:
─────────────────────────────────────────────────────────
输入: 参考策略π_ref, 奖励模型r_φ, 组大小G

初始化: π_θ ← π_ref

for iteration = 1, 2, ... do:
    1. 采样提示: {x_1, ..., x_B} ~ D

    2. 对每个提示x_b:
       生成G个响应: {y_b,1, ..., y_b,G} ~ π_θ_old(·|x_b)
       计算奖励: {r_b,1, ..., r_b,G}

       归一化优势:
       μ_b = mean({r_b,i})
       σ_b = std({r_b,i})
       A_b,i = (r_b,i - μ_b) / σ_b

    3. 计算概率比:
       r_θ(y_b,i) = π_θ(y_b,i|x_b) / π_θ_old(y_b,i|x_b)

    4. 计算截断目标:
       L = (1/BG) Σ_b Σ_i min(r_θ A_b,i, clip(r_θ, 1-ε, 1+ε) A_b,i)

    5. 添加KL惩罚:
       L_total = L - β · D_KL(π_θ || π_ref)

    6. 更新: θ ← θ + α∇_θL_total
─────────────────────────────────────────────────────────
```

### 与PPO的比较

| 方面 | PPO | GRPO |
|------|-----|------|
| 基线 | 学习的价值函数$V_\phi(s)$ | 组奖励均值 |
| 额外网络 | Actor + Critic | 仅Actor |
| 样本效率 | 可以重用样本 | 需要新鲜的组 |
| 方差 | 有好评论家时较低 | 取决于组大小 |

### 与排名的数学联系

GRPO的优势可以看作软排名：

如果$G=2$且有$(y_w, y_l)$，$r_w > r_l$：

$$\hat{A}_w = \frac{r_w - (r_w + r_l)/2}{\sqrt{(r_w - r_l)^2/2}} = \frac{r_w - r_l}{|r_w - r_l|} \cdot \frac{1}{\sqrt{2}} = \frac{1}{\sqrt{2}}$$

$$\hat{A}_l = -\frac{1}{\sqrt{2}}$$

这表明$G=2$的GRPO简化为成对排名的归一化版本！

---

## 方法比较

### 一览表

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           方法比较                                          │
└─────────────────────────────────────────────────────────────────────────────┘

                    需要奖励模型?    需要评论家?    需要参考策略?
                         │             │              │
TRPO                    是            是             可选
                         │             │              │
PPO                     是            是             是 (用于KL)
                         │             │              │
DPO                     否            否             是 (必需)
                         │             │              │
GRPO                    是            否             是 (用于KL)

─────────────────────────────────────────────────────────────────────────────

                    偏好格式        优化类型         复杂度
                         │              │               │
TRPO                奖励模型      约束优化            高
                                   (信任区域)
                         │              │               │
PPO                 奖励模型      截断代理            中
                         │              │               │
DPO                 成对偏好      对比学习            低
                    (y_w, y_l)    (分类)
                         │              │               │
GRPO                奖励模型      截断 +              中
                                   组基线
```

### 详细比较表

| 方法 | 需要奖励模型 | 需要评论家 | 偏好格式 | 训练稳定性 | 样本效率 | 实现复杂度 |
|------|------------|-----------|---------|-----------|---------|-----------|
| **TRPO** | 是 | 是 | 标量奖励 | 非常高（保证改进） | 中等 | 高（Fisher矩阵，线搜索） |
| **PPO** | 是 | 是 | 标量奖励 | 高 | 中等 | 中等 |
| **DPO** | 否 | 否 | 成对$(y_w, y_l)$ | 中等 | 高（离线） | 低 |
| **GRPO** | 是 | 否 | 标量奖励 | 高 | 中等 | 中等 |

### 优缺点

#### TRPO

**优点：**
- 单调改进的理论保证
- 训练非常稳定
- 适用于任何优势估计器

**缺点：**
- 计算昂贵（Fisher矩阵求逆）
- 线搜索增加开销
- 实现复杂
- 由于成本，很少用于LLM

#### PPO

**优点：**
- 实现简单
- 稳定性和性能的良好平衡
- 灵活——适用于任何奖励信号
- 可以在线学习（奖励模型随策略更新）
- 广泛测试和理解

**缺点：**
- 需要奖励模型（训练昂贵）
- 需要评论家网络（更多参数）
- 如果KL调得不好可能发生奖励黑客
- 样本效率低（同策略）

#### DPO

**优点：**
- 不需要奖励模型——更简单的流程
- 不需要评论家——更少参数
- 训练稳定——只是分类损失
- 样本效率高——在偏好数据上离线
- 数学上优雅

**缺点：**
- 限于成对偏好
- 不能整合任意奖励信号
- 参考策略必须固定（不能更新）
- 在分布外提示上可能不如PPO
- 对在线/迭代改进不够灵活

#### GRPO

**优点：**
- 不需要评论家网络
- 适用于奖励模型（灵活信号）
- 自然处理多个响应
- 适合基于结果的奖励（数学、代码）
- 比PPO简单（无需训练价值函数）

**缺点：**
- 需要为每个提示生成多个响应
- 方差取决于组大小$G$
- 比DPO样本效率低
- 仍然需要奖励模型

### 何时使用每种方法

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 最大理论保证 | TRPO | 保证改进（如果负担得起） |
| 通用对齐 | PPO | 灵活，易理解，性能好 |
| 离线训练有偏好数据 | DPO | 简单，高效，无奖励模型 |
| 可验证答案的数学/代码任务 | GRPO | 可以使用基于结果的奖励 |
| 计算有限 | DPO | 无奖励模型，无评论家 |
| 持续改进流程 | PPO | 可以随时间更新奖励模型 |
| 偏好数据少的冷启动 | PPO | 可以用简单奖励代理引导 |

### 权衡三角

```
                    简单性
                       /\
                      /  \
                     /    \
                    / DPO  \
                   /________\
                  /          \
                 /   GRPO    \
                /____________ \
               /              \
              /      PPO       \
             /_________________ \
            /                   \
           /       TRPO          \
          /_______________________\
         灵活性 ←────────→ 稳定性
```

- **TRPO**: 最大稳定性，中等灵活性，低简单性
- **PPO**: 三者平衡良好
- **DPO**: 最大简单性，有限灵活性，中等稳定性
- **GRPO**: 中等简单性，良好灵活性，良好稳定性

---

## 实现示例：DPO损失

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # log π_θ(y_w|x)
    policy_rejected_logps: torch.Tensor,  # log π_θ(y_l|x)
    ref_chosen_logps: torch.Tensor,       # log π_ref(y_w|x)
    ref_rejected_logps: torch.Tensor,     # log π_ref(y_l|x)
    beta: float = 0.1
) -> torch.Tensor:
    """
    计算DPO损失。

    参数:
        policy_chosen_logps: 选择响应在π_θ下的对数概率
        policy_rejected_logps: 拒绝响应在π_θ下的对数概率
        ref_chosen_logps: 选择响应在π_ref下的对数概率
        ref_rejected_logps: 拒绝响应在π_ref下的对数概率
        beta: 温度参数

    返回:
        DPO损失（标量）
    """
    # 计算隐式奖励
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO损失: -log σ(r_w - r_l)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss


def compute_sequence_logprob(logits, labels, mask):
    """计算序列的对数概率。"""
    # logits: (batch, seq_len, vocab_size)
    # labels: (batch, seq_len)
    # mask: (batch, seq_len) - 有效token为1，padding为0

    log_probs = F.log_softmax(logits, dim=-1)

    # 获取实际token的对数概率
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # 对序列求和，掩码掉padding
    sequence_log_probs = (token_log_probs * mask).sum(dim=-1)

    return sequence_log_probs
```

---

## 测验

<details>
<summary><strong>Q1: 为什么RLHF包含KL惩罚项？</strong></summary>

**答案**：KL惩罚防止**奖励黑客**。没有它，策略可以利用奖励模型的弱点生成高奖励但无意义的输出。KL项$D_{KL}(\pi_\theta \| \pi_{ref})$使优化后的策略保持接近参考（SFT）策略，确保模型保留其语言能力。

**关键方程**：
$$\max_\pi \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi \| \pi_{ref})$$

**常见陷阱**：$\beta$设得太低导致奖励黑客；太高阻止学习。

</details>

<details>
<summary><strong>Q2: 推导为什么DPO的隐式奖励是$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$。</strong></summary>

**答案**：从最优RLHF策略开始：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

两边取对数：
$$\log \pi^*(y|x) = \log \pi_{ref}(y|x) + \frac{r(x,y)}{\beta} - \log Z(x)$$

重新整理求$r$：
$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

在DPO中，我们参数化$\pi^* \approx \pi_\theta$，配分函数$Z(x)$在比较对时抵消。

**常见陷阱**：忘记$Z(x)$只在成对比较设置中抵消。

</details>

<details>
<summary><strong>Q3: PPO和GRPO估计优势的关键区别是什么？</strong></summary>

**答案**：

**PPO**使用学习的**价值函数**（评论家）：
$$A_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{(TD误差)}$$
或多步回报的GAE。

**GRPO**使用**组统计**：
$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

关键区别：
- PPO需要训练额外网络（评论家）
- GRPO需要每个提示多个样本（组）
- PPO的基线依赖状态；GRPO的依赖提示
- GRPO基线无偏但小组时可能有更高方差

</details>

<details>
<summary><strong>Q4: 为什么DPO不能像PPO那样轻松整合任意奖励信号？</strong></summary>

**答案**：DPO的推导假设奖励通过最优策略与参考策略的关系隐式定义：

$$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

这个推导需要：
1. 偏好数据$(y_w, y_l)$对
2. Bradley-Terry假设关于奖励如何与偏好相关

对于任意奖励信号（如代码执行结果、数学正确性），DPO需要：
- 将标量奖励转换为成对比较
- 假设奖励遵循Bradley-Terry模型
- 可能在转换中丢失信息

PPO通过策略梯度直接优化任何标量奖励。

</details>

<details>
<summary><strong>Q5: 什么使TRPO比PPO计算成本高？</strong></summary>

**答案**：TRPO需要：

1. **Fisher信息矩阵**：$F = \mathbb{E}[\nabla_\theta \log \pi \cdot \nabla_\theta \log \pi^T]$
   - 大小：$|\theta| \times |\theta|$ — 对大网络来说巨大！

2. **自然梯度**：$d = F^{-1}g$
   - 直接求逆是$O(|\theta|^3)$
   - 使用共轭梯度：$O(k \cdot |\theta|^2)$，其中$k$是CG迭代次数

3. **线搜索**：多次前向传播以找到满足KL约束的有效步长

**PPO的简化**：用截断替换约束
- 无需Fisher矩阵计算
- 无需矩阵求逆
- 无需线搜索
- 只是对截断目标的梯度下降

</details>

<details>
<summary><strong>Q6: 在GRPO中，如果组大小G太小会发生什么？</strong></summary>

**答案**：当$G$很小时：

1. **基线方差高**：少量样本的均值有噪声
2. **std估计有偏**：当$G=2$时，std只是$|r_1 - r_2|/\sqrt{2}$
3. **二值化优势**：当$G=2$时，优势本质上是$\pm$常数

**$G=2$的数学例子**：
$$\hat{A}_1 = \frac{r_1 - (r_1+r_2)/2}{\sigma} = \frac{r_1 - r_2}{2\sigma} = \pm\frac{1}{\sqrt{2}}$$

无论实际奖励差异大小如何！

**典型建议**：稳定训练$G \geq 4$；实践中$G = 8$-$16$常见。

</details>

---

## 参考文献

1. **TRPO**: Schulman等 (2015). "Trust Region Policy Optimization." ICML.
2. **PPO**: Schulman等 (2017). "Proximal Policy Optimization Algorithms." arXiv.
3. **RLHF**: Ouyang等 (2022). "Training language models to follow instructions with human feedback." NeurIPS.
4. **DPO**: Rafailov等 (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS.
5. **GRPO**: Shao等 (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv.

---

## 面试记忆要点

| 方法 | 关键方程 | 一句话总结 |
|------|---------|-----------|
| **TRPO** | $\max L^{CPI}$ s.t. $D_{KL} \leq \delta$ | 带信任区域的约束优化 |
| **PPO** | $\min(r_t A_t, \text{clip}(r_t) A_t)$ | 截断代理替换约束 |
| **DPO** | $-\log\sigma(\beta(r_w - r_l))$，$r = \log\frac{\pi_\theta}{\pi_{ref}}$ | 策略比率的隐式奖励 |
| **GRPO** | $A_i = (r_i - \mu) / \sigma$ | 组归一化优势，无评论家 |
