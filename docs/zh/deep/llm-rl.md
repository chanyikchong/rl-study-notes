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
