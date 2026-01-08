# 深度Q网络（DQN）

## 面试摘要

**DQN** 将Q学习与深度神经网络结合，解决高维问题（例如从像素玩Atari）。两个关键创新：**经验回放**（存储和采样过去的转移）和**目标网络**（通过使用冻结的Q作为目标来稳定训练）。第一个在Atari游戏上达到人类水平性能的算法。现代基于价值的深度RL的基础。

**需要记忆的**：DQN损失函数，回放缓冲区目的，目标网络目的，致命三元组，为什么两个技巧都是必要的。

---

## 设计动机：为什么需要DQN

### 梦想：Q学习 + 神经网络

Q学习对小问题效果很好。我们能否直接用神经网络来表示Q？

$$Q(s, a) \approx Q(s, a; \theta)$$

**朴素方法**：运行Q学习，但用神经网络代替表格。

```python
# 朴素深度Q学习（不工作！）
target = r + gamma * max(Q(next_state; theta))
loss = (target - Q(state, action; theta))^2
theta = theta - lr * gradient(loss)
```

### 为什么朴素深度Q学习失败

**问题1：样本相关**
- Q学习更新使用连续转移：$(s_t, a_t, r_t, s_{t+1})$
- 连续样本高度相关
- 神经网络 + 相关数据 = 不稳定训练
- SGD假设i.i.d.样本！

**问题2：移动目标**
- 目标：$y = r + \gamma \max_{a'} Q(s', a'; \theta)$
- 每次梯度步骤改变 $\theta$
- 这改变目标 $y$
- 我们在追逐移动的目标！

```
更新 θ → Q变化 → 目标变化 → 更新 θ → ...
          （这个反馈循环导致发散！）
```

**问题3：致命三元组**
Sutton & Barto指出，结合这三个会导致不稳定：
1. **函数逼近**（神经网络）
2. **自举**（使用Q估计作为目标）
3. **异策略学习**（从回放缓冲区学习）

DQN有所有三个！那它如何工作？

### DQN的解决方案：两个简单技巧

| 问题 | 解决方案 |
|---------|----------|
| 相关样本 | **经验回放**：存储转移，随机采样 |
| 移动目标 | **目标网络**：冻结目标Q，周期性更新 |

这两个技巧将不稳定的深度Q学习转变为稳定的算法。

---

## 核心定义

### DQN架构

**输入**：状态 $s$（例如Atari的84×84×4堆叠帧）
**输出**：所有动作 $a \in A$ 的 $Q(s, a; \theta)$

```
状态（像素） → 卷积层 → 全连接层 → 每个动作的Q值
                     ↓
              [Q(s,左), Q(s,右), Q(s,上), Q(s,下)]
```

**关键洞见**：一次输出所有动作的Q值。一次前向传播，然后取argmax。

### 经验回放缓冲区

在固定大小的缓冲区 $D$ 中存储转移 $(s, a, r, s', \text{done})$。
**均匀随机**采样小批量用于训练。

**为什么有效：**

| 没有回放 | 有回放 |
|----------------|-------------|
| 在 $(s_1, s_2, s_3, ...)$ 上训练 | 在随机 $(s_{42}, s_{1337}, s_{7}, ...)$ 上训练 |
| 样本相关 | 样本近似i.i.d. |
| 忘记旧经验 | 多次重用经验 |
| 高方差 | 更低方差 |

**类比**：像通过随机复习卡片来准备考试，而不是按顺序重新阅读课本。

### 目标网络

维护**两个**网络：
- **在线网络** $Q(s, a; \theta)$：每步更新
- **目标网络** $Q(s, a; \theta^-)$：冻结，每 $C$ 步更新

$$\theta^- \leftarrow \theta \quad \text{每 } C \text{ 步}$$

**为什么有效：**

| 没有目标网络 | 有目标网络 |
|------------------------|---------------------|
| 目标 = $r + \gamma \max Q(s'; \theta)$ | 目标 = $r + \gamma \max Q(s'; \theta^-)$ |
| 目标每步变化 | 目标稳定 $C$ 步 |
| 追逐移动目标 | 稳定的回归问题 |
| 振荡/发散 | 收敛 |

**类比**：像射击一个暂时静止然后移动的目标。比射击持续移动的目标容易得多。

---

## 数学与推导

### DQN损失函数

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\left[ \left( y^{DQN} - Q(s, a; \theta) \right)^2 \right]$$

其中目标是：

$$y^{DQN} = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

**重要**：$y^{DQN}$ 被视为**常数**（没有梯度流过 $\theta^-$）。

### 理解梯度

$$\nabla_\theta L = \mathbb{E}\left[ -2 \left( y^{DQN} - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

**注意**：梯度只通过预测 $Q(s,a;\theta)$，不通过目标！

### 为什么经验回放在数学上有帮助

标准Q学习梯度：

$$\nabla_\theta L = (y - Q(s_t, a_t; \theta)) \nabla_\theta Q(s_t, a_t; \theta)$$

有回放，我们对许多样本取平均：

$$\nabla_\theta L = \frac{1}{B} \sum_{i=1}^{B} (y_i - Q(s_i, a_i; \theta)) \nabla_\theta Q(s_i, a_i; \theta)$$

通过从缓冲区均匀采样，样本近似i.i.d.，所以这是期望梯度的有效蒙特卡洛估计。

### Huber损失（平滑L1）

MSE对异常值敏感。Huber损失更稳健：

$$L_\delta(x) = \begin{cases} \frac{1}{2}x^2 & |x| \leq \delta \\ \delta(|x| - \frac{\delta}{2}) & \text{否则} \end{cases}$$

- 接近零时二次（像MSE）→ 平滑梯度
- 大误差时线性 → 不会因异常值爆炸

---

## 算法概述

```
算法：DQN

超参数：γ（折扣），ε（探索），N（缓冲区大小），
         B（批量大小），C（目标更新频率）

1. 随机初始化Q网络 θ
2. 初始化目标网络 θ^- = θ
3. 初始化回放缓冲区 D（容量 N）

4. 对 episode = 1 到 M：
     s = env.reset()
     对 t = 1 到 T：
         # ε-贪婪动作选择
         以概率 ε：a = 随机动作
         否则：a = argmax_a Q(s, a; θ)

         # 环境步骤
         s', r, done = env.step(a)

         # 存储转移（经验回放）
         存储 (s, a, r, s', done) 到 D
         s = s'

         # 学习步骤
         从 D 随机采样 B 个转移的小批量

         对每个 (s_j, a_j, r_j, s'_j, done_j)：
             如果 done_j：
                 y_j = r_j                          # 终止：无未来
             否则：
                 y_j = r_j + γ max_a' Q(s'_j, a'; θ^-)   # 目标网络

         # 梯度下降
         L = (1/B) Σ (y_j - Q(s_j, a_j; θ))²
         θ ← θ - α∇_θ L

         # 目标网络更新（每 C 步）
         每 C 步：θ^- ← θ

         如果 done：break
```

### 关键设计选择解释

| 设计选择 | 为什么？ |
|--------------|------|
| 输出所有动作的Q | 单次前向传播，高效argmax |
| 均匀回放采样 | 近似i.i.d.样本 |
| 冻结目标网络 | 稳定的回归目标 |
| ε-贪婪探索 | 简单，对离散动作有效 |
| 帧堆叠（Atari） | 捕获运动、速度信息 |
| 奖励裁剪（Atari） | 防止大奖励的梯度爆炸 |

### 关键超参数

| 超参数 | 典型值 | 为什么这个值 |
|----------------|---------------|----------------|
| 回放缓冲区大小 | 1M | 足够多样，不太占内存 |
| 批量大小 | 32 | 标准小批量大小 |
| 目标更新（C） | 10K步 | 足够稳定，不太陈旧 |
| 学习率 | 2.5e-4 | 低以防止不稳定 |
| ε衰减 | 1.0 → 0.1 在1M步内 | 早期探索，后期利用 |
| γ | 0.99 | 重视长期奖励 |

---

## DQN扩展：思想演进

```
DQN (2015)
  ↓ 问题：高估Q值
Double DQN (2016)
  ↓ 问题：不是所有经验同等重要
优先经验回放 (2016)
  ↓ 问题：价值和优势混在一起
Dueling DQN (2016)
  ↓ 结合所有改进
Rainbow (2017) - 以上所有 + 更多
```

### Double DQN

**问题**：$\max_{a'} Q(s', a'; \theta^-)$ 会高估，因为噪声估计的max有向上偏差。

**解决方案**：解耦动作选择和评估：

$$y^{DDQN} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

- 用在线网络 $\theta$ **选择**最佳动作
- 用目标网络 $\theta^-$ **评估**该动作

### 优先经验回放

**问题**：均匀采样在"简单"转移上浪费时间。

**解决方案**：按TD误差比例采样：

$$P(i) \propto |\delta_i|^\alpha$$

高惊喜度的转移被更频繁采样。

### Dueling DQN

**问题**：Q值 = 动作有多好 + 状态有多好。这两个混在一起。

**解决方案**：将Q分解为价值和优势：

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')$$

---

## 常见陷阱

1. **缓冲区太小**：需要多样性。Atari用1M，简单环境用10K+。小缓冲区 = 对近期经验过拟合。

2. **目标网络更新太频繁**：如果每步更新，你会失去稳定性好处。尝试每1000-10000步。

3. **ε衰减太快**：早期需要探索。如果ε太快到0.1，你可能错过好策略。

4. **忘记奖励裁剪**：Atari将奖励裁剪到{-1, 0, +1}。没有这个，梯度爆炸。

5. **缺少帧堆叠**：对于视觉输入，单帧不显示速度/方向。堆叠4帧。

6. **忽略done标志**：终止状态必须有 $y = r$，不是 $r + \gamma Q(s')$。否则你从不存在的状态自举！

7. **梯度通过目标**：确保没有梯度流过 $\theta^-$。使用 `.detach()` 或 `stop_gradient()`。

---

## 小例子

**CartPole DQN：**

```python
# 网络
q_network = MLP(state_dim=4, hidden=[128, 128], output_dim=2)
target_network = copy(q_network)
buffer = ReplayBuffer(capacity=10000)

for episode in range(500):
    state = env.reset()
    while not done:
        # ε-贪婪动作选择
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(q_network(state))

        next_state, reward, done = env.step(action)
        buffer.add(state, action, reward, next_state, done)

        # 训练（如果缓冲区有足够样本）
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size=32)

            # 计算目标（没有梯度通过target_network！）
            with no_grad():
                next_q = target_network(batch.next_states).max(dim=1)
                targets = batch.rewards + gamma * next_q * (1 - batch.dones)

            # 计算预测
            predictions = q_network(batch.states)[batch.actions]

            # MSE损失和梯度步骤
            loss = MSE(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 周期性更新目标网络
        if steps % 100 == 0:
            target_network.load_state_dict(q_network.state_dict())

        state = next_state
```

**预期**：在约100-200个回合内解决CartPole。

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 什么是"致命三元组"，DQN如何存活？</summary>

**答案**：致命三元组是以下组合：
1. **函数逼近**（神经网络）
2. **自举**（在目标中使用Q估计）
3. **异策略学习**（从回放缓冲区学习）

这种组合通常导致发散。DQN通过以下方式存活：
- **经验回放**：减少相关性，使数据更接近i.i.d.
- **目标网络**：稳定自举目标

**关键洞见**：DQN没有消除致命三元组——它通过这两个技巧来管理它。

**常见陷阱**：认为任一技巧单独就足够。你需要**两个**。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> 为什么经验回放打破相关性，为什么这很重要？</summary>

**答案**：
- 没有回放：在 $(s_1, s_2, s_3, ...)$ 上训练——连续、相关
- 有回放：在随机 $(s_{42}, s_{1337}, s_7, ...)$ 上训练——近似i.i.d.

**为什么重要**：SGD理论假设i.i.d.样本。相关样本导致梯度在特定方向偏置，导致：
- 对近期经验过拟合
- 忘记早期经验
- 不稳定训练

**类比**：只复习考试前最后一章 vs 随机复习所有章节。

**常见陷阱**：使用太小的缓冲区。如果缓冲区只有100个转移，样本仍然相关。
</details>

<details markdown="1">
<summary><strong>问题3（概念）：</strong> 用类比解释为什么我们需要目标网络。</summary>

**答案**：想象学习射箭时目标每次你射击都移动。

**没有目标网络**：
- 你瞄准位置X，射击
- 在你的箭落地之前，目标移动到Y
- 你调整瞄准到Y，射击
- 目标移动到Z...
- 你永远无法收敛因为你总在追逐

**有目标网络**：
- 目标在X保持100次射击
- 你学会稳定命中X
- 然后目标移动到X'
- 你调整并学习X'
- 逐步改进！

**关键洞见**：临时稳定性允许学习。

**常见陷阱**：更新目标太频繁会抵消好处。
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> 写出DQN损失并解释为什么梯度不流过目标。</summary>

**答案**：

$$L(\theta) = \mathbb{E}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

**为什么没有梯度通过目标**：
- $\theta^-$ 是冻结副本，不是 $\theta$ 的函数
- 在代码中：`target = r + gamma * target_network(s').max()` 使用 `target_network`，它有独立参数
- 我们只想让 $Q(s,a;\theta)$ 向目标移动，不是让目标向 $Q$ 移动

**如果我们包含通过目标的梯度**：两者都会移动，创建反馈循环 → 不稳定。

**常见陷阱**：在实现中忘记 `.detach()` 或 `stop_gradient()`。
</details>

<details markdown="1">
<summary><strong>问题5（数学）：</strong> 什么是Double DQN，为什么标准DQN会高估？</summary>

**答案**：标准DQN使用 $\max_{a'} Q(s', a'; \theta^-)$，这会高估。

**为什么高估**：
- Q值是噪声估计
- 噪声值的 $\max$ > 真实max（Jensen不等式对凸函数）
- 误差通过自举累积

**Double DQN解决方案**：

$$y^{DDQN} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

- 用在线网络 $\theta$ **选择**动作
- 用目标网络 $\theta^-$ **评估**
- 如果选择过于乐观，用不同参数评估会纠正它

**常见陷阱**：认为你需要两个独立网络。只需不同地使用 $\theta$ 和 $\theta^-$。
</details>

<details markdown="1">
<summary><strong>问题6（实践）：</strong> 你的DQN训练曲线显示初始学习然后性能崩溃。诊断。</summary>

**答案**：可能的原因：

1. **回放缓冲区溢出**：旧的好经验被推出，只剩最近（可能差的）经验
   - 修复：更大缓冲区，或优先回放

2. **目标网络太陈旧**：目标基于非常旧的策略
   - 修复：更频繁更新目标（但不是每步）

3. **ε衰减太多**：卡在利用次优策略
   - 修复：更慢的ε衰减，或周期性探索提升

4. **Q值爆炸**：检查Q值是否无限增长
   - 修复：梯度裁剪，奖励缩放，Huber损失

**调试技巧**：
- 绘制Q值随时间变化（应该稳定，不爆炸）
- 绘制回放缓冲区统计
- 检查ε计划

**常见陷阱**：没有记录足够的指标。总是跟踪Q值、损失、缓冲区大小。
</details>

---

## 参考文献

- **Mnih et al. (2015)**, 通过深度强化学习实现人类水平控制（Nature DQN论文）
- **Van Hasselt et al. (2016)**, 带双Q学习的深度RL
- **Schaul et al. (2016)**, 优先经验回放
- **Wang et al. (2016)**, 对决网络架构
- **Hessel et al. (2017)**, Rainbow：结合深度RL的改进

**面试需要记忆的**：DQN损失，为什么经验回放（i.i.d.），为什么目标网络（稳定目标），致命三元组，Double DQN，关键超参数。

**代码示例**：[dqn.py](../../../rl_examples/algorithms/dqn.py)
