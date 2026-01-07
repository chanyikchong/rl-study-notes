# 实践训练流程

## 面试摘要

真实世界的RL训练需要的不仅是算法：**种子** 用于可重复性，**日志** 用于调试，**评估** 与训练分离，**超参数调优**，以及广泛的**调试技能**。常见问题：奖励bug，环境错误，错误的超参数，实现bug。大多数RL失败是工程问题，不是算法问题。

**需要记忆的**：可重复性实践，记录什么，常见调试模式，超参数调优策略。

---

## 核心定义

### 训练 vs 评估

- **训练**：智能体学习（探索，更新）
- **评估**：测量真实性能（贪婪，无更新）

**为什么分开？** 训练性能包括探索（ε-贪婪），这会降低回报。评估显示实际学到的行为。

### 可重复性

用相同代码和种子获得相同结果：
1. 随机种子（numpy, torch, env）
2. 确定性操作
3. 代码和数据的版本控制
4. 记录所有超参数

### 关键跟踪指标

| 指标 | 显示什么 |
|--------|--------------|
| 回合回报 | 智能体性能 |
| 回合长度 | 学会生存 |
| Q值大小 | 稳定性（不应爆炸）|
| 策略熵 | 探索水平 |
| 损失值 | 学习进度 |
| 梯度范数 | 训练稳定性 |

---

## 数学与推导

### 用于可重复性的种子设置

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 完全确定性（更慢）：
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 环境种子
env = gym.make("CartPole-v1")
env.seed(seed)
env.action_space.seed(seed)
```

### 评估协议

```python
def evaluate(agent, env, n_episodes=10, seed=42):
    """不带探索或学习地评估智能体。"""
    env.seed(seed)
    returns = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_return = 0
        done = False

        while not done:
            action = agent.get_action(state, greedy=True)  # 无探索
            state, reward, done, info = env.step(action)
            episode_return += reward

        returns.append(episode_return)

    return np.mean(returns), np.std(returns)
```

### 学习曲线

带置信区间绘图：
```
跨多个种子的均值回报 ± std/sqrt(n_seeds)
```

最少3个种子，发表最好5-10个。

---

## 算法概述

### 标准训练循环

```python
# 设置
set_seed(args.seed)
env = make_env(args.env)
agent = make_agent(args)
logger = setup_logging(args)

# 训练
for step in range(args.total_steps):
    # 收集经验
    action = agent.get_action(state, explore=True)
    next_state, reward, done, info = env.step(action)
    agent.buffer.add(state, action, reward, next_state, done)

    # 更新智能体
    if step > args.warmup_steps:
        metrics = agent.update()
        logger.log(metrics)

    # 周期性评估
    if step % args.eval_freq == 0:
        eval_return, eval_std = evaluate(agent, eval_env)
        logger.log({"eval_return": eval_return, "eval_std": eval_std})
        logger.save_checkpoint(agent, step)

    # 如果完成则重置
    if done:
        state = env.reset()
    else:
        state = next_state
```

### 超参数调优顺序

1. **学习率**：最敏感。尝试1e-5, 1e-4, 1e-3
2. **网络大小**：从小开始（2层，64-256单元）
3. **批量大小**：典型32-256
4. **折扣γ**：默认0.99，调优范围0.9-0.999
5. **探索**：ε计划，熵系数
6. **算法特定**：回放缓冲区，目标更新等

### 调试层级

```
1. 检查奖励信号
   → 打印奖励，验证它们符合预期
   → 常见bug：错误符号，缺少终止奖励

2. 检查环境
   → 渲染并手动交互
   → 验证状态空间，动作效果

3. 检查学习基础
   → 损失在减少吗？梯度在流动吗？
   → 智能体能记住单个转移吗？

4. 检查探索
   → 智能体探索了足够状态吗？
   → 打印状态访问统计

5. 检查超参数
   → 从已知有效的设置开始
   → 一次调一个参数
```

---

## 常见陷阱

1. **不设置种子**：运行之间结果变化很大。总是设置种子。

2. **训练时评估**：训练回报有噪声（探索）。使用单独的贪婪评估。

3. **错误的奖励函数**：最常见的bug。手动验证奖励。

4. **终止状态处理**：忘记设置V(terminal)=0会破坏价值学习。

5. **超参数盲从**：论文超参数可能不适合你的设置。为你的环境调优。

6. **日志不够**：当事情失败时，你需要数据。记录所有东西。

7. **同时改多个东西**：一次调试一个改变。

---

## 小例子

**CartPole训练示例：**

```python
# 验证有效的CartPole-v1超参数
config = {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 10000,
    "buffer_size": 10000,
    "target_update": 100,
    "hidden_size": 128,
    "total_steps": 50000,
    "eval_freq": 1000,
}

# 预期结果：
# - 在约10K-20K步内解决（回报 > 195）
# - Q值稳定在20-50左右
# - Epsilon在10K步内从1.0衰减到0.01
```

**当它不工作时，检查：**
1. 奖励被正确接收吗？（打印验证）
2. 探索在发生吗？（打印epsilon，动作分布）
3. Q值合理吗？（解决的CartPole应该约20-50）

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 为什么评估必须与训练分离？</summary>

**答案**：训练包括探索（随机动作）这会降低测量性能。评估应该使用贪婪策略来测量真实能力。

**解释**：用ε=0.1的ε-贪婪，10%动作是随机的 → 更低回报。用ε=0的评估显示学到的策略实际达到什么。

**另外**：训练更新网络，创造非平稳性。评估是当前策略的快照。

**常见陷阱**：报告训练回报作为结果。总是报告贪婪评估。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> 为什么RL调试比监督学习难？</summary>

**答案**：几个原因：
1. **没有ground truth**：无法将预测与标签比较
2. **延迟信号**：奖励可能比动作晚很多
3. **非平稳**：策略变化 → 数据分布变化
4. **探索混淆**：差结果可能是探索，不是坏策略
5. **许多失败模式**：环境bug，奖励bug，算法bug，超参数

**解释**：在监督学习中，如果损失不降，有问题。在RL中，损失可能降但策略可能差（对缓冲区过拟合）或损失增加但学习在发生。

**常见陷阱**：假设损失行为指示学习。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 如何知道Q值是否合理？</summary>

**答案**：用折扣奖励和做健全性检查：

对于CartPole（最大500步，每步奖励=1，γ=0.99）：

$$Q_{max} \approx \frac{1}{1-\gamma} = \frac{1}{0.01} = 100$$

对于在T步结束的回合任务：

$$Q_{max} \approx \sum_{t=0}^{T} \gamma^t r_{max}$$

**如果Q >> 期望最大值**：过估计，可能发散
**如果Q ≈ 0 总是**：没在学习（检查奖励，梯度）

**常见陷阱**：不检查Q值大小。容易的诊断经常被忽略。
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> 应该运行多少个种子才能得到可靠结果？</summary>

**答案**：最少3个，发表质量结果最好5-10个。

**解释**：RL跨种子方差大因为：
- 随机初始化
- 探索随机性
- 环境随机性

标准误差随√n种子减少。用5个种子，你可以估计均值 ± 0.45σ（95%置信）。

**最佳实践**：报告均值 ± 标准误差，或在图中显示所有单独运行。

**常见陷阱**：报告单种子结果（挑选最好的种子）。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> 你的智能体在一个环境学习但在另一个失败。尝试什么？</summary>

**答案**：迁移清单：

1. **尺度差异**：归一化观测和奖励
   ```python
   obs = (obs - obs_mean) / obs_std
   reward = np.clip(reward, -10, 10)
   ```

2. **动作空间**：不同范围需要不同缩放

3. **回合长度**：更长回合需要更小学习率，更大缓冲区

4. **奖励稀疏性**：稀疏奖励需要更多探索（好奇心，更高ε）

5. **状态复杂度**：更复杂状态需要更大网络

6. **超参数敏感性**：为新环境重新调优

**解释**：RL算法惊人地环境特定。CartPole有效的可能Atari失败。

**常见陷阱**：假设一组超参数到处有效。
</details>

---

## 参考文献

- **Henderson et al. (2018)**, 深度RL的重要性
- **Engstrom et al. (2020)**, 深度RL中实现很重要
- **Andrychowicz et al. (2020)**, 同策略RL中什么重要
- **Islam et al. (2017)**, 基准深度RL任务的可重复性

**面试需要记忆的**：种子实践，评估vs训练，调试层级，超参数调优顺序，常见问题。
