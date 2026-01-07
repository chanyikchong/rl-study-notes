# 期望SARSA

## 面试摘要

**期望SARSA** 使用所有可能下一动作的期望Q值，按策略概率加权：\(\sum_{a'} \pi(a'|s') Q(s', a')\)。相比SARSA（采样一个动作）降低了方差，同时保持同策略语义。当目标策略是贪婪时，期望SARSA变成Q学习。它是一个灵活的中间方案。

**需要记忆的**：带期望的更新规则，相对于SARSA的方差降低，与Q学习的关系（贪婪情况），同策略但更低方差。

---

## 核心定义

### 期望SARSA更新规则

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a'} \pi(a'|S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

**关键洞见**：不是采样 \(A'\) 并使用 \(Q(S', A')\)，而是计算所有动作的期望。

### 与其他算法的关系

| 算法 | 目标使用 |
|-----------|-------------|
| SARSA | \(Q(S', A')\) 其中 \(A' \sim \pi\) |
| 期望SARSA | \(\sum_{a'} \pi(a' \mid S') Q(S', a')\) |
| Q学习 | \(\max_{a'} Q(S', a')\) |

**观察**：Q学习是带贪婪目标策略的期望SARSA。

---

## 数学与推导

### 方差降低

SARSA使用单个样本：

$$\text{SARSA目标} = R + \gamma Q(S', A')$$

期望SARSA使用真正的期望：

$$\text{期望SARSA目标} = R + \gamma \sum_{a'} \pi(a'|S') Q(S', a')$$

**方差比较**：
- SARSA有来自采样 \(A'\) 的方差
- 期望SARSA消除了这个方差（只剩转移方差）

### 贝尔曼方程关系

期望SARSA向以下方向更新：

$$Q^\pi(s, a) = \mathbb{E}[R_{t+1}] + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s', a')$$

这正是 \(Q^\pi\) 的贝尔曼期望方程。

### 收敛性

期望SARSA收敛到 \(Q^\pi\)（策略 \(\pi\) 的Q函数）。如果 \(\pi\) 是ε-贪婪且 ε → 0，它收敛到 \(Q^*\)。

---

## 算法概述

```
算法：期望SARSA

输入：α, γ, ε
输出：Q ≈ Q^π

1. 任意初始化 Q(s,a)（Q(terminal, ·) = 0）
2. 对每个回合：
     S = 初始状态
     当 S 不是终止状态时：
         A = 从 Q(S, ·) 的 ε-贪婪动作
         执行动作 A，观察 R, S'
         expected_q = Σ_a' π(a'|S') Q(S', a')
         Q(S,A) ← Q(S,A) + α[R + γ · expected_q - Q(S,A)]
         S ← S'
3. 返回 Q
```

### 计算期望

对于ε-贪婪策略：

$$\sum_{a'} \pi(a'|s') Q(s', a') = (1-\epsilon) \max_{a'} Q(s', a') + \frac{\epsilon}{|A|} \sum_{a'} Q(s', a')$$

**解释**：贪婪动作权重为 \((1-\epsilon)\)，\(\epsilon\) 均匀分布。

---

## 常见陷阱

1. **大动作空间的开销**：需要对所有动作求和。SARSA只采样一个。

2. **与Q学习混淆**：期望SARSA使用 \(\pi\)（可能是探索性的）；Q学习使用贪婪。

3. **同策略语义**：尽管计算了期望，它仍然是同策略的（学习 \(Q^\pi\)，不是 \(Q^*\)）。

4. **策略必须已知**：与SARSA不同，你需要知道动作概率来计算期望。

---

## 小例子

**2动作老虎机设置**（单状态）：

动作：A，B。\(Q(A) = 10\)，\(Q(B) = 5\)。策略：ε = 0.2的ε-贪婪。

**SARSA**：采样 \(A'\)。80%概率选A（给 \(Q = 10\)），20%概率选B（给 \(Q = 5\)）。

**期望SARSA**：

$$\mathbb{E}[Q] = 0.9 \times 10 + 0.1 \times 5 + 0.1 \times 10 + 0.9 \times 5 \, ?$$

等等，让我重新计算：
- ε-贪婪：prob(A) = 0.8 + 0.1 = 0.9？不对。
- 当 ε = 0.2 时：prob(贪婪) = 0.8，prob(每个随机) = 0.1
- 如果A是贪婪的：prob(A) = 0.8 + 0.2/2 = 0.9，prob(B) = 0.2/2 = 0.1

$$\mathbb{E}[Q] = 0.9 \times 10 + 0.1 \times 5 = 9.5$$

**关键点**：期望SARSA总是得到9.5；SARSA有时得到10，有时得到5，但平均是9.5。

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 期望SARSA如何比SARSA降低方差？</summary>

**答案**：SARSA采样一个动作 \(A'\) 并使用 \(Q(S', A')\)。期望SARSA计算精确期望 \(\sum_{a'} \pi(a'|S') Q(S', a')\)，消除采样方差。

**解释**：\(A'\) 的随机性给SARSA贡献方差。通过对所有可能的 \(A'\) 取平均，期望SARSA移除了这个噪声源。

**关键洞见**：只剩转移方差（\(S'\) 的随机性）。

**常见陷阱**：认为更低方差总是更好。在大动作空间中，计算期望代价高。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> 期望SARSA何时等价于Q学习？</summary>

**答案**：当目标策略 \(\pi\) 相对于Q是贪婪的时候。

**解释**：如果 \(\pi(a'|s') = 1\) 对于 \(a' = \arg\max Q(s', a')\)，否则为0：

$$\sum_{a'} \pi(a'|s') Q(s', a') = \max_{a'} Q(s', a')$$

这正是Q学习的目标。

**关键洞见**：Q学习是带贪婪目标策略的期望SARSA的特例。

**常见陷阱**：认为Q学习和期望SARSA是根本不同的算法。
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 推导ε-贪婪策略的期望SARSA目标。</summary>

**答案**：对于ε-贪婪，设 \(a^* = \arg\max_{a'} Q(s', a')\)：

$$\sum_{a'} \pi(a'|s') Q(s', a') = (1-\epsilon) Q(s', a^*) + \epsilon \cdot \frac{1}{|A|} \sum_{a'} Q(s', a')$$

**简化**：设 \(\bar{Q}(s') = \frac{1}{|A|} \sum_{a'} Q(s', a')\)（平均Q）：

$$= (1-\epsilon) \max_{a'} Q(s', a') + \epsilon \cdot \bar{Q}(s')$$

**解释**：贪婪动作以概率 \((1-\epsilon)\) 被选中，随机动作以概率 \(\epsilon\)。

**常见陷阱**：忘记随机部分对每个动作有概率 \(\epsilon/|A|\)，不是 \(\epsilon\)。
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> 期望SARSA vs SARSA的计算成本是多少？</summary>

**答案**：
- **SARSA**：目标计算O(1)（只查找一个Q值）
- **期望SARSA**：目标计算O(|A|)（对所有动作求和）

**解释**：每次更新，期望SARSA必须遍历所有动作。在大动作空间（例如连续）中，没有近似这是不可行的。

**权衡**：更低方差但每次更新计算更高。

**常见陷阱**：在非常大的动作空间使用期望SARSA而不考虑成本。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> 你应该何时优先选择期望SARSA而不是SARSA或Q学习？</summary>

**答案**：优先选择期望SARSA当：
1. 动作空间小（期望计算便宜）
2. 你想要更低方差的同策略学习
3. 你在使用函数逼近（稳定性重要）
4. 策略是已知/可控的

当动作空间大、同策略可以接受时优先选择SARSA
当你想要最优Q的异策略学习时优先选择Q学习

**解释**：期望SARSA是SARSA的方差减少版本。当你能承担计算成本时是个好的默认选择。

**常见陷阱**：当期望SARSA可能更稳定时默认使用Q学习。
</details>

---

## 参考文献

- **Sutton & Barto**, 强化学习：导论，第6.6章
- **Van Seijen et al. (2009)**, 期望Sarsa的理论和实证分析

**面试需要记忆的**：带期望的更新规则，方差降低机制，与Q学习的关系，ε-贪婪期望公式。

**代码示例**：[expected_sarsa.py](../../../rl_examples/algorithms/expected_sarsa.py)
