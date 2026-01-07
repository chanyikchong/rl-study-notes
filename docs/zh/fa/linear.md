# 线性函数逼近

## 面试摘要

**线性函数逼近** 将价值表示为 \(V(s) = \mathbf{w}^\top \boldsymbol{\phi}(s)\)，其中 \(\boldsymbol{\phi}(s)\) 是特征，\(\mathbf{w}\) 是学习的权重。这使得跨状态泛化成为可能——相似的状态有相似的价值。关键优势：存在理论保证（策略评估收敛）。关键挑战：需要好的特征工程。

**需要记忆的**：线性形式 \(\mathbf{w}^\top \boldsymbol{\phi}\)，梯度更新，同策略TD的收敛，特征设计的重要性。

---

## 核心定义

### 线性价值函数

$$\hat{V}(s; \mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s) = \sum_{i=1}^{d} w_i \phi_i(s)$$

其中：
- \(\boldsymbol{\phi}(s) \in \mathbb{R}^d\)：特征向量（手工设计或学习）
- \(\mathbf{w} \in \mathbb{R}^d\)：权重向量（学习）

### 线性Q函数

$$\hat{Q}(s, a; \mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s, a)$$

特征可以依赖于状态-动作对。

### 梯度

$$\nabla_\mathbf{w} \hat{V}(s; \mathbf{w}) = \boldsymbol{\phi}(s)$$

梯度就是特征向量——简单且计算便宜。

---

## 数学与推导

### 均方价值误差（MSVE）

$$\text{MSVE}(\mathbf{w}) = \sum_s d^\pi(s) [V^\pi(s) - \hat{V}(s; \mathbf{w})]^2$$

其中 \(d^\pi(s)\) 是同策略状态分布。

**目标**：找到最小化MSVE的 \(\mathbf{w}^*\)。

### 梯度下降更新

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\alpha}{2} \nabla_\mathbf{w} [V^\pi(s) - \hat{V}(s; \mathbf{w})]^2$$

$$= \mathbf{w} + \alpha [V^\pi(s) - \hat{V}(s; \mathbf{w})] \boldsymbol{\phi}(s)$$

**问题**：我们不知道 \(V^\pi(s)\)。替换为：
- MC：使用回报 \(G_t\)
- TD：使用 \(R_{t+1} + \gamma \hat{V}(S_{t+1}; \mathbf{w})\)

### 带线性函数逼近的TD(0)

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha [R_{t+1} + \gamma \hat{V}(S_{t+1}; \mathbf{w}) - \hat{V}(S_t; \mathbf{w})] \boldsymbol{\phi}(S_t)$$

**注意**：这是半梯度——我们不对目标 \(\hat{V}(S_{t+1})\) 求微分。

### 收敛结果

**定理**：带同策略采样的线性TD(0)收敛到：

$$\mathbf{w}_{TD} = \arg\min_\mathbf{w} \text{MSPBE}(\mathbf{w})$$

其中MSPBE是均方投影贝尔曼误差。这接近但不完全是MSVE。

---

## 算法概述

```
算法：带线性FA的半梯度TD(0)

输入：α, γ, 特征函数 φ
输出：近似 V^π

1. 初始化 w = 0（或小随机值）
2. 对每个回合：
     S = 初始状态
     当 S 不是终止状态时：
         A = 从 π(S) 的动作
         执行 A，观察 R, S'
         δ = R + γ w·φ(S') - w·φ(S)  # TD误差
         w ← w + α · δ · φ(S)
         S ← S'
3. 返回 w
```

### 常见特征

| 特征类型 | 描述 | 示例 |
|--------------|-------------|---------|
| 多项式 | \(\phi = [1, s, s^2, \ldots]\) | CartPole中的位置+速度 |
| 瓦片编码 | 来自瓦片化的二值特征 | 离散化连续状态 |
| RBF | 高斯凸起 | \(\exp(-\|s - c_i\|^2 / \sigma^2)\) |
| 傅里叶基 | 正弦 | \(\cos(\pi \mathbf{c}^\top \mathbf{s})\) |

---

## 常见陷阱

1. **坏特征**：如果特征没有捕获相关的状态区分，学习会失败。特征工程至关重要。

2. **忽视表示限制**：线性FA只能表示特征张成空间中的价值。一些V函数无法达到。

3. **异策略发散**：线性TD在异策略数据下可能发散（见致命三角）。

4. **特征缩放**：尺度非常不同的特征导致学习问题。归一化特征。

5. **特征太多**：过拟合是可能的，特别是数据有限时。

---

## 小例子

**带位置特征的Mountain Car**：

状态：(位置, 速度)。简单特征：\(\boldsymbol{\phi}(s) = [\text{位置}, \text{速度}, 1]\)（3D）。

学习后：

$$\hat{V}(s) = w_1 \cdot \text{位置} + w_2 \cdot \text{速度} + w_3$$

- 高位置（接近目标）→ 更高V
- 高速度（朝向目标）→ 更高V

**限制**：无法表示非线性价值景观。需要瓦片编码或神经网络处理复杂关系。

---

## 测验

<details markdown="1">
<summary><strong>问题1（概念）：</strong> 为什么使用函数逼近而不是表格方法？</summary>

**答案**：跨状态泛化：
1. 大/无限状态空间无法使用表格
2. 相似状态应该有相似价值
3. 通过共享信息更快学习

**解释**：表格方法独立处理每个状态。有10亿个状态，你需要10亿个参数和访问。函数逼近使用共享参数——更新一个状态影响其他状态。

**关键洞见**：权衡是表示能力vs泛化。

**常见陷阱**：假设更多参数总是更好。参数太多→过拟合，太少→欠拟合。
</details>

<details markdown="1">
<summary><strong>问题2（概念）：</strong> 为什么更新被称为"半梯度"？</summary>

**答案**：我们只对损失的一部分求微分。TD目标 \(R + \gamma \hat{V}(S'; \mathbf{w})\) 包含 \(\mathbf{w}\)，但我们将其视为常数。

**解释**：完整梯度应该是：

$$\nabla_\mathbf{w} [R + \gamma \hat{V}(S') - \hat{V}(S)]^2$$

但我们只使用 \(\nabla_\mathbf{w} \hat{V}(S)\)，忽略 \(\nabla_\mathbf{w} \hat{V}(S')\)。

**为什么**：对目标求微分导致复杂性（双重采样，偏差）。半梯度更简单，实践中效果好。

**常见陷阱**：认为半梯度是错误的。它对同策略线性TD收敛！
</details>

<details markdown="1">
<summary><strong>问题3（数学）：</strong> 推导半梯度TD(0)更新。</summary>

**答案**：从MSVE开始：

$$L = [V^\pi(s) - \hat{V}(s; \mathbf{w})]^2$$

梯度（使用TD目标作为 \(V^\pi\) 的代理）：

$$\nabla_\mathbf{w} L \approx -2[R + \gamma \hat{V}(S') - \hat{V}(S)] \nabla_\mathbf{w} \hat{V}(S)$$

$$= -2 \delta \cdot \boldsymbol{\phi}(S)$$

更新：

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\alpha}{2} \nabla_\mathbf{w} L = \mathbf{w} + \alpha \delta \cdot \boldsymbol{\phi}(S)$$

**关键方程**：\(\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \boldsymbol{\phi}(S)\)

**常见陷阱**：在梯度中包含 \(\boldsymbol{\phi}(S')\)——那是完整梯度方法的。
</details>

<details markdown="1">
<summary><strong>问题4（数学）：</strong> 线性FA的表示能力是什么？</summary>

**答案**：价值必须在特征的张成空间中：

$$\hat{V}(s) \in \text{span}\{\phi_1, \ldots, \phi_d\}$$

如果真正的 \(V^\pi\) 不在这个张成空间中，我们得到逼近误差（偏差）。

**解释**：线性FA将 \(V^\pi\) 投影到特征子空间。我们能做的最好是最小化到这个投影的距离。

**例子**：用 \(\boldsymbol{\phi} = [1, s]\)，我们只能表示直线。如果 \(V^\pi(s) = s^2\)，我们得到线性逼近。

**常见陷阱**：假设足够的特征总是有效。特征必须张成正确的子空间。
</details>

<details markdown="1">
<summary><strong>问题5（实践）：</strong> 你的线性FA智能体没有学习。检查什么？</summary>

**答案**：调试步骤：
1. **特征质量**：特征是否区分了相关状态？
2. **特征缩放**：归一化到相似的量级
3. **学习率**：从小的α开始（0.01）
4. **同策略采样**：异策略可能发散
5. **特征表示**：目标价值能被表示吗？
6. **初始化**：零初始化通常是安全的

**解释**：大多数失败与特征相关。尝试可视化学习的权重和它们产生的价值。

**常见陷阱**：当特征不好时责怪算法。
</details>

---

## 参考文献

- **Sutton & Barto**, 强化学习：导论，第9-10章
- **Tsitsiklis & Van Roy (1997)**, 带函数逼近的TD(λ)分析
- **Sutton (1988)**, 通过时序差分方法学习预测

**面试需要记忆的**：线性形式，梯度 \(= \phi(s)\)，半梯度更新，同策略收敛，特征的重要性。

**代码示例**：[linear_fa.py](../../../rl_examples/algorithms/linear_fa.py)
