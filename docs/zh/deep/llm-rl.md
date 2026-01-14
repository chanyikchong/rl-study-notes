# LLM训练的RL方法

!!! info "页面已重组"
    此内容已重组为单独的页面以便更好导航。请访问以下页面：

## 各方法详细页面

RLHF方法已按发布日期拆分为专门页面：

1. **[RLHF概述](rlhf-overview.md)** — RLHF介绍
   - 为什么用RL训练LLM
   - RLHF流程
   - 方法对比和何时使用

2. **[TRPO (2015)](trpo.md)** — 信任域策略优化
   - 策略优化的理论基础
   - 单调改进保证
   - 自然梯度和Fisher信息矩阵

3. **[DPO (2023)](dpo.md)** — 直接偏好优化
   - 绕过奖励模型
   - 隐式奖励推导
   - Bradley-Terry模型联系

4. **[GRPO (2024)](grpo.md)** — 组相对策略优化
   - 基于组的优势估计
   - 无需评论家
   - 与排序损失的联系

---

## 快速参考

| 方法 | 年份 | 关键创新 | 何时使用 |
|------|------|----------|----------|
| [TRPO](trpo.md) | 2015 | 信任域约束 | 需要最大稳定性 |
| [PPO](ppo.md) | 2017 | 裁剪代理 | 通用RLHF |
| [DPO](dpo.md) | 2023 | 隐式奖励 | 计算资源有限、离线数据 |
| [GRPO](grpo.md) | 2024 | 组归一化 | 可验证任务（数学/代码） |

---

## 关键公式总结

| 方法 | 核心公式 |
|------|----------|
| **RLHF目标** | $\max_\pi \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi \|\| \pi_{ref})$ |
| **TRPO** | $\max L^{CPI}$ s.t. $D_{KL} \leq \delta$ |
| **PPO** | $L = \min(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t)$ |
| **DPO** | $L = -\log\sigma(\beta(r_w - r_l))$ 其中 $r = \log\frac{\pi_\theta}{\pi_{ref}}$ |
| **GRPO** | $A_i = (r_i - \mu) / \sigma$（组归一化） |
