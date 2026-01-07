# Linear Function Approximation

## Interview Summary

**Linear function approximation** represents values as \(V(s) = \mathbf{w}^\top \boldsymbol{\phi}(s)\) where \(\boldsymbol{\phi}(s)\) are features and \(\mathbf{w}\) are learned weights. This enables generalization across states — similar states have similar values. Key advantage: theoretical guarantees exist (convergence for policy evaluation). Key challenge: requires good feature engineering.

**What to memorize**: Linear form \(\mathbf{w}^\top \boldsymbol{\phi}\), gradient update, convergence for on-policy TD, feature design importance.

---

## Core Definitions

### Linear Value Function

$$\hat{V}(s; \mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s) = \sum_{i=1}^{d} w_i \phi_i(s)$$

where:
- \(\boldsymbol{\phi}(s) \in \mathbb{R}^d\): Feature vector (hand-designed or learned)
- \(\mathbf{w} \in \mathbb{R}^d\): Weight vector (learned)

### Linear Q-Function

$$\hat{Q}(s, a; \mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s, a)$$

Features can be state-action dependent.

### Gradient

$$\nabla_\mathbf{w} \hat{V}(s; \mathbf{w}) = \boldsymbol{\phi}(s)$$

The gradient is just the feature vector — simple and cheap to compute.

---

## Math and Derivations

### Mean Squared Value Error (MSVE)

$$\text{MSVE}(\mathbf{w}) = \sum_s d^\pi(s) [V^\pi(s) - \hat{V}(s; \mathbf{w})]^2$$

where \(d^\pi(s)\) is the on-policy state distribution.

**Goal**: Find \(\mathbf{w}^*\) that minimizes MSVE.

### Gradient Descent Update

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\alpha}{2} \nabla_\mathbf{w} [V^\pi(s) - \hat{V}(s; \mathbf{w})]^2$$

$$= \mathbf{w} + \alpha [V^\pi(s) - \hat{V}(s; \mathbf{w})] \boldsymbol{\phi}(s)$$

**Problem**: We don't know \(V^\pi(s)\). Replace with:
- MC: Use return \(G_t\)
- TD: Use \(R_{t+1} + \gamma \hat{V}(S_{t+1}; \mathbf{w})\)

### TD(0) with Linear Function Approximation

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha [R_{t+1} + \gamma \hat{V}(S_{t+1}; \mathbf{w}) - \hat{V}(S_t; \mathbf{w})] \boldsymbol{\phi}(S_t)$$

**Note**: This is semi-gradient — we don't differentiate through the target \(\hat{V}(S_{t+1})\).

### Convergence Result

**Theorem**: Linear TD(0) with on-policy sampling converges to:

$$\mathbf{w}_{TD} = \arg\min_\mathbf{w} \text{MSPBE}(\mathbf{w})$$

where MSPBE is the Mean Squared Projected Bellman Error. This is close to but not exactly MSVE.

---

## Algorithm Sketch

```
Algorithm: Semi-gradient TD(0) with Linear FA

Input: α, γ, feature function φ
Output: Approximate V^π

1. Initialize w = 0 (or small random)
2. For each episode:
     S = initial state
     While S is not terminal:
         A = action from π(S)
         Take A, observe R, S'
         δ = R + γ w·φ(S') - w·φ(S)  # TD error
         w ← w + α · δ · φ(S)
         S ← S'
3. Return w
```

### Common Features

| Feature Type | Description | Example |
|--------------|-------------|---------|
| Polynomials | \(\phi = [1, s, s^2, \ldots]\) | Position + velocity in CartPole |
| Tile coding | Binary features from tiling | Discretize continuous state |
| RBF | Gaussian bumps | \(\exp(-\|s - c_i\|^2 / \sigma^2)\) |
| Fourier basis | Sinusoidal | \(\cos(\pi \mathbf{c}^\top \mathbf{s})\) |

---

## Common Pitfalls

1. **Bad features**: If features don't capture relevant state distinctions, learning fails. Feature engineering is crucial.

2. **Ignoring representation limit**: Linear FA can only represent values in the span of features. Some V-functions are unreachable.

3. **Off-policy divergence**: Linear TD can diverge with off-policy data (see deadly triad).

4. **Feature scaling**: Features with very different scales cause learning issues. Normalize features.

5. **Too many features**: Overfitting is possible, especially with limited data.

---

## Mini Example

**Mountain Car with Position Feature**:

State: (position, velocity). Simple feature: \(\boldsymbol{\phi}(s) = [\text{position}, \text{velocity}, 1]\) (3D).

After learning:

$$\hat{V}(s) = w_1 \cdot \text{position} + w_2 \cdot \text{velocity} + w_3$$

- High position (near goal) → higher V
- High velocity (toward goal) → higher V

**Limitation**: Can't represent nonlinear value landscapes. Tile coding or neural nets needed for complex relationships.

---

## Quiz

<details markdown="1">
<summary><strong>Q1 (Conceptual):</strong> Why use function approximation instead of tabular methods?</summary>

**Answer**: Generalization across states:
1. Large/infinite state spaces can't use tables
2. Similar states should have similar values
3. Faster learning by sharing information

**Explanation**: Tabular methods treat each state independently. With 1 billion states, you'd need 1 billion parameters and visits. Function approximation uses shared parameters — updating one state affects others.

**Key insight**: The trade-off is representational capacity vs generalization.

**Common pitfall**: Assuming more parameters is always better. Too many parameters → overfitting, too few → underfitting.
</details>

<details markdown="1">
<summary><strong>Q2 (Conceptual):</strong> Why is the update called "semi-gradient"?</summary>

**Answer**: We only differentiate part of the loss. The TD target \(R + \gamma \hat{V}(S'; \mathbf{w})\) contains \(\mathbf{w}\), but we treat it as constant.

**Explanation**: Full gradient would be:

$$\nabla_\mathbf{w} [R + \gamma \hat{V}(S') - \hat{V}(S)]^2$$

But we only use \(\nabla_\mathbf{w} \hat{V}(S)\), ignoring \(\nabla_\mathbf{w} \hat{V}(S')\).

**Why**: Differentiating the target leads to complications (double sampling, bias). Semi-gradient is simpler and works well in practice.

**Common pitfall**: Thinking semi-gradient is wrong. It converges for on-policy linear TD!
</details>

<details markdown="1">
<summary><strong>Q3 (Math):</strong> Derive the semi-gradient TD(0) update.</summary>

**Answer**: Starting from MSVE:

$$L = [V^\pi(s) - \hat{V}(s; \mathbf{w})]^2$$

Gradient (using TD target as proxy for \(V^\pi\)):

$$\nabla_\mathbf{w} L \approx -2[R + \gamma \hat{V}(S') - \hat{V}(S)] \nabla_\mathbf{w} \hat{V}(S)$$

$$= -2 \delta \cdot \boldsymbol{\phi}(S)$$

Update:

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\alpha}{2} \nabla_\mathbf{w} L = \mathbf{w} + \alpha \delta \cdot \boldsymbol{\phi}(S)$$

**Key equation**: \(\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \boldsymbol{\phi}(S)\)

**Common pitfall**: Including \(\boldsymbol{\phi}(S')\) in the gradient — that's for full gradient methods.
</details>

<details markdown="1">
<summary><strong>Q4 (Math):</strong> What is the representational capacity of linear FA?</summary>

**Answer**: Values must lie in the span of features:

$$\hat{V}(s) \in \text{span}\{\phi_1, \ldots, \phi_d\}$$

If true \(V^\pi\) is not in this span, we get approximation error (bias).

**Explanation**: Linear FA projects \(V^\pi\) onto the feature subspace. The best we can do is minimize distance to this projection.

**Example**: With \(\boldsymbol{\phi} = [1, s]\), we can only represent lines. If \(V^\pi(s) = s^2\), we get a linear approximation.

**Common pitfall**: Assuming enough features always works. The features must span the right subspace.
</details>

<details markdown="1">
<summary><strong>Q5 (Practical):</strong> Your linear FA agent isn't learning. What to check?</summary>

**Answer**: Debugging steps:
1. **Feature quality**: Are features distinguishing relevant states?
2. **Feature scaling**: Normalize to similar magnitudes
3. **Learning rate**: Start with small α (0.01)
4. **On-policy sampling**: Off-policy can diverge
5. **Feature representation**: Can the target value be represented?
6. **Initialization**: Zero init is usually safe

**Explanation**: Most failures are feature-related. Try visualizing learned weights and what values they produce.

**Common pitfall**: Blaming the algorithm when the features are bad.
</details>

---

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction, Chapters 9-10
- **Tsitsiklis & Van Roy (1997)**, An Analysis of TD(λ) with Function Approximation
- **Sutton (1988)**, Learning to Predict by the Methods of Temporal Differences

**What to memorize for interviews**: Linear form, gradient \(= \phi(s)\), semi-gradient update, on-policy convergence, feature importance.

**Code example**: [linear_fa.py](../../../rl_examples/algorithms/linear_fa.py)
