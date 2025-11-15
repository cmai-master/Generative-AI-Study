# Normalizing Flows 완전 정복

> **학습 목표**: Change of Variables를 완전히 이해하고, RealNVP, Glow를 직접 구현할 수 있다
>
> **난이도**: 중급-고급 (Intermediate to Advanced)
> **예상 학습 시간**: 2-3주
> **선수 지식**: 미적분학, 선형대수, 확률론, PyTorch

---

## 📚 목차

1. [Normalizing Flows 기초](#1-normalizing-flows-기초)
2. [Change of Variables Theorem](#2-change-of-variables-theorem)
3. [Invertible Neural Networks](#3-invertible-neural-networks)
4. [Coupling Layers](#4-coupling-layers)
5. [RealNVP 구현](#5-realnvp-구현)
6. [Glow: Generative Flow](#6-glow-generative-flow)
7. [Continuous Normalizing Flows (Neural ODE)](#7-continuous-normalizing-flows-neural-ode)
8. [Flow Matching](#8-flow-matching)
9. [실습 프로젝트](#9-실습-프로젝트)

---

## 1. Normalizing Flows 기초

### 1.1 정의

**Normalizing Flow**는 간단한 분포(예: 가우시안)를 **가역 변환(invertible transformation)**으로 복잡한 분포로 변환하는 생성 모델입니다.

#### 핵심 아이디어

```
Simple Distribution (z)  →  [Invertible Transform]  →  Complex Distribution (x)
      z ~ p_z(z)                    f: z ↔ x                    x ~ p_x(x)
```

**특징:**
- **Exact likelihood**: 정확한 확률 밀도 계산 가능
- **Efficient sampling**: z ~ N(0, I) 샘플링 후 x = f(z)
- **Bi-directional**: x → z (inference), z → x (generation)

### 1.2 왜 Normalizing Flows인가?

#### 다른 생성 모델과의 비교

| 모델 | Exact Likelihood | Easy Sampling | Latent Space |
|------|-----------------|---------------|--------------|
| **VAE** | ✗ (ELBO only) | ✅ | ✅ (smooth) |
| **GAN** | ✗ | ✅ | ✗ (mode collapse) |
| **Diffusion** | ✗ (variational) | ✗ (slow) | ✅ |
| **Flows** | ✅ **Exact!** | ✅ | ✅ |

**장점:**
- Tractable likelihood → MLE 학습
- Exact inference: p(z|x) 계산 가능
- Stable training (GAN처럼 불안정하지 않음)

**단점:**
- Architectural constraints (invertibility)
- Computational cost (Jacobian determinant)

### 1.3 기본 예시: Linear Flow

가장 간단한 flow:

$$
f(\mathbf{z}) = \mathbf{Wz} + \mathbf{b}
$$

**역변환:**
$$
f^{-1}(\mathbf{x}) = \mathbf{W}^{-1}(\mathbf{x} - \mathbf{b})
$$

하지만 linear는 표현력이 약함 → **Nonlinear invertible transforms** 필요!

---

## 2. Change of Variables Theorem

### 2.1 1D Change of Variables

#### 기본 정리

확률 변수 z ~ p_z(z)가 있고, x = f(z)일 때:

$$
p_x(x) = p_z(f^{-1}(x)) \left| \frac{dz}{dx} \right|
$$

#### 직관

**확률 보존의 법칙:**
```
작은 구간 [z, z+dz]의 확률 = 변환된 구간 [x, x+dx]의 확률

p_z(z) dz = p_x(x) dx

p_x(x) = p_z(z) |dz/dx|
```

#### 예시

**z ~ N(0, 1), x = 2z + 3**

```python
# Forward
x = 2 * z + 3

# Inverse
z = (x - 3) / 2

# dz/dx
dz_dx = 1/2

# Probability
p_x(x) = p_z((x-3)/2) * (1/2)
       = N((x-3)/2; 0, 1) * 0.5
```

### 2.2 Multidimensional Change of Variables

#### Jacobian Matrix

**z** ∈ ℝ^d → **x** = f(**z**) ∈ ℝ^d

**Jacobian matrix** J:

$$
J = \frac{\partial \mathbf{x}}{\partial \mathbf{z}} = \begin{bmatrix}
\frac{\partial x_1}{\partial z_1} & \cdots & \frac{\partial x_1}{\partial z_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial x_d}{\partial z_1} & \cdots & \frac{\partial x_d}{\partial z_d}
\end{bmatrix}
$$

#### Change of Variables Formula

$$
p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f^{-1}(\mathbf{x})) \left| \det \frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

또는:

$$
p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(\mathbf{z}) \left| \det \frac{\partial f(\mathbf{z})}{\partial \mathbf{z}} \right|^{-1}
$$

**Log probability:**

$$
\log p_{\mathbf{x}}(\mathbf{x}) = \log p_{\mathbf{z}}(\mathbf{z}) - \log \left| \det \frac{\partial f(\mathbf{z})}{\partial \mathbf{z}} \right|
$$

### 2.3 여러 Flow 합성

K개의 변환 f₁, f₂, ..., f_K를 합성:

$$
\mathbf{x} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z})
$$

**Log probability:**

$$
\log p_{\mathbf{x}}(\mathbf{x}) = \log p_{\mathbf{z}}(\mathbf{z}) - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial \mathbf{z}_k} \right|
$$

**핵심:**
- 여러 간단한 변환을 쌓아서 복잡한 분포 생성
- Log-det-Jacobian을 모두 더함

### 2.4 Jacobian Determinant 계산

#### 문제

d차원에서 det(J) 계산은 **O(d³)** 복잡도!

**해결책:**
1. **Triangular Jacobian**: O(d)
2. **Block-diagonal Jacobian**: 병렬화
3. **Hutchinson's trace estimator**: 근사

#### Triangular Jacobian

J가 삼각 행렬이면:

$$
\det(J) = \prod_{i=1}^{d} J_{ii}
$$

**예시:**

$$
J = \begin{bmatrix}
2 & 0 & 0 \\
5 & 3 & 0 \\
1 & 4 & 6
\end{bmatrix} \implies \det(J) = 2 \times 3 \times 6 = 36
$$

---

## 3. Invertible Neural Networks

### 3.1 요구사항

Flow를 위한 함수 f는:

1. **Invertible (가역)**: f⁻¹이 존재
2. **Efficient computation**: f와 f⁻¹을 빠르게 계산
3. **Tractable Jacobian**: det(∂f/∂z)을 효율적으로 계산

### 3.2 간단한 Invertible Transforms

#### 1. Element-wise Transforms

**Activation functions:**

```python
# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return torch.where(x > 0, x, alpha * x)

def leaky_relu_inverse(y, alpha=0.01):
    return torch.where(y > 0, y, y / alpha)

# Jacobian (diagonal)
def leaky_relu_log_det(x, alpha=0.01):
    log_det = torch.where(x > 0, 0.0, math.log(alpha))
    return log_det.sum(dim=-1)
```

#### 2. Permutations

**차원 섞기:**

```python
# Fixed permutation
perm = torch.randperm(d)
x = z[..., perm]

# Inverse
z = x[..., torch.argsort(perm)]

# Log-det-Jacobian = 0 (permutation matrix의 determinant = ±1)
```

#### 3. Linear Transformations

```python
# A를 LU 분해
# A = PLU (P: permutation, L: lower triangular, U: upper triangular)

class LinearFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        W = torch.randn(dim, dim)
        self.W_LU = nn.Parameter(W)  # LU parameterization

    def forward(self, z):
        x = torch.matmul(z, self.W_LU.T)
        log_det = torch.slogdet(self.W_LU)[1].expand(z.size(0))
        return x, log_det

    def inverse(self, x):
        W_inv = torch.inverse(self.W_LU)
        z = torch.matmul(x, W_inv.T)
        return z
```

---

## 4. Coupling Layers

### 4.1 Affine Coupling Layer (RealNVP의 핵심)

#### 아이디어

입력을 두 부분으로 나누고, 한쪽으로 다른 쪽을 변환:

```
z = [z₁, z₂]  (split)
    ↓
x₁ = z₁  (identity, 그대로)
x₂ = z₂ ⊙ exp(s(z₁)) + t(z₁)  (affine transform)
    ↓
x = [x₁, x₂]  (concatenate)
```

여기서:
- s(·): scale 함수 (neural network)
- t(·): translation 함수 (neural network)
- ⊙: element-wise product

#### 왜 이렇게?

**Invertibility:**
```python
# Forward
x1 = z1
x2 = z2 * torch.exp(s(z1)) + t(z1)

# Inverse (쉽게 계산 가능!)
z1 = x1
z2 = (x2 - t(x1)) / torch.exp(s(x1))
```

**Jacobian:**

$$
J = \begin{bmatrix}
I & 0 \\
\frac{\partial x_2}{\partial z_1} & \text{diag}(\exp(s(z_1)))
\end{bmatrix}
$$

**Triangular!** → det(J) = exp(s(z₁)의 합)

$$
\log |\det J| = \sum_i s_i(z_1)
$$

**O(d) 복잡도!**

### 4.2 Additive Coupling Layer (NICE)

**더 간단한 버전** (s=0인 경우):

$$
x_1 = z_1, \quad x_2 = z_2 + t(z_1)
$$

**Jacobian determinant = 1** (volume-preserving)

$$
\log |\det J| = 0
$$

### 4.3 Masking

어떻게 z를 z₁, z₂로 나눌까?

#### Checkerboard Masking (이미지용)

```
Image (spatial):
[1, 0, 1, 0]
[0, 1, 0, 1]
[1, 0, 1, 0]
[0, 1, 0, 1]

1: 그대로 (z₁)
0: 변환 (z₂)
```

#### Channel-wise Masking

```
Channels [C₁, C₂, ..., C_d]
         ↓
z₁ = [C₁, C₂, ..., C_{d/2}]
z₂ = [C_{d/2+1}, ..., C_d]
```

### 4.4 Coupling Layer 구현

```python
import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=256, mask_type='half'):
        """
        Affine Coupling Layer (RealNVP)

        Args:
            dim: 입력 차원
            hidden_dim: 변환 함수 s, t의 hidden dimension
            mask_type: 'half' 또는 'checkerboard'
        """
        super().__init__()

        self.dim = dim

        # Mask 생성
        if mask_type == 'half':
            self.mask = torch.cat([
                torch.ones(dim // 2),
                torch.zeros(dim - dim // 2)
            ])
        elif mask_type == 'checkerboard':
            self.mask = torch.tensor([i % 2 for i in range(dim)], dtype=torch.float)

        # Scale 함수 s(z₁)
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()  # Scale을 제한 (-1, 1)
        )

        # Translation 함수 t(z₁)
        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, z, reverse=False):
        """
        Forward or inverse transform

        Args:
            z: (batch, dim)
            reverse: False (z→x), True (x→z)

        Returns:
            output: (batch, dim)
            log_det: (batch,) log|det(Jacobian)|
        """
        mask = self.mask.to(z.device)

        if not reverse:
            # Forward: z → x
            z1 = mask * z
            z2 = (1 - mask) * z

            # s(z₁), t(z₁) 계산
            s = self.scale_net(z1)
            t = self.translate_net(z1)

            # Masking 적용 (z₁ 부분은 0)
            s = (1 - mask) * s
            t = (1 - mask) * t

            # x₂ = z₂ ⊙ exp(s) + t
            x = mask * z + (1 - mask) * (z * torch.exp(s) + t)

            # Log-det-Jacobian
            log_det = s.sum(dim=-1)

            return x, log_det

        else:
            # Inverse: x → z
            x1 = mask * z
            x2 = (1 - mask) * z

            # s(x₁), t(x₁) 계산
            s = self.scale_net(x1)
            t = self.translate_net(x1)

            s = (1 - mask) * s
            t = (1 - mask) * t

            # z₂ = (x₂ - t) / exp(s)
            z = mask * z + (1 - mask) * ((z - t) * torch.exp(-s))

            # Log-det-Jacobian (역변환은 음수)
            log_det = -s.sum(dim=-1)

            return z, log_det
```

---

## 5. RealNVP 구현

### 5.1 RealNVP 아키텍처

**RealNVP (Real-valued Non-Volume Preserving)**는 여러 coupling layer를 쌓아 만든 flow 모델.

#### 전체 구조

```
z ~ N(0, I)
    ↓
Coupling Layer 1 (mask type A)
    ↓
Coupling Layer 2 (mask type B)  ← mask 교대로
    ↓
...
    ↓
Coupling Layer K
    ↓
x ~ p_data(x)
```

**핵심:**
- Mask를 교대로 사용 (모든 차원이 변환되도록)
- Multi-scale architecture (이미지용)

### 5.2 Multi-Scale Architecture

**문제:** 고해상도 이미지는 차원이 너무 높음

**해결:** Pyramid structure

```
Image (64×64×3)
    ↓ Coupling Layers
    ↓ Squeeze (shape 변환)
    ↓ Coupling Layers
    ↓ Split (일부 차원 분리)
    ↓ Coupling Layers
    ↓ Split
    ↓
Latent z (8×8×12)
```

#### Squeeze Operation

```
(H, W, C) → (H/2, W/2, 4C)

예: (64, 64, 3) → (32, 32, 12)
```

**구현:**

```python
def squeeze(x):
    """
    Squeeze operation for multi-scale architecture

    Args:
        x: (batch, C, H, W)

    Returns:
        y: (batch, 4C, H/2, W/2)
    """
    b, c, h, w = x.size()
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * 4, h // 2, w // 2)
    return x

def unsqueeze(x):
    """
    Inverse of squeeze

    Args:
        x: (batch, 4C, H, W)

    Returns:
        y: (batch, C, 2H, 2W)
    """
    b, c, h, w = x.size()
    x = x.view(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, c // 4, h * 2, w * 2)
    return x
```

### 5.3 완전한 RealNVP 모델

```python
class RealNVP(nn.Module):
    def __init__(self, input_dim, num_coupling_layers=8):
        """
        RealNVP Flow Model

        Args:
            input_dim: 입력 차원
            num_coupling_layers: Coupling layer 개수
        """
        super().__init__()

        self.input_dim = input_dim

        # Coupling layers (mask 교대로)
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(
                input_dim,
                hidden_dim=256,
                mask_type='half' if i % 2 == 0 else 'checkerboard'
            )
            for i in range(num_coupling_layers)
        ])

        # Base distribution
        self.register_buffer('base_mu', torch.zeros(input_dim))
        self.register_buffer('base_std', torch.ones(input_dim))

    def forward(self, x):
        """
        Compute log-likelihood

        Args:
            x: (batch, input_dim) 데이터

        Returns:
            log_prob: (batch,) log p(x)
        """
        log_det_sum = 0

        # x → z (inverse flow)
        z = x
        for layer in reversed(self.coupling_layers):
            z, log_det = layer(z, reverse=True)
            log_det_sum += log_det

        # Base distribution log-prob
        log_prob_z = -0.5 * (z ** 2).sum(dim=-1) - \
                     0.5 * self.input_dim * math.log(2 * math.pi)

        # Total log-prob
        log_prob_x = log_prob_z + log_det_sum

        return log_prob_x

    def sample(self, num_samples):
        """
        Generate samples

        Args:
            num_samples: 생성할 샘플 수

        Returns:
            x: (num_samples, input_dim)
        """
        # Sample from base distribution
        z = torch.randn(num_samples, self.input_dim).to(self.base_mu.device)

        # z → x (forward flow)
        x = z
        for layer in self.coupling_layers:
            x, _ = layer(x, reverse=False)

        return x
```

### 5.4 학습

```python
def train_realnvp(model, dataloader, num_epochs=100, lr=1e-4):
    """
    RealNVP 학습
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in dataloader:
            x = batch  # (batch, input_dim)

            # Forward: compute log p(x)
            log_prob = model(x)

            # Loss: negative log-likelihood
            loss = -log_prob.mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} - NLL: {avg_loss:.4f}')

        # 샘플 생성 (시각화)
        if (epoch + 1) % 10 == 0:
            samples = model.sample(64)
            # save_image(samples, ...)
```

---

## 6. Glow: Generative Flow

### 6.1 Glow의 개선점

**Glow**는 RealNVP의 개선 버전:

1. **Actnorm**: Batch Normalization 대신
2. **1×1 Invertible Convolution**: Permutation 대신
3. **Affine Coupling**: RealNVP와 동일

### 6.2 Actnorm (Activation Normalization)

#### 문제

Batch Normalization:
- Batch size에 의존
- Inference 시 running statistics 필요

#### Actnorm

$$
y = s \odot x + b
$$

**파라미터:**
- s, b: 학습 가능 (per-channel)

**초기화 (data-dependent):**
```python
# 첫 배치로 s, b 초기화
mean = x.mean(dim=[0, 2, 3])  # (C,)
std = x.std(dim=[0, 2, 3])

b = -mean
s = 1 / (std + 1e-6)
```

**Log-det-Jacobian:**
$$
\log |\det J| = H \times W \times \sum_c \log |s_c|
$$

#### 구현

```python
class Actnorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.num_channels = num_channels
        self.initialized = False

        # Learnable parameters
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x, reverse=False):
        """
        Args:
            x: (batch, C, H, W)
        """
        if not self.initialized:
            self.initialize(x)
            self.initialized = True

        b, c, h, w = x.size()

        if not reverse:
            # Forward
            y = self.log_scale.exp() * x + self.bias
            log_det = h * w * self.log_scale.sum()
            log_det = log_det.expand(b)
            return y, log_det
        else:
            # Inverse
            y = (x - self.bias) / self.log_scale.exp()
            log_det = -h * w * self.log_scale.sum()
            log_det = log_det.expand(b)
            return y, log_det

    def initialize(self, x):
        """Data-dependent initialization"""
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)

            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(torch.log(1 / (std + 1e-6)))
```

### 6.3 1×1 Invertible Convolution

#### 아이디어

Permutation 대신 **학습 가능한 1×1 convolution** 사용.

**수식:**
$$
y = Wx
$$

여기서 W ∈ ℝ^{C×C}는 **invertible matrix**.

**Log-det-Jacobian:**
$$
\log |\det J| = H \times W \times \log |\det W|
$$

#### LU Decomposition

**문제:** det(W) 계산은 O(C³)

**해결:** W를 LU 분해
$$
W = PLU
$$

- P: Permutation (고정)
- L: Lower triangular (학습)
- U: Upper triangular (학습)

**det(W) = det(P)det(L)det(U)**

Triangular이므로:
$$
\det(W) = \prod_i L_{ii} \times \prod_i U_{ii}
$$

#### 구현

```python
class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.num_channels = num_channels

        # Initialize with random rotation
        W = torch.qr(torch.randn(num_channels, num_channels))[0]

        # LU decomposition
        P, L, U = torch.lu_unpack(*torch.lu(W))

        # Register as buffer (not trainable)
        self.register_buffer('P', P)

        # Lower triangular (set diagonal to 1)
        L = L - torch.diag(torch.diag(L)) + torch.eye(num_channels)
        self.L = nn.Parameter(L)

        # Upper triangular
        self.U = nn.Parameter(U)

        # Log of diagonal of U
        self.log_s = nn.Parameter(torch.log(torch.abs(torch.diag(U))))

    def forward(self, x, reverse=False):
        """
        Args:
            x: (batch, C, H, W)
        """
        b, c, h, w = x.size()

        # Reconstruct W
        L = torch.tril(self.L, diagonal=-1) + torch.eye(c).to(self.L.device)
        U = torch.triu(self.U, diagonal=1)
        U = U + torch.diag(self.log_s.exp())

        W = self.P @ L @ U  # (C, C)

        if not reverse:
            # Forward: y = Wx
            x_flat = x.view(b, c, h * w)  # (b, c, h*w)
            y_flat = torch.matmul(W.unsqueeze(0), x_flat)  # (b, c, h*w)
            y = y_flat.view(b, c, h, w)

            # Log-det
            log_det = h * w * self.log_s.sum()
            log_det = log_det.expand(b)

            return y, log_det
        else:
            # Inverse: y = W⁻¹x
            W_inv = torch.inverse(W)

            x_flat = x.view(b, c, h * w)
            y_flat = torch.matmul(W_inv.unsqueeze(0), x_flat)
            y = y_flat.view(b, c, h, w)

            log_det = -h * w * self.log_s.sum()
            log_det = log_det.expand(b)

            return y, log_det
```

### 6.4 Glow Step

**하나의 Glow Step** = Actnorm + 1×1 Conv + Affine Coupling

```python
class GlowStep(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.actnorm = Actnorm(num_channels)
        self.inv_conv = InvertibleConv1x1(num_channels)
        self.coupling = AffineCouplingLayer(num_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            x, log_det1 = self.actnorm(x)
            x, log_det2 = self.inv_conv(x)
            x, log_det3 = self.coupling(x)

            log_det = log_det1 + log_det2 + log_det3

            return x, log_det
        else:
            x, log_det3 = self.coupling(x, reverse=True)
            x, log_det2 = self.inv_conv(x, reverse=True)
            x, log_det1 = self.actnorm(x, reverse=True)

            log_det = log_det1 + log_det2 + log_det3

            return x, log_det
```

---

## 7. Continuous Normalizing Flows (Neural ODE)

### 7.1 From Discrete to Continuous

**일반 Flow:** 이산적인 변환들의 합성
$$
\mathbf{z}_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z}_0)
$$

**Continuous Flow:** 연속적인 변화
$$
\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t; \theta)
$$

### 7.2 Change of Variables for ODEs

**Instantaneous Change of Variables:**

$$
\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\text{tr}\left(\frac{\partial f}{\partial \mathbf{z}(t)}\right)
$$

**적분하면:**

$$
\log p(\mathbf{z}(T)) = \log p(\mathbf{z}(0)) - \int_0^T \text{tr}\left(\frac{\partial f}{\partial \mathbf{z}(t)}\right) dt
$$

### 7.3 Neural ODE 구현

```python
from torchdiffeq import odeint

class CNF(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()

        # Neural network for f(z, t)
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, t, z):
        """
        ODE function: dz/dt = f(z, t)

        Args:
            t: scalar time
            z: (batch, dim)

        Returns:
            dz_dt: (batch, dim)
        """
        t_vec = t * torch.ones(z.size(0), 1).to(z.device)
        zt = torch.cat([z, t_vec], dim=1)
        dz_dt = self.net(zt)
        return dz_dt

# Usage
cnf = CNF(dim=2)

# Forward: z(0) → z(T)
t = torch.tensor([0.0, 1.0])
z_0 = torch.randn(100, 2)
z_T = odeint(cnf, z_0, t)[1]  # [1] for final time

# Inverse: z(T) → z(0)
t_reverse = torch.tensor([1.0, 0.0])
z_0_recon = odeint(cnf, z_T, t_reverse)[1]
```

**장점:**
- 메모리 효율적 (adjoint method)
- 유연한 아키텍처

**단점:**
- 느린 샘플링 (ODE solver)

---

## 8. Flow Matching

### 8.1 개요

**Flow Matching**은 최근 제안된 방법으로, diffusion과 flow를 결합:

**핵심 아이디어:**
- Continuous flow 학습
- Diffusion처럼 간단한 학습 목적 함수
- ODE solver로 빠른 샘플링

### 8.2 Optimal Transport

**문제:** p₀ → p₁로 가는 최적 경로는?

**Conditional Flow:**
$$
\phi_t(x) = (1 - t)x_0 + tx_1 + \sigma(t) \epsilon
$$

**Flow Matching Loss:**
$$
\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1}\left[\left\|v_\theta(x_t, t) - \frac{dx_t}{dt}\right\|^2\right]
$$

### 8.3 Rectified Flow

**더 간단한 버전:** 직선 경로

$$
x_t = (1 - t)x_0 + tx_1
$$

**Velocity:**
$$
v_t = x_1 - x_0
$$

**Loss:**
$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_1}\left[\left\|v_\theta(x_t, t) - (x_1 - x_0)\right\|^2\right]
$$

---

## 9. 실습 프로젝트

### 프로젝트 1: 2D Toy Data

**목표:** RealNVP로 2D 분포 학습

```python
# 1. 데이터 생성 (two moons)
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=10000, noise=0.05)

# 2. RealNVP 학습
model = RealNVP(input_dim=2, num_coupling_layers=8)
train_realnvp(model, X, num_epochs=1000)

# 3. 샘플 생성 및 시각화
samples = model.sample(1000).detach().numpy()
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
```

### 프로젝트 2: MNIST 생성

```bash
# RealNVP로 MNIST 학습
python train_flow.py --model realnvp --dataset mnist --epochs 100

# 샘플 생성
python generate.py --checkpoint realnvp_mnist.pth --num-samples 100
```

### 프로젝트 3: Latent Space Interpolation

```python
# Encode two images
z1 = model.encode(img1)
z2 = model.encode(img2)

# Interpolate in latent space
for alpha in np.linspace(0, 1, 10):
    z = (1 - alpha) * z1 + alpha * z2
    img = model.decode(z)
    save_image(img, f'interp_{alpha:.1f}.png')
```

---

## 📚 참고 자료

### 필수 논문

1. **NICE** (Dinh et al., 2014)
   - [arXiv](https://arxiv.org/abs/1410.8516)
   - Additive coupling layers

2. **RealNVP** (Dinh et al., 2016)
   - [arXiv](https://arxiv.org/abs/1605.08803)
   - Affine coupling + multi-scale

3. **Glow** (Kingma & Dhariwal, 2018)
   - [arXiv](https://arxiv.org/abs/1807.03039)
   - 1×1 conv + actnorm

4. **Neural ODE** (Chen et al., 2018)
   - [arXiv](https://arxiv.org/abs/1806.07366)
   - Continuous normalizing flows

5. **Flow Matching** (Lipman et al., 2022)
   - [arXiv](https://arxiv.org/abs/2210.02747)
   - Optimal transport flows

---

## ✅ 학습 체크리스트

- [ ] Change of Variables Theorem 완전 이해
- [ ] Jacobian determinant 계산 방법 숙지
- [ ] Coupling layer 작동 원리 이해
- [ ] RealNVP 아키텍처 구현
- [ ] Actnorm, 1×1 Conv 구현
- [ ] Glow 모델 이해
- [ ] Neural ODE 개념 파악
- [ ] Flow Matching 최신 연구 학습
- [ ] 2D toy data로 실습 완료

---

**최종 업데이트**: 2025-11-15
**작성자**: Claude
**라이선스**: MIT
