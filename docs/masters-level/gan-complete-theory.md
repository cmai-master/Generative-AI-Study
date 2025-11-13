# GAN (Generative Adversarial Networks) - 대학원 수준 완전 분석

> **대상**: 석사 과정 학생, 연구자, 고급 학습자
> **선수 지식**: 확률론, 최적화 이론, 게임 이론, 딥러닝 기초
> **목표**: GAN의 이론적 기반부터 최신 연구까지 완벽히 이해

---

## 목차

1. [게임 이론의 수학적 기초](#1-게임-이론의-수학적-기초)
2. [GAN의 수학적 정의](#2-gan의-수학적-정의)
3. [학습 알고리즘과 수렴성 분석](#3-학습-알고리즘과-수렴성-분석)
4. [Mode Collapse와 학습 불안정성](#4-mode-collapse와-학습-불안정성)
5. [Wasserstein GAN의 이론](#5-wasserstein-gan의-이론)
6. [GAN 변형 모델들](#6-gan-변형-모델들)
7. [이론적 분석과 한계](#7-이론적-분석과-한계)
8. [최신 연구 동향](#8-최신-연구-동향)

---

## 1. 게임 이론의 수학적 기초

### 1.1 비협조 게임 이론 (Non-cooperative Game Theory)

#### 1.1.1 기본 개념

**정의 1.1 (Two-player Game)**
두 플레이어 게임은 다음으로 구성됩니다:
```
G = (S₁, S₂, u₁, u₂)

여기서:
- S₁, S₂: 각 플레이어의 전략 공간
- u₁, u₂: 각 플레이어의 효용 함수
- uᵢ: S₁ × S₂ → ℝ
```

**GAN의 경우**:
```
플레이어 1 (Generator):
  - 전략 공간: 생성기 파라미터 θ_G
  - 목표: 판별기를 속이기

플레이어 2 (Discriminator):
  - 전략 공간: 판별기 파라미터 θ_D
  - 목표: 진짜와 가짜 구별하기
```

#### 1.1.2 Nash 균형

**정의 1.2 (Nash Equilibrium)**
전략 프로필 (s₁*, s₂*)이 Nash 균형이면:
```
u₁(s₁*, s₂*) ≥ u₁(s₁, s₂*)  ∀s₁ ∈ S₁
u₂(s₁*, s₂*) ≥ u₂(s₁*, s₂)  ∀s₂ ∈ S₂
```

**의미**: 어느 플레이어도 상대방의 전략이 고정되었을 때 일방적으로 전략을 바꿔서 이득을 얻을 수 없음

**정리 1.1 (Nash 균형의 존재성)**
유한 전략 공간을 가진 게임에서 혼합 전략 Nash 균형이 항상 존재합니다 (Nash, 1950).

**증명 스케치**:
Kakutani 부동점 정리를 사용. 최적 응답 대응(best response correspondence)이 연속이고 컨벡스하므로 부동점이 존재.

#### 1.1.3 Minimax 정리

**정의 1.3 (Zero-sum Game)**
제로섬 게임에서: u₁(s₁, s₂) + u₂(s₁, s₂) = 0

**정리 1.2 (Von Neumann Minimax Theorem)**
컴팩트 볼록 전략 공간과 연속 효용 함수를 가진 제로섬 게임에서:
```
max_{s₁} min_{s₂} u₁(s₁, s₂) = min_{s₂} max_{s₁} u₁(s₁, s₂)
```

**증명 핵심**:
1. 강한 duality (Sion's minimax theorem)
2. 최적 혼합 전략의 존재

**GAN과의 연결**:
GAN은 엄밀히 제로섬이 아니지만, minimax 프레임워크를 사용합니다.

### 1.2 Divergence와 거리 척도

#### 1.2.1 f-divergence

**정의 1.4 (f-divergence)**
볼록 함수 f: ℝ₊ → ℝ (f(1) = 0)에 대해:
```
D_f(P||Q) = ∫ q(x) f(p(x)/q(x)) dx
```

**예시**:

1. **KL Divergence**: f(t) = t log t
```
D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
```

2. **Reverse KL**: f(t) = -log t
```
D_KL(Q||P) = ∫ q(x) log(q(x)/p(x)) dx
```

3. **JS Divergence**:
```
D_JS(P||Q) = ½D_KL(P||M) + ½D_KL(Q||M)
여기서 M = ½(P + Q)
```

4. **Total Variation Distance**:
```
TV(P, Q) = ½∫ |p(x) - q(x)| dx
```

**정리 1.3 (f-divergence의 성질)**
1. **비음수성**: D_f(P||Q) ≥ 0, 등호는 P = Q일 때
2. **볼록성**: D_f는 (P, Q)에 대해 jointly convex
3. **데이터 처리 부등식**: 변환 T에 대해 D_f(T(P)||T(Q)) ≤ D_f(P||Q)

#### 1.2.2 Integral Probability Metrics (IPM)

**정의 1.5 (IPM)**
함수 클래스 ℱ에 대해:
```
d_ℱ(P, Q) = sup_{f∈ℱ} |E_P[f(x)] - E_Q[f(x)]|
```

**예시**:

1. **Total Variation**: ℱ = {f: ||f||_∞ ≤ 1}
2. **Wasserstein-1**: ℱ = {f: ||f||_L ≤ 1} (Lipschitz)
3. **Maximum Mean Discrepancy (MMD)**: ℱ는 재생 커널 힐베르트 공간

**정리 1.4 (IPM과 f-divergence의 차이)**
- f-divergence: 절대 연속성 요구, 지지 집합 불일치 시 무한대
- IPM: 약한 토폴로지, 지지 집합 불일치에 강건

---

## 2. GAN의 수학적 정의

### 2.1 원본 GAN 정식화

#### 2.1.1 목적 함수

**정의 2.1 (GAN Objective)**
```
min_G max_D V(G, D) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

여기서:
- G: 생성기 (Generator)
- D: 판별기 (Discriminator)
- p_data: 실제 데이터 분포
- p_z: 잠재 변수 분포 (보통 N(0, I))
- p_g: 생성 분포 G(z), z ~ p_z

**해석**:
- **Discriminator 목표**: V를 최대화
  - D(x) → 1 (진짜를 진짜로 판별)
  - D(G(z)) → 0 (가짜를 가짜로 판별)

- **Generator 목표**: V를 최소화
  - D(G(z)) → 1 (가짜를 진짜처럼 속이기)

#### 2.1.2 최적 판별기

**정리 2.1 (Optimal Discriminator)**
고정된 G에 대해 최적 판별기는:
```
D*_G(x) = p_data(x) / (p_data(x) + p_g(x))
```

**증명**:
```
목표: max_D V(G, D)

V(G, D) = ∫ p_data(x) log D(x) dx + ∫ p_g(x) log(1 - D(x)) dx
        = ∫ [p_data(x) log D(x) + p_g(x) log(1 - D(x))] dx

각 x에 대해 독립적으로 최적화:
∂/∂D(x) [p_data(x) log D(x) + p_g(x) log(1 - D(x))]
= p_data(x)/D(x) - p_g(x)/(1 - D(x)) = 0

⇒ p_data(x)(1 - D(x)) = p_g(x)D(x)
⇒ p_data(x) = [p_data(x) + p_g(x)]D(x)
⇒ D*(x) = p_data(x)/(p_data(x) + p_g(x))
```

**의미**:
- D*(x) = 0.5 ⟺ p_data(x) = p_g(x) (완벽한 생성)
- D*(x) → 1 ⟺ p_g(x) → 0 (생성기가 x를 생성 못함)
- D*(x) → 0 ⟺ p_data(x) → 0 (실제 데이터 없음)

#### 2.1.3 전역 최적성

**정리 2.2 (Global Optimality)**
최적 판별기 D*_G를 대입하면:
```
C(G) = max_D V(G, D) = -log 4 + 2·D_JS(p_data || p_g)
```

여기서 D_JS는 Jensen-Shannon divergence.

**증명**:
```
C(G) = E_{x~p_data}[log D*_G(x)] + E_{x~p_g}[log(1 - D*_G(x))]

     = E_{x~p_data}[log p_data(x)/(p_data(x) + p_g(x))]
       + E_{x~p_g}[log p_g(x)/(p_data(x) + p_g(x))]

     = E_{x~p_data}[log p_data(x)/M(x)] + E_{x~p_g}[log p_g(x)/M(x)]
       - 2log 2

여기서 M = (p_data + p_g)/2

     = D_KL(p_data||M) + D_KL(p_g||M) - 2log 2
     = 2·D_JS(p_data||p_g) - log 4
```

**따름정리 2.1 (Global Minimum)**
```
C* = min_G C(G) = -log 4
```

이는 p_g = p_data일 때 달성됩니다.

**증명**:
D_JS(p_data||p_g) ≥ 0이고 등호는 p_data = p_g일 때 성립. 따라서 C(G) ≥ -log 4이고 최소값은 -log 4.

### 2.2 대안적 목적 함수

#### 2.2.1 Non-saturating GAN

**문제**: 원본 목적 함수에서 G의 기울기가 D(G(z)) ≈ 0일 때 소실 (vanishing gradient)

**해결**: Generator 목적 함수 변경
```
원본:    min_G E_z[log(1 - D(G(z)))]
대안:    max_G E_z[log D(G(z))]
```

**분석**:
```
원본 기울기: ∇_G E_z[log(1 - D(G(z)))]
           = E_z[(1-D(G(z)))^(-1) · (-∇_G D(G(z)))]
           ≈ 0 when D(G(z)) ≈ 0

대안 기울기: ∇_G E_z[log D(G(z))]
           = E_z[D(G(z))^(-1) · ∇_G D(G(z))]
           → ∞ when D(G(z)) → 0 (강한 신호!)
```

**트레이드오프**:
- 장점: 학습 초기 강한 기울기
- 단점: 더 이상 JS divergence 최소화 아님

#### 2.2.2 Least Squares GAN (LSGAN)

**목적 함수**:
```
min_D V_LS(D) = ½E_{x~p_data}[(D(x) - 1)²] + ½E_{z~p_z}[(D(G(z)))²]
min_G E_{z~p_z}[(D(G(z)) - 1)²]
```

**장점**:
1. Vanishing gradient 문제 완화
2. 더 안정적인 학습
3. Pearson χ² divergence 최소화

**정리 2.3 (LSGAN의 최적해)**
최적 판별기:
```
D*(x) = (p_data(x) - p_g(x))/(p_data(x) + p_g(x))
```

최소화되는 divergence:
```
χ²_Pearson(p_data||p_g) = ∫ (p_data(x) - p_g(x))²/p_g(x) dx
```

### 2.3 조건부 GAN (Conditional GAN)

#### 2.3.1 정식화

**목적 함수**:
```
min_G max_D V(G,D) = E_{x,y~p_data}[log D(x|y)]
                    + E_{z~p_z, y~p_y}[log(1 - D(G(z|y)|y))]
```

여기서 y는 조건 (레이블, 클래스, 텍스트 등)

**응용**:
- Image-to-image translation (pix2pix)
- Text-to-image synthesis
- Class-conditional generation

#### 2.3.2 정보 이론적 해석

**상호 정보량 최대화**:
```
I(x; y) = H(y) - H(y|x)
```

Conditional GAN은 암묵적으로 I(G(z); y)를 최대화합니다.

---

## 3. 학습 알고리즘과 수렴성 분석

### 3.1 동시 경사 하강법 (Simultaneous Gradient Descent)

#### 3.1.1 알고리즘

**Algorithm 3.1 (GAN Training)**
```
for iteration = 1 to max_iterations:
    # 1. Discriminator 학습
    for k steps:
        Sample minibatch {x₁, ..., x_m} from p_data
        Sample minibatch {z₁, ..., z_m} from p_z

        Update D by ascending its gradient:
        ∇_θ_D [1/m Σᵢ log D(xᵢ) + log(1 - D(G(zᵢ)))]

    # 2. Generator 학습
    Sample minibatch {z₁, ..., z_m} from p_z

    Update G by descending its gradient:
    ∇_θ_G [1/m Σᵢ log(1 - D(G(zᵢ)))]
    # 또는 non-saturating: -∇_θ_G [1/m Σᵢ log D(G(zᵢ))]
```

**하이퍼파라미터**:
- k: Discriminator steps per generator step (보통 1 또는 5)
- Learning rate: 보통 α_D = α_G = 10^(-4)
- Optimizer: Adam (β₁=0.5, β₂=0.999)

#### 3.1.2 수렴성 문제

**문제 3.1 (비수렴성)**
동시 경사 하강법이 Nash 균형으로 수렴한다는 보장이 없습니다.

**예시 3.1 (Simple Non-convergence)**
```
간단한 bilinear 게임:
V(x, y) = xy

∇_x V = y, ∇_y V = x

Gradient dynamics:
dx/dt = y
dy/dt = -x

이는 회전하며 수렴하지 않음!
```

**시각화**:
```python
import numpy as np
import matplotlib.pyplot as plt

# 경사 하강 동역학
dt = 0.1
x, y = 1.0, 0.0
trajectory = [(x, y)]

for _ in range(100):
    x += dt * y      # ∇_x V = y
    y += dt * (-x)   # ∇_y V = -x
    trajectory.append((x, y))

# 원을 그리며 회전 (수렴 안함!)
```

### 3.2 수렴성 이론

#### 3.2.1 국소 수렴성

**정리 3.1 (Local Convergence near Equilibrium)**
Nash 균형 (θ*_G, θ*_D) 근처에서, 적절한 학습률과 조건 하에 국소적으로 수렴할 수 있습니다.

**조건**:
1. Hessian의 eigenvalue가 실수부가 음수
2. 학습률이 충분히 작음
3. 목적 함수가 충분히 부드러움

**증명 스케치**:
선형화 후 Lyapunov 안정성 분석. Jacobian의 eigenvalue 조건 확인.

#### 3.2.2 Unrolled GAN

**아이디어**: Generator 업데이트 시 Discriminator가 k스텝 앞을 "내다봄"

**Algorithm 3.2 (Unrolled GAN)**
```python
def generator_objective(G, D, k_unroll):
    D_unrolled = copy(D)

    # D를 k스텝 업데이트 (unroll)
    for _ in range(k_unroll):
        # D_unrolled를 현재 G에 대해 최적화
        D_unrolled = update_discriminator(D_unrolled, G)

    # Unrolled D를 사용해 G 업데이트
    loss = -E[log D_unrolled(G(z))]
    return loss
```

**장점**:
- Mode collapse 완화
- 더 안정적인 학습

**단점**:
- 계산 비용 증가 (backprop through k steps)
- 메모리 요구량 증가

### 3.3 실전 학습 기법

#### 3.3.1 One-sided Label Smoothing

**기법**: 실제 데이터 레이블을 1 대신 0.9 사용
```python
# ❌ 하드 레이블
real_labels = torch.ones(batch_size, 1)
fake_labels = torch.zeros(batch_size, 1)

# ✅ 레이블 스무딩
real_labels = torch.ones(batch_size, 1) * 0.9  # 0.9 instead of 1
fake_labels = torch.zeros(batch_size, 1)       # 0은 그대로
```

**이유**:
- Discriminator가 과도하게 확신하는 것 방지
- Gradient 소실 완화

**주의**: 가짜 레이블은 스무딩하지 않음 (양방향 스무딩은 오히려 해로움)

#### 3.3.2 Historical Averaging

**목적 함수에 추가**:
```
V_avg(θ) = V(θ) + λ/2 ||θ - 1/t Σᵢ₌₁ᵗ θᵢ||²
```

**효과**: 과거 파라미터의 평균 근처에 머물도록 정규화

#### 3.3.3 Experience Replay

**아이디어**: 이전에 생성된 가짜 샘플 재사용

```python
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.buffer = []
        self.max_size = max_size

    def push_and_pop(self, data):
        to_return = []
        for element in data:
            if len(self.buffer) < self.max_size:
                self.buffer.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.buffer[i].clone())
                    self.buffer[i] = element
                else:
                    to_return.append(element)
        return torch.stack(to_return)
```

**효과**: Discriminator가 과거 Generator에 대해서도 강건

---

## 4. Mode Collapse와 학습 불안정성

### 4.1 Mode Collapse 현상

#### 4.1.1 정의와 유형

**정의 4.1 (Mode Collapse)**
생성기가 데이터 분포의 일부 모드만 생성하고 다른 모드를 무시하는 현상

**유형**:

1. **Complete Mode Collapse**: 하나의 샘플만 생성
2. **Partial Mode Collapse**: 일부 모드만 커버
3. **Mode Hopping**: 학습 중 다른 모드로 점프

**예시**:
```python
# MNIST에서 mode collapse
# Generator가 숫자 1만 생성하고 0, 2, ..., 9는 생성 안 함
```

#### 4.1.2 이론적 분석

**원인 1: KL vs Reverse KL**

**Forward KL** (VAE 사용):
```
D_KL(p_data || p_g) = E_{x~p_data}[log p_data(x)/p_g(x)]
```
- p_data > 0, p_g = 0이면 무한대
- 결과: p_g는 p_data의 모든 모드를 커버 (mode-covering)

**Reverse KL** (GAN 암묵적 사용):
```
D_KL(p_g || p_data) = E_{x~p_g}[log p_g(x)/p_data(x)]
```
- p_g > 0, p_data = 0이면 무한대
- 결과: p_g는 p_data의 높은 확률 영역만 선택 (mode-seeking)

**정리 4.1 (Mode-Seeking Behavior)**
GAN의 Generator는 암묵적으로 reverse KL을 최소화하려 하므로 mode-seeking 행동을 보입니다.

**증명 스케치**:
최적 D*를 사용하면 Generator는 JS divergence를 최소화. JS divergence는 두 KL의 조합이지만, 학습 동역학상 reverse KL 방향이 더 강함.

#### 4.1.3 수학적 모델

**간단한 예시**:
```
p_data = 0.5·N(-1, 0.1) + 0.5·N(1, 0.1)  (두 개의 가우시안 혼합)
p_g = N(μ, σ²)  (단일 가우시안)

목표: min_μ,σ D_JS(p_data || p_g)
```

**결과**:
- 이상적: μ = 0, σ² 커짐 (두 모드 모두 커버)
- 실전: μ → -1 또는 μ → 1 (한 모드만 선택)

### 4.2 Mode Collapse 감지

#### 4.2.1 평가 지표

**1. Number of Modes Captured**
```python
def count_modes_captured(generated_samples, real_modes, threshold=0.1):
    """
    생성된 샘플이 실제 모드를 얼마나 커버하는지 계산
    """
    modes_captured = 0
    for mode_center in real_modes:
        # 각 모드 근처에 생성 샘플이 있는지 확인
        distances = np.linalg.norm(generated_samples - mode_center, axis=1)
        if np.min(distances) < threshold:
            modes_captured += 1
    return modes_captured / len(real_modes)
```

**2. Reverse KL Divergence Approximation**
```python
def estimate_reverse_kl(p_g_samples, p_data_density):
    """
    Monte Carlo로 D_KL(p_g || p_data) 근사
    """
    log_p_g = compute_log_density_gan(p_g_samples)  # 어려움!
    log_p_data = p_data_density(p_g_samples)

    reverse_kl = np.mean(log_p_g - log_p_data)
    return reverse_kl
```

**3. Birthday Paradox Test**
동일한 샘플이 반복 생성되는지 확인:
```python
def birthday_paradox_score(generated_samples, threshold=0.01):
    """
    중복 샘플 비율 계산
    """
    n = len(generated_samples)
    duplicates = 0

    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(generated_samples[i] - generated_samples[j]) < threshold:
                duplicates += 1

    return duplicates / (n * (n-1) / 2)
```

### 4.3 Mode Collapse 완화 기법

#### 4.3.1 Minibatch Discrimination

**아이디어**: Discriminator가 개별 샘플뿐 아니라 미니배치의 다양성도 고려

**구현**:
```python
class MinibatchDiscrimination(nn.Module):
    """
    Salimans et al., 2016: Improved Techniques for Training GANs
    """
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean

        # T tensor: [in_features, out_features, kernel_dims]
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        # x: [batch_size, in_features]

        # M = x @ T: [batch_size, out_features, kernel_dims]
        M = x @ self.T

        # 각 샘플 간의 L1 거리 계산
        # diffs: [batch_size, batch_size, out_features, kernel_dims]
        diffs = M.unsqueeze(0) - M.unsqueeze(1)
        abs_diffs = torch.abs(diffs).sum(3)  # [batch_size, batch_size, out_features]

        # 각 샘플의 다른 샘플들과의 거리 합
        if self.mean:
            minibatch_features = abs_diffs.mean(1)  # [batch_size, out_features]
        else:
            minibatch_features = torch.exp(-abs_diffs).sum(1)

        return torch.cat([x, minibatch_features], 1)
```

**효과**:
- Discriminator가 미니배치 내 다양성을 판별 기준으로 사용
- Generator는 다양한 샘플을 생성하도록 압력받음

#### 4.3.2 Unrolled GAN (재방문)

**핵심 아이디어**: Generator 업데이트 시 Discriminator의 미래 반응 예측

**수학적 분석**:
```
원본 Generator 목표:
min_G L_G(G, D)

Unrolled Generator 목표:
min_G L_G(G, D^k_G)

여기서 D^k_G는 현재 G에 대해 k스텝 최적화한 Discriminator
```

**효과**:
- Generator가 Discriminator의 적응을 고려
- Mode hopping 감소

#### 4.3.3 Feature Matching

**목표**: Generator가 실제 데이터의 통계를 매칭하도록

**손실 함수**:
```python
def feature_matching_loss(real_images, fake_images, discriminator):
    """
    ||E_{x~p_data}[f(x)] - E_{x~p_g}[f(x)]||²

    여기서 f(x)는 discriminator의 중간 특징
    """
    # Discriminator의 중간 층 특징 추출
    real_features = discriminator.get_features(real_images)
    fake_features = discriminator.get_features(fake_images)

    # 특징의 평균 매칭
    loss = F.mse_loss(fake_features.mean(0), real_features.mean(0))
    return loss
```

**장점**:
- Discriminator의 완벽한 속임보다 통계적 일치 추구
- 더 안정적인 학습

---

## 5. Wasserstein GAN의 이론

### 5.1 Wasserstein Distance

#### 5.1.1 정의

**정의 5.1 (Wasserstein-1 Distance)**
Earth Mover's Distance (EMD)라고도 함:
```
W(p, q) = inf_{γ∈Π(p,q)} E_{(x,y)~γ}[||x - y||]
```

여기서:
- Π(p,q): p와 q를 주변 분포로 하는 모든 결합 분포
- γ: "운송 계획" (transport plan)
- ||x - y||: 운송 비용

**직관**:
p를 q로 변환하는데 필요한 최소 "작업량"

**예시 5.1**:
```
p = δ_0 (0에 모든 질량)
q = δ_1 (1에 모든 질량)

W(p, q) = 1  (단위 질량을 거리 1만큼 이동)
```

#### 5.1.2 Kantorovich-Rubinstein Duality

**정리 5.1 (Kantorovich-Rubinstein Duality)**
```
W(p, q) = sup_{||f||_L≤1} [E_{x~p}[f(x)] - E_{x~q}[f(x)]]
```

여기서 ||f||_L ≤ 1은 1-Lipschitz 조건:
```
|f(x) - f(y)| ≤ ||x - y||  ∀x,y
```

**증명 스케치**:
선형 프로그래밍의 강한 쌍대성. 원문제(primal)는 운송 계획 최적화, 쌍대 문제(dual)는 Lipschitz 함수 최적화.

**GAN과의 연결**:
이것이 WGAN의 핵심! Discriminator를 1-Lipschitz 함수로 제약하면 Wasserstein 거리를 근사할 수 있습니다.

#### 5.1.3 JS Divergence vs Wasserstein Distance

**문제 5.1 (Disjoint Support)**
```
p_data와 p_g의 지지 집합이 겹치지 않으면:
- D_JS(p_data || p_g) = log 2 (상수!)
- D_KL(p_data || p_g) = ∞

하지만:
- W(p_data, p_g)는 유의미한 값 (거리에 비례)
```

**예시**:
```
p_data = U[0, 1] (0과 1 사이 균등 분포)
p_g = U[θ, θ+1] (θ와 θ+1 사이 균등 분포)

θ ≠ 0이면:
- D_JS = log 2 (θ 값에 무관!)
- W = |θ| (θ에 선형적으로 의존)
```

**의미**:
Wasserstein distance는 지지 집합이 겹치지 않아도 의미 있는 기울기 제공!

### 5.2 WGAN 알고리즘

#### 5.2.1 목적 함수

**WGAN Objective**:
```
min_G max_{D:||D||_L≤1} E_{x~p_data}[D(x)] - E_{z~p_z}[D(G(z))]
```

**원본 GAN과 비교**:
```
Original GAN: min_G max_D V(D,G) with sigmoid(D(x)) ∈ [0,1]
WGAN:         min_G max_D W(D,G) with D(x) ∈ ℝ (unbounded)
```

#### 5.2.2 Weight Clipping

**WGAN (Arjovsky et al., 2017) 원본 방법**:

**Algorithm 5.1 (WGAN with Weight Clipping)**
```python
for iteration in range(max_iterations):
    # Discriminator (Critic) 학습
    for _ in range(n_critic):  # 보통 n_critic=5
        # 실제 데이터
        real_data = sample_real_data()
        # 가짜 데이터
        z = sample_noise()
        fake_data = generator(z)

        # Wasserstein loss
        d_loss = -torch.mean(critic(real_data)) + torch.mean(critic(fake_data))

        # Backward and optimize
        critic_optimizer.zero_grad()
        d_loss.backward()
        critic_optimizer.step()

        # WEIGHT CLIPPING
        for p in critic.parameters():
            p.data.clamp_(-clip_value, clip_value)  # clip_value = 0.01

    # Generator 학습
    z = sample_noise()
    fake_data = generator(z)
    g_loss = -torch.mean(critic(fake_data))

    generator_optimizer.zero_grad()
    g_loss.backward()
    generator_optimizer.step()
```

**Weight Clipping 분석**:

**목적**: Lipschitz 제약 강제
```
||f||_L ≤ K ⟹ |f(x) - f(y)| ≤ K||x - y||
```

**문제점**:
1. **제약이 너무 엄격**: 함수 표현력 제한
2. **Capacity underuse**: 가중치가 ±clip_value로 몰림
3. **Gradient vanishing/exploding**: 부적절한 clip_value 선택 시

### 5.3 WGAN-GP (Gradient Penalty)

#### 5.3.1 개선된 제약 방법

**문제**: Weight clipping은 부작용이 많음

**해결**: Gradient penalty (Gulrajani et al., 2017)

**핵심 아이디어**:
1-Lipschitz ⟺ ||∇_x f(x)|| ≤ 1 everywhere

**목적 함수**:
```
L = E_{x~p_data}[D(x)] - E_{x~p_g}[D(x)]
    + λ·E_{x̂~p_{x̂}}[(||∇_{x̂} D(x̂)||₂ - 1)²]
```

여기서:
- λ: 페널티 가중치 (보통 10)
- p_{x̂}: 실제와 가짜 데이터 사이의 보간
- x̂ = εx + (1-ε)x̃, ε ~ U[0,1]

#### 5.3.2 구현

```python
def compute_gradient_penalty(critic, real_data, fake_data, device='cuda'):
    """
    WGAN-GP gradient penalty

    Args:
        critic: Discriminator network
        real_data: [batch_size, ...] real samples
        fake_data: [batch_size, ...] generated samples

    Returns:
        gradient_penalty: scalar
    """
    batch_size = real_data.size(0)

    # Random weight term for interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)

    # Interpolated samples
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)

    # Critic scores for interpolated samples
    d_interpolates = critic(interpolates)

    # Gradients w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten gradients
    gradients = gradients.view(batch_size, -1)

    # Gradient penalty: (||∇_x̂ D(x̂)||₂ - 1)²
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
```

**전체 학습 루프**:
```python
for iteration in range(max_iterations):
    # Critic 학습
    for _ in range(n_critic):
        real_data = next(data_loader)
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)

        # Wasserstein loss
        real_validity = critic(real_data)
        fake_validity = critic(fake_data)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            critic, real_data.data, fake_data.data
        )

        # Total critic loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) \
                 + lambda_gp * gradient_penalty

        critic_optimizer.zero_grad()
        d_loss.backward()
        critic_optimizer.step()

    # Generator 학습
    z = torch.randn(batch_size, latent_dim)
    fake_data = generator(z)
    g_loss = -torch.mean(critic(fake_data))

    generator_optimizer.zero_grad()
    g_loss.backward()
    generator_optimizer.step()
```

#### 5.3.3 이론적 정당성

**정리 5.2 (Gradient Penalty의 효과)**
만약 최적 critic D*가 거의 모든 곳에서 ||∇D*|| = 1을 만족하면, D*는 Wasserstein distance를 정확히 계산합니다.

**증명 스케치**:
KR duality의 최적해는 다음을 만족:
```
∇D*(x) = (x_real - x_fake) / ||x_real - x_fake||  (거의 모든 곳에서)
⇒ ||∇D*|| = 1
```

**실전적 의미**:
- Gradient penalty는 최적해가 만족해야 할 필요조건 강제
- Weight clipping보다 부드러운 제약

### 5.4 Spectral Normalization

#### 5.4.1 동기

**목표**: 신경망의 Lipschitz 상수 직접 제어

**정의 5.2 (Spectral Norm)**
행렬 W의 spectral norm:
```
||W||₂ = σ_max(W) = max_{||x||=1} ||Wx||
```

이는 W의 최대 특이값(singular value)입니다.

**정리 5.3 (Lipschitz Constant of Neural Network)**
신경망 f(x) = W_L σ(W_{L-1} ... σ(W_1 x))에 대해:
```
||f||_L ≤ ∏ᵢ ||Wᵢ||₂
```

여기서 σ는 1-Lipschitz 활성화 함수 (ReLU, LeakyReLU 등).

#### 5.4.2 구현

**Algorithm 5.2 (Spectral Normalization)**
```python
def spectral_norm(W, u, n_iterations=1):
    """
    Power iteration으로 spectral norm 근사

    Args:
        W: [out_features, in_features] weight matrix
        u: [out_features] left singular vector 추정
        n_iterations: power iteration 횟수

    Returns:
        W_normalized: Spectral normalized weight
        u_new: Updated left singular vector
    """
    # Power iteration
    for _ in range(n_iterations):
        # v = W^T u / ||W^T u||
        v = W.t() @ u
        v = v / (v.norm() + 1e-12)

        # u = W v / ||W v||
        u = W @ v
        u = u / (u.norm() + 1e-12)

    # Spectral norm: σ = u^T W v
    sigma = u @ W @ v

    # Normalize W
    W_normalized = W / sigma

    return W_normalized, u
```

**PyTorch 구현**:
```python
import torch.nn.utils.spectral_norm as SpectralNorm

class SpectralNormalizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = SpectralNorm(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        )

    def forward(self, x):
        return self.conv(x)

# 사용
discriminator = nn.Sequential(
    SpectralNormalizedConv2d(3, 64, 4, stride=2, padding=1),
    nn.LeakyReLU(0.2),
    SpectralNormalizedConv2d(64, 128, 4, stride=2, padding=1),
    nn.LeakyReLU(0.2),
    # ...
)
```

**장점**:
1. **계산 효율적**: 매 iteration마다 하나의 power iteration만
2. **부드러운 제약**: Gradient penalty보다 계산 비용 낮음
3. **안정적**: Weight clipping의 문제 없음

**WGAN-GP vs Spectral Normalization**:
```
WGAN-GP:
+ 더 정확한 Wasserstein distance 근사
- Gradient penalty 계산 비용 높음 (backward pass 추가)

Spectral Normalization:
+ 계산 효율적
+ 메모리 효율적
- Lipschitz 상수 제어가 더 보수적
```

---

## 6. GAN 변형 모델들

### 6.1 DCGAN (Deep Convolutional GAN)

#### 6.1.1 아키텍처 가이드라인

**Radford et al., 2015**가 제안한 안정적 학습을 위한 원칙:

**Generator 아키텍처**:
```python
class DCGANGenerator(nn.Module):
    """
    DCGAN Generator

    핵심 원칙:
    1. Fully connected layer 제거, 모두 conv로
    2. Batch Normalization 사용 (출력층 제외)
    3. ReLU 활성화 (출력은 Tanh)
    4. Transposed Convolution으로 upsampling
    """
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super().__init__()

        self.init_size = 4  # Initial spatial size

        # Project and reshape
        self.fc = nn.Linear(latent_dim, feature_maps * 8 * self.init_size ** 2)

        # Transposed conv layers
        self.conv_blocks = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4,
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # 8x8 → 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2,
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # 16x16 → 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps,
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # 32x32 → 64x64
            nn.ConvTranspose2d(feature_maps, img_channels,
                              kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # [-1, 1] range
        )

    def forward(self, z):
        # z: [batch_size, latent_dim]
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x
```

**Discriminator 아키텍처**:
```python
class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator

    핵심 원칙:
    1. Strided convolution으로 downsampling (pooling 제거)
    2. Batch Normalization (첫 층 제외)
    3. LeakyReLU 활성화
    """
    def __init__(self, img_channels=3, feature_maps=64):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(img_channels, feature_maps,
                     kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 첫 층에는 BatchNorm 없음!

            # 32x32 → 16x16
            nn.Conv2d(feature_maps, feature_maps * 2,
                     kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 → 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4,
                     kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 → 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8,
                     kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 → 1x1
            nn.Conv2d(feature_maps * 8, 1,
                     kernel_size=4, stride=1, padding=0),
            # Sigmoid 제거 (BCEWithLogitsLoss 사용)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        return x.view(x.size(0), -1)
```

**왜 이런 선택들이 중요한가?**

1. **Batch Normalization**:
   - Internal covariate shift 감소
   - 더 높은 학습률 사용 가능
   - 정규화 효과

2. **Strided Convolution**:
   - 학습 가능한 downsampling
   - Pooling보다 더 나은 특징 학습

3. **LeakyReLU**:
   - Dying ReLU 문제 방지
   - 음수 영역에서도 기울기 전달

### 6.2 Progressive GAN

#### 6.2.1 핵심 아이디어

**문제**: 고해상도 이미지 직접 생성은 어려움

**해결**: 점진적으로 해상도 증가

**Algorithm 6.1 (Progressive Growing)**
```
1단계: 4×4 해상도에서 G와 D 학습
2단계: 8×8로 층 추가, fade-in
3단계: 16×16로 층 추가, fade-in
...
N단계: 1024×1024 최종 해상도
```

#### 6.2.2 Fade-in Mechanism

**부드러운 전환**:
```python
def fadein_layer(new_layer, old_output, alpha):
    """
    새 층을 부드럽게 추가

    Args:
        new_layer: 새로 추가되는 층
        old_output: 이전 해상도 출력 (upsampled)
        alpha: fade-in 가중치 [0, 1]

    Returns:
        Blended output
    """
    new_output = new_layer(...)

    # α=0: 완전히 old_output
    # α=1: 완전히 new_output
    output = alpha * new_output + (1 - alpha) * old_output
    return output
```

**학습 스케줄**:
```python
for resolution in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    # Stabilization phase: α = 1 (새 층만 사용)
    for _ in range(stabilization_iterations):
        train_step(alpha=1.0)

    # Fade-in phase: α: 0 → 1
    for step in range(fadein_iterations):
        alpha = step / fadein_iterations
        train_step(alpha=alpha)

    # Add next layer
    add_next_resolution_layer()
```

#### 6.2.3 추가 기법

**1. Minibatch Standard Deviation**:
```python
class MinibatchStdDev(nn.Module):
    """
    배치 내 다양성을 discriminator 입력에 추가
    """
    def forward(self, x):
        # x: [batch_size, channels, height, width]

        # 배치의 표준편차 계산
        std = torch.std(x, dim=0, keepdim=True)  # [1, C, H, W]
        mean_std = std.mean()  # 스칼라

        # 추가 채널로 concatenate
        batch_size, _, height, width = x.size()
        std_channel = mean_std.repeat(batch_size, 1, height, width)

        return torch.cat([x, std_channel], dim=1)
```

**2. Equalized Learning Rate**:
```python
class EqualizedConv2d(nn.Module):
    """
    He initialization을 런타임에 적용
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

        # He initialization scale
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = np.sqrt(2.0 / fan_in)

        # 가중치를 N(0,1)로 초기화
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.conv.weight.bias)

    def forward(self, x):
        # 런타임에 스케일링
        return self.conv(x * self.scale)
```

**효과**:
- 모든 가중치가 동일한 학습 속도
- 더 안정적인 학습

### 6.3 StyleGAN

#### 6.3.1 혁신적 아이디어

**핵심**: 스타일(style)과 노이즈(stochastic variation) 분리

**아키텍처 개요**:
```
z (latent) → Mapping Network → w (intermediate latent)
                                ↓
                         Style modulation
                                ↓
         Synthesis Network (+ noise injection)
                                ↓
                            Generated image
```

#### 6.3.2 Mapping Network

**목적**: 잠재 공간을 "disentangled" 공간으로 변환

```python
class MappingNetwork(nn.Module):
    """
    Z space → W space

    Z: 표준 정규분포 (entangled)
    W: 더 선형적이고 disentangled한 공간
    """
    def __init__(self, latent_dim=512, num_layers=8):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2)
            ])

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        # z: [batch_size, latent_dim]
        w = self.mapping(z)  # [batch_size, latent_dim]
        return w
```

**왜 8층이나?**
- 깊은 네트워크가 비선형 변환을 더 잘 학습
- Disentanglement 향상

#### 6.3.3 Adaptive Instance Normalization (AdaIN)

**핵심 메커니즘**:
```python
def adaptive_instance_norm(content, style):
    """
    AdaIN(x, y) = σ(y) * normalize(x) + μ(y)

    Args:
        content: [batch_size, channels, height, width]
        style: [batch_size, channels] - 스타일 벡터

    Returns:
        Modulated features
    """
    # Instance normalization
    size = content.size()
    content_mean = content.view(size[0], size[1], -1).mean(2).view(size[0], size[1], 1, 1)
    content_std = content.view(size[0], size[1], -1).std(2).view(size[0], size[1], 1, 1)

    normalized = (content - content_mean) / (content_std + 1e-8)

    # Style modulation
    style_mean = style[:, :size[1]]  # μ(y)
    style_std = style[:, size[1]:]   # σ(y)

    output = style_std.view(size[0], size[1], 1, 1) * normalized \
             + style_mean.view(size[0], size[1], 1, 1)

    return output
```

**StyleGAN 블록**:
```python
class StyleGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=512):
        super().__init__()

        # Convolution
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Style modulation
        self.style = nn.Linear(latent_dim, out_channels * 2)  # mean & std

        # Noise injection
        self.noise_weight = nn.Parameter(torch.zeros(1))

        # Activation
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w, noise=None):
        # x: [batch_size, in_channels, H, W]
        # w: [batch_size, latent_dim]

        # Convolution
        out = self.conv(x)

        # Style modulation (AdaIN)
        style = self.style(w)
        out = adaptive_instance_norm(out, style)

        # Noise injection
        if noise is None:
            noise = torch.randn(out.size(0), 1, out.size(2), out.size(3), device=out.device)
        out = out + self.noise_weight * noise

        # Activation
        out = self.activation(out)

        return out
```

#### 6.3.4 Style Mixing

**기법**: 다른 w 벡터를 다른 해상도에 사용

```python
def style_mixing_regularization(generator, z1, z2, mixing_prob=0.9):
    """
    Style mixing for regularization

    일정 확률로 두 개의 잠재 코드 혼합
    """
    if random.random() < mixing_prob:
        # 두 잠재 코드 생성
        w1 = generator.mapping(z1)
        w2 = generator.mapping(z2)

        # 무작위로 분할점 선택
        crossover_point = random.randint(1, generator.num_layers - 1)

        # 혼합: 앞쪽 층은 w1, 뒤쪽 층은 w2
        w_mixed = [w1] * crossover_point + [w2] * (generator.num_layers - crossover_point)

        return generator.synthesis(w_mixed)
    else:
        w = generator.mapping(z1)
        return generator.synthesis([w] * generator.num_layers)
```

**효과**:
- 스타일 간 독립성 강화
- Localization 향상 (각 층이 특정 특징 담당)

#### 6.3.5 스타일 계층 구조

**경험적 관찰** (Karras et al., 2019):
```
Low resolution (4×4 ~ 8×8):
- 거시적 특징 (포즈, 얼굴 형태)
- "Coarse" styles

Medium resolution (16×16 ~ 32×32):
- 얼굴 특징 (눈, 코, 입)
- "Middle" styles

High resolution (64×64 ~ 1024×1024):
- 미세한 디테일 (피부 텍스처, 머리카락)
- "Fine" styles
```

### 6.4 BigGAN

#### 6.4.1 Scaling Up

**핵심 전략**:
1. **대규모 배치 크기**: 2048 (엄청남!)
2. **깊고 넓은 네트워크**: ResNet 기반
3. **클래스 조건부 Batch Normalization**

**Class-conditional BatchNorm**:
```python
class ConditionalBatchNorm2d(nn.Module):
    """
    Batch normalization with class-conditional scaling and bias
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features

        # Standard batch norm (without affine)
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # Class-conditional parameters
        self.gamma_embed = nn.Embedding(num_classes, num_features)
        self.beta_embed = nn.Embedding(num_classes, num_features)

    def forward(self, x, class_labels):
        # x: [batch_size, num_features, H, W]
        # class_labels: [batch_size]

        # Standard normalization
        out = self.bn(x)

        # Class-conditional modulation
        gamma = self.gamma_embed(class_labels).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(class_labels).view(-1, self.num_features, 1, 1)

        out = gamma * out + beta
        return out
```

#### 6.4.2 Truncation Trick

**문제**: Latent space의 극단 영역은 학습 부족

**해결**: 샘플링 시 truncation 적용

```python
def truncated_sampling(mean=0, std=1, truncation=0.5):
    """
    Truncated normal sampling

    Args:
        truncation: [0, 1]
            0 = 평균으로 수렴
            1 = 표준 정규분포
    """
    z = np.random.normal(mean, std, size=(batch_size, latent_dim))

    # Truncate: z를 [-threshold, threshold]로 clipping
    threshold = truncation * 2  # 2σ rule
    z = np.clip(z, -threshold, threshold)

    return z
```

**Trade-off**:
- 낮은 truncation: 더 나은 품질, 낮은 다양성
- 높은 truncation: 높은 다양성, 낮은 품질

#### 6.4.3 Orthogonal Regularization

**목적**: Generator 가중치 조건수(condition number) 개선

```python
def orthogonal_regularization(model, strength=1e-4):
    """
    Regularize weights to be orthogonal

    For weight matrix W:
    Minimize ||W^T W - I||²_F
    """
    reg_loss = 0
    for param in model.parameters():
        if len(param.shape) < 2:  # Skip biases
            continue

        W = param
        # Flatten to 2D
        W = W.view(W.size(0), -1)

        # W^T W
        WtW = W.t() @ W

        # ||W^T W - I||²
        reg = ((WtW - torch.eye(WtW.size(0), device=W.device)) ** 2).sum()
        reg_loss += reg

    return strength * reg_loss
```

**효과**:
- 학습 안정성 향상
- Gradient flow 개선

---

## 7. 이론적 분석과 한계

### 7.1 수렴성 분석

#### 7.1.1 비수렴 예시

**정리 7.1 (Dirac-GAN)**
가장 단순한 GAN도 수렴 보장이 없습니다 (Mescheder et al., 2018).

**예시**: Dirac-GAN
```
Generator: G(z) = θ_G (scalar parameter)
Discriminator: D(x) = θ_D · x (linear)
Real data: x = 0 (Dirac delta at 0)

목적 함수:
V = θ_D · 0 + log(1 + exp(-θ_D · θ_G))

Gradients:
∇_θ_D V = -θ_G / (1 + exp(θ_D · θ_G))
∇_θ_G V = θ_D / (1 + exp(θ_D · θ_G))

동역학:
dθ_D/dt = -θ_G / (1 + exp(θ_D · θ_G))
dθ_G/dt = -θ_D / (1 + exp(θ_D · θ_G))
```

**분석**:
```python
# 수치적 시뮬레이션
theta_G, theta_D = 0.5, 0.5
learning_rate = 0.1

trajectory = [(theta_G, theta_D)]

for _ in range(1000):
    exp_term = np.exp(theta_D * theta_G)
    grad_D = -theta_G / (1 + exp_term)
    grad_G = theta_D / (1 + exp_term)

    theta_D -= learning_rate * grad_D
    theta_G -= learning_rate * grad_G

    trajectory.append((theta_G, theta_D))

# 결과: 원점 주위를 회전하며 수렴하지 않음
```

**의미**: 가장 단순한 설정에서도 GAN이 불안정!

#### 7.1.2 Consensus Optimization

**해결책** (Mescheder et al., 2017):

**Regularized GAN Objective**:
```
L_reg = V(G, D) + γ/2 (||∇_θ_D V||² + ||∇_θ_G V||²)
```

**효과**:
- Nash 균형 근처에서 안정화
- Gradient magnitude 제약

**정리 7.2 (Consensus Optimization)**
적절한 γ 선택 시, regularized objective는 Nash 균형 근처에서 국소적으로 수렴합니다.

**증명 스케치**:
Lyapunov 함수 V_cons = ||θ - θ*||²를 정의. 정규화 항이 dV_cons/dt < 0을 보장.

#### 7.1.3 최근 이론적 진전

**Arora et al., 2017**: "Generalization and Equilibrium in GANs"
- Neural network discriminator의 유한 capacity 고려
- 일반화 오차 분석

**핵심 결과**:
```
p_g와 p_data 사이의 거리 ≤ ε_approx + ε_gen + ε_opt

여기서:
- ε_approx: 최적 discriminator의 근사 오차
- ε_gen: 유한 샘플로 인한 일반화 오차
- ε_opt: 최적화 오차
```

### 7.2 평가의 어려움

#### 7.2.1 Likelihood 평가 불가

**문제**: GAN은 명시적 likelihood 없음
```
VAE:  p_θ(x) 근사 가능 (ELBO)
GAN:  p_g(x) 계산 불가 (암묵적 모델)
```

**결과**:
- 표준 likelihood-based 지표 사용 불가
- Overfitting 감지 어려움

#### 7.2.2 Inception Score (IS)

**정의**:
```
IS(G) = exp(E_x [D_KL(p(y|x) || p(y))])

여기서:
- x ~ p_g: 생성된 이미지
- p(y|x): Inception 모델의 예측 분포
- p(y) = E_x[p(y|x)]: 주변 분포
```

**해석**:
1. **p(y|x) sharp**: 각 이미지는 명확한 클래스
2. **p(y) uniform**: 다양한 클래스 생성

**수식 전개**:
```
IS = exp(E_x [Σ_y p(y|x) log(p(y|x)/p(y))])
   = exp(H(Y) - E_X[H(Y|X)])
```

**한계**:
- Inception 모델에 의존
- 실제 데이터와 비교 없음
- Mode collapse 완전 포착 못함

**구현**:
```python
def inception_score(images, inception_model, splits=10):
    """
    Inception Score 계산

    Args:
        images: Generated images [N, C, H, W]
        inception_model: Pre-trained Inception v3
        splits: Number of splits for std calculation

    Returns:
        mean, std: IS mean and standard deviation
    """
    # Resize to 299x299 for Inception
    images = F.interpolate(images, size=(299, 299), mode='bilinear')

    # Get predictions
    with torch.no_grad():
        preds = inception_model(images)
        preds = F.softmax(preds, dim=1).cpu().numpy()

    # Split and compute IS for each split
    scores = []
    N = len(preds)
    split_size = N // splits

    for i in range(splits):
        part = preds[i * split_size: (i + 1) * split_size]

        # p(y|x)
        py_given_x = part

        # p(y) = E[p(y|x)]
        py = np.mean(part, axis=0)

        # KL divergence for each sample
        kl = py_given_x * (np.log(py_given_x + 1e-16) - np.log(py + 1e-16))
        kl = np.sum(kl, axis=1)

        # IS for this split
        scores.append(np.exp(np.mean(kl)))

    return np.mean(scores), np.std(scores)
```

#### 7.2.3 Fréchet Inception Distance (FID)

**동기**: IS의 한계 극복, 실제 데이터와 비교

**정의**:
```
FID(real, fake) = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2(Σ_real Σ_fake)^{1/2})
```

여기서:
- μ, Σ: Inception 특징 공간의 평균과 공분산
- Tr: Trace (대각합)

**이론적 배경**:
두 다변량 가우시안 사이의 Wasserstein-2 distance (Fréchet distance)

**정리 7.3 (Fréchet Distance for Gaussians)**
```
p ~ N(μ_1, Σ_1), q ~ N(μ_2, Σ_2)

W_2²(p, q) = ||μ_1 - μ_2||² + Tr(Σ_1 + Σ_2 - 2(Σ_1 Σ_2)^{1/2})
```

**구현**:
```python
import numpy as np
from scipy import linalg

def calculate_fid(real_features, fake_features):
    """
    Fréchet Inception Distance

    Args:
        real_features: [N_real, feature_dim] real image features
        fake_features: [N_fake, feature_dim] fake image features

    Returns:
        fid: Scalar FID score
    """
    # 평균과 공분산 계산
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Mean difference
    diff = mu_real - mu_fake
    mean_term = np.dot(diff, diff)

    # Covariance term: Tr(Σ_real + Σ_fake - 2√(Σ_real Σ_fake))
    # Matrix square root
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    cov_term = np.trace(sigma_real + sigma_fake - 2 * covmean)

    fid = mean_term + cov_term
    return fid
```

**장점**:
- 실제 데이터와 직접 비교
- Mode collapse에 민감
- 인간 평가와 상관관계 높음

**단점**:
- Inception 모델 의존
- 계산 비용 (많은 샘플 필요)

#### 7.2.4 Precision and Recall

**동기**: 품질(precision)과 다양성(recall) 분리 측정

**정의** (Kynkäänniemi et al., 2019):
```
Precision = |{fake ∈ support(real)}| / |{fake}|
Recall = |{real ∈ support(fake)}| / |{real}|
```

**구현 (k-NN 기반)**:
```python
def compute_precision_recall(real_features, fake_features, k=3):
    """
    k-NN based precision and recall

    Args:
        real_features: [N_real, dim]
        fake_features: [N_fake, dim]
        k: Number of neighbors

    Returns:
        precision, recall
    """
    from sklearn.neighbors import NearestNeighbors

    # Real manifold
    real_nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    real_nn.fit(real_features)

    # Fake manifold
    fake_nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    fake_nn.fit(fake_features)

    # Precision: fake samples close to real manifold
    distances_to_real, _ = real_nn.kneighbors(fake_features)
    real_radii = distances_to_real[:, -1]  # k-th nearest neighbor distance

    distances_fake_to_real, _ = real_nn.kneighbors(fake_features)
    in_real_manifold = distances_fake_to_real[:, 0] < real_radii.mean()
    precision = in_real_manifold.mean()

    # Recall: real samples close to fake manifold
    distances_to_fake, _ = fake_nn.kneighbors(real_features)
    fake_radii = distances_to_fake[:, -1]

    distances_real_to_fake, _ = fake_nn.kneighbors(real_features)
    in_fake_manifold = distances_real_to_fake[:, 0] < fake_radii.mean()
    recall = in_fake_manifold.mean()

    return precision, recall
```

**해석**:
- **High Precision, Low Recall**: 고품질이지만 다양성 부족 (mode collapse)
- **Low Precision, High Recall**: 다양하지만 품질 낮음
- **High Precision, High Recall**: 이상적!

### 7.3 이론적 한계

#### 7.3.1 암묵적 모델의 본질적 어려움

**문제 7.1 (Intractable Likelihood)**
```
p_g(x) = ∫ δ(x - G(z)) p(z) dz
```

이 적분은 일반적으로 계산 불가능합니다.

**결과**:
- Overfitting 감지 어려움
- 확률 기반 평가 불가
- Bayesian inference 적용 불가

#### 7.3.2 학습 목표의 모호성

**문제**: 다양한 divergence 최소화 가능
- Original GAN: JS divergence
- WGAN: Wasserstein distance
- f-GAN: 임의의 f-divergence

**정리 7.4 (Divergence Mismatch)**
다른 divergence는 다른 특성:
- KL: Mode-covering
- Reverse KL: Mode-seeking
- JS: 중간
- Wasserstein: 지지 불일치에 강건

**의미**: "올바른" divergence는 응용에 의존

#### 7.3.3 Discriminator의 과도한 힘

**문제 7.2 (Discriminator Overfitting)**
유한 데이터로 학습 시 Discriminator가 쉽게 과적합:
```
D(x) → 1 for all x in training set
D(x) → 0 elsewhere
```

**결과**:
- Generator에 무의미한 gradient
- 학습 정체

**완화**:
- Regularization (spectral norm, gradient penalty)
- Data augmentation
- Discriminator capacity 제한

---

## 8. 최신 연구 동향 (2023-2025)

### 8.1 Diffusion Models와의 융합

#### 8.1.1 Adversarial Diffusion Distillation

**동기**: Diffusion 모델은 고품질이지만 느림, GAN은 빠르지만 학습 어려움

**아이디어**: Diffusion 모델을 teacher로, GAN을 student로

**방법** (Sauer et al., 2023):
```
1. Pre-trained diffusion model로 샘플 생성
2. GAN generator를 distillation loss로 학습:
   L = L_adversarial + λ·L_distillation

L_distillation = ||G(z) - Diffusion(t, noise)||²
```

**효과**:
- Diffusion 수준의 품질
- GAN 수준의 속도 (1-step generation)

#### 8.1.2 Denoising Diffusion GAN

**방법** (Xiao et al., 2022):
```
Generator: Diffusion model의 denoiser
Discriminator: Time-dependent discriminator D(x, t)

목적 함수:
min_G max_D E_t [E_{x~q_t} [log D(x, t)] + E_{x~p_t} [log(1 - D(G(x,t), t))]]
```

**장점**:
- Diffusion의 안정성
- GAN의 샘플링 속도

### 8.2 Text-to-Image Generation

#### 8.2.1 GAN-based Approaches

**AttnGAN** (Xu et al., 2018):
```
Text Encoder → Word features + Sentence feature
              ↓
Multi-stage Generator (64×64 → 128×128 → 256×256)
              ↓
Attention mechanism: Generator attends to relevant words
```

**StackGAN++** (Zhang et al., 2018):
- Multi-scale architecture
- Color-consistency regularization
- Conditional augmentation

#### 8.2.2 현재 상황

**관찰**: Diffusion 모델이 우세 (DALL-E 2, Stable Diffusion, Midjourney)

**이유**:
- 더 안정적인 학습
- 더 나은 mode coverage
- 더 높은 해상도

**GAN의 역할**:
- Real-time application (diffusion distillation)
- Specific domains (얼굴, 특정 스타일)
- Editing and manipulation

### 8.3 GAN Inversion

#### 8.3.1 문제 정의

**GAN Inversion**: 실제 이미지 x에 대해 z를 찾기
```
min_z ||G(z) - x||² + R(z)
```

**응용**:
- Image editing
- Style transfer
- Interpolation

#### 8.3.2 방법

**1. Optimization-based**:
```python
def gan_inversion_optimization(generator, target_image, num_steps=1000):
    """
    Gradient descent로 z 찾기
    """
    z = torch.randn(1, latent_dim, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=0.01)

    for step in range(num_steps):
        generated = generator(z)
        loss = F.mse_loss(generated, target_image)

        # Perceptual loss 추가 (선택적)
        loss += perceptual_loss(generated, target_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return z
```

**2. Encoder-based**:
```python
class Encoder(nn.Module):
    """
    x → z 직접 학습
    """
    def __init__(self):
        # ResNet backbone
        self.encoder = resnet50(pretrained=True)
        self.fc = nn.Linear(2048, latent_dim)

    def forward(self, x):
        features = self.encoder(x)
        z = self.fc(features)
        return z

# 학습
encoder_loss = F.mse_loss(generator(encoder(x)), x)
```

**3. Hybrid**:
Encoder로 초기화 + Optimization으로 정제

#### 8.3.3 Semantic Editing

**GAN의 latent space에서 의미 있는 방향 찾기**:

```python
# 예: "smile" 방향 찾기
def find_semantic_direction(generator, attribute_classifier, attribute='smile'):
    """
    Supervised approach
    """
    # 많은 z 샘플링
    z_samples = torch.randn(1000, latent_dim)
    images = generator(z_samples)

    # Attribute scores
    scores = attribute_classifier(images, attribute)

    # Linear regression: score = w^T z + b
    from sklearn.svm import SVR
    model = SVR()
    model.fit(z_samples, scores)

    # Direction
    direction = model.coef_
    return direction

# 사용
direction = find_semantic_direction(generator, classifier, 'smile')

# Editing
z_original = encode(image)
z_edited = z_original + alpha * direction  # alpha controls strength
image_edited = generator(z_edited)
```

### 8.4 3D-aware GANs

#### 8.4.1 EG3D (NVIDIA, 2022)

**핵심**: Explicit 3D representation + Neural rendering

**아키텍처**:
```
z → Generator → Triplane representation (3×256×256×C)
                ↓
Camera pose → Ray marching + Volume rendering
                ↓
              2D image
```

**Triplane representation**:
- 3개의 orthogonal planes
- 3D point (x,y,z) → query 3 planes → aggregate features

**효과**:
- Multi-view consistent
- 3D geometry 학습
- Pose control 가능

#### 8.4.2 응용

**Virtual try-on**: 3D body model + GAN
**Face reenactment**: 3D face model + expression control
**Novel view synthesis**: Single image → 3D scene

### 8.5 미해결 문제

#### 8.5.1 이론적 문제

1. **완전한 수렴성 증명**
   - 현실적 설정에서 수렴 보장?
   - 어떤 조건 필요?

2. **최적 Divergence 선택**
   - 응용별 올바른 divergence?
   - 자동 선택 가능?

3. **Mode Coverage 보장**
   - 이론적으로 모든 모드 커버 보장?
   - 필요충분조건?

#### 8.5.2 실용적 문제

1. **극도로 고해상도** (4K+)
   - 메모리 제약
   - 학습 시간

2. **Few-shot Learning**
   - 적은 데이터로 GAN 학습?
   - Transfer learning 한계?

3. **Controllability**
   - Fine-grained control
   - Disentangled editing

### 8.6 연구 방향 제안

#### 8.6.1 단기 (1-2년)

1. **Hybrid Architectures**
   - GAN + Diffusion 결합 최적화
   - 장점만 취하기

2. **Improved Evaluation**
   - 더 나은 자동 평가 지표
   - 인간 평가 자동화

3. **Efficient Training**
   - 더 적은 데이터로
   - 더 빠른 수렴

#### 8.6.2 장기 (3-5년)

1. **이론 기반 강화**
   - 수렴 보장 알고리즘
   - Optimal divergence 이론

2. **3D 및 4D**
   - Temporal coherence
   - 동영상 생성

3. **Interactive Generation**
   - Real-time editing
   - User-in-the-loop

---

## 9. 종합 및 결론

### 9.1 GAN의 핵심 기여

**이론적 기여**:
1. **게임 이론 프레임워크**: 생성 모델을 게임으로 정식화
2. **암묵적 생성 모델**: Likelihood-free training
3. **Adversarial learning**: Discriminator를 teacher로

**실용적 기여**:
1. **고품질 샘플**: VAE보다 sharp한 이미지
2. **빠른 샘플링**: 단일 forward pass
3. **다양한 응용**: Image synthesis, editing, translation

**영향**:
- 컴퓨터 비전 혁명
- 창의적 AI 응용
- 다른 도메인으로 확장 (텍스트, 오디오, 3D)

### 9.2 다른 생성 모델과의 비교 (재방문)

| 특성 | VAE | GAN | Diffusion | Autoregressive |
|------|-----|-----|-----------|----------------|
| **Likelihood** | 명시적 (ELBO) | 암묵적 | 명시적 | 명시적 |
| **학습 안정성** | 높음 | 낮음 | 높음 | 높음 |
| **샘플 품질** | 중간 | 높음 | 매우 높음 | 높음 |
| **샘플 속도** | 빠름 (1-step) | 빠름 (1-step) | 느림 (1000-step) | 매우 느림 (sequential) |
| **Mode Coverage** | 좋음 | 나쁨 | 좋음 | 좋음 |
| **Controllability** | 중간 | 높음 (latent) | 높음 (guidance) | 낮음 |

### 9.3 학습 로드맵

**초급 (1-2주)**:
1. 이론: Section 1-2 (게임 이론, GAN 정의)
2. 구현: Vanilla GAN on MNIST
3. 실험: Mode collapse 관찰

**중급 (3-4주)**:
4. 이론: Section 3-4 (학습 알고리즘, 안정화)
5. 구현: DCGAN, WGAN
6. 실험: FID, IS 계산

**고급 (5-8주)**:
7. 이론: Section 5-7 (WGAN 이론, 변형 모델)
8. 구현: StyleGAN, Progressive GAN
9. 연구: 최신 논문 및 응용

### 9.4 필독 논문

**기초 (필수)**:
1. ⭐⭐⭐ Generative Adversarial Nets (Goodfellow et al., 2014)
2. ⭐⭐ NIPS 2016 Tutorial: GANs (Goodfellow, 2016)

**안정화 기법**:
3. ⭐⭐⭐ Wasserstein GAN (Arjovsky et al., 2017)
4. ⭐⭐⭐ Improved Training of WGANs (Gulrajani et al., 2017)
5. ⭐⭐ Spectral Normalization (Miyato et al., 2018)

**아키텍처**:
6. ⭐⭐ DCGAN (Radford et al., 2015)
7. ⭐⭐ Progressive GAN (Karras et al., 2017)
8. ⭐⭐⭐ StyleGAN (Karras et al., 2019)

**이론**:
9. ⭐⭐ f-GAN (Nowozin et al., 2016)
10. ⭐ Do GANs learn the distribution? (Arora et al., 2017)

### 9.5 실전 조언

**학습 안정화**:
1. Spectral normalization 사용
2. Two-timescale update rule (TTUR)
3. Gradient penalty (λ=10)
4. Label smoothing (one-sided, 0.9)

**하이퍼파라미터**:
```python
recommended_config = {
    'generator_lr': 1e-4,
    'discriminator_lr': 4e-4,  # D를 G보다 빠르게
    'optimizer': 'Adam',
    'beta1': 0.0,  # 0.5가 아닌 0!
    'beta2': 0.999,
    'batch_size': 64,
    'n_critic': 5,  # WGAN
    'gp_lambda': 10,  # WGAN-GP
}
```

**디버깅 체크리스트**:
- [ ] Mode collapse 발생?
- [ ] Discriminator가 perfect (loss→0)?
- [ ] Generator loss가 증가?
- [ ] 생성 샘플이 의미 있는가?
- [ ] FID가 감소하는가?

---

## 부록

### A. 수학적 배경

#### A.1 게임 이론 심화

**정리 A.1 (Minimax Theorem - General Form)**
```
Compact convex strategy spaces X, Y
Continuous bilinear payoff u(x,y)

⇒ max_x min_y u(x,y) = min_y max_x u(x,y)
```

**증명**: Sion's minimax theorem (강한 duality)

#### A.2 최적 운송 이론 (Optimal Transport)

**Monge Problem** (1781):
```
최소 비용으로 질량 p를 q로 재배치:
min_T ∫ c(x, T(x)) p(x) dx
subject to: T#p = q  (push-forward)
```

**Kantorovich Relaxation** (1942):
```
min_γ ∫∫ c(x,y) dγ(x,y)
subject to: ∫ γ(x,y) dy = p(x)
           ∫ γ(x,y) dx = q(y)
```

**연결**: Wasserstein distance는 Kantorovich problem의 optimal value

#### A.3 함수 공간의 기하학

**Riemannian Metric on P(X)**:
```
P(X): 확률 분포의 공간

Wasserstein-2 metric은 Fisher-Rao metric의 특별한 경우
```

**Gradient Flow**:
```
∂p/∂t = ∇·(p∇(δF/δp))

F: 자유 에너지 함수
δF/δp: 변분 도함수
```

### B. 구현 팁

#### B.1 완전한 WGAN-GP 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

def compute_gradient_penalty(D, real_samples, fake_samples):
    """WGAN-GP gradient penalty"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.ones(real_samples.size(0), 1).requires_grad_(False).to(real_samples.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training loop
generator = Generator()
discriminator = Discriminator()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.0, 0.9))

lambda_gp = 10
n_critic = 5

for epoch in range(n_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):

        # Train Discriminator
        for _ in range(n_critic):
            optimizer_D.zero_grad()

            z = torch.randn(real_imgs.size(0), latent_dim)
            fake_imgs = generator(z).detach()

            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)

            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        z = torch.randn(real_imgs.size(0), latent_dim)
        gen_imgs = generator(z)

        g_loss = -torch.mean(discriminator(gen_imgs))

        g_loss.backward()
        optimizer_G.step()
```

#### B.2 디버깅 도구

```python
class GANMonitor:
    """GAN 학습 모니터링"""

    def __init__(self):
        self.d_losses = []
        self.g_losses = []
        self.fid_scores = []

    def log_step(self, d_loss, g_loss, epoch, step):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)

        # Early warning signs
        if d_loss < 0.01:
            print(f"WARNING: Discriminator too strong at epoch {epoch}, step {step}")

        if g_loss > 10:
            print(f"WARNING: Generator loss exploding at epoch {epoch}, step {step}")

    def plot_losses(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.d_losses, label='Discriminator')
        plt.plot(self.g_losses, label='Generator')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')

        plt.subplot(1, 2, 2)
        plt.plot(self.fid_scores)
        plt.xlabel('Epoch')
        plt.ylabel('FID')
        plt.title('FID Score over Training')

        plt.tight_layout()
        plt.show()
```

### C. 추가 자료

**온라인 튜토리얼**:
- PyTorch GAN Tutorial: [pytorch.org/tutorials](https://pytorch.org/tutorials)
- Stanford CS236: Deep Generative Models
- MIT 6.S191: Introduction to Deep Learning

**코드 저장소**:
- [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN): 다양한 GAN 구현
- [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch): NVIDIA 공식
- [WGAN-GP](https://github.com/caogang/wgan-gp): 원본 구현

**도서**:
- "GANs in Action" (Jakub Langr & Vladimir Bok)
- "Deep Learning" Chapter 20 (Goodfellow et al.)

---

**마지막 업데이트**: 2025-01
**작성자**: Generative AI Study Group
**라이선스**: CC BY-NC-SA 4.0

---

**감사의 말**:
이 문서는 수많은 연구자들의 기여를 바탕으로 작성되었습니다. 특히 Ian Goodfellow, Martin Arjovsky, Tero Karras 등 GAN 분야를 개척한 연구자들께 감사드립니다.
