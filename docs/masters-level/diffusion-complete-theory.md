# Diffusion Models 완전 이론

> 석사 과정 학생을 위한 Diffusion Models의 완전한 수학적 기초와 이론적 분석

## 목차

1. [서론 및 개요](#1-서론-및-개요)
2. [Forward Diffusion Process의 수학적 기초](#2-forward-diffusion-process의-수학적-기초)
3. [Reverse Diffusion Process와 Score Function](#3-reverse-diffusion-process와-score-function)
4. [Score Matching Theory의 완전한 증명](#4-score-matching-theory의-완전한-증명)
5. [DDPM의 수학적 기초](#5-ddpm의-수학적-기초)
6. [DDIM과 Deterministic Sampling](#6-ddim과-deterministic-sampling)
7. [고급 기법들](#7-고급-기법들)
8. [최신 연구 동향 (2023-2025)](#8-최신-연구-동향-2023-2025)
9. [결론 및 향후 연구 방향](#9-결론-및-향후-연구-방향)
10. [부록](#10-부록)

---

## 1. 서론 및 개요

### 1.1 Diffusion Models란?

Diffusion Models는 데이터에 점진적으로 노이즈를 추가하는 **forward process**와 이를 역으로 제거하는 **reverse process**를 학습하는 생성 모델입니다. 2020년대 들어 이미지 생성, 오디오 합성, 분자 생성 등 다양한 분야에서 SOTA 성능을 달성하며 생성 모델의 새로운 패러다임을 제시했습니다.

**핵심 아이디어:**
- **Forward Process**: 데이터 x₀에 Gaussian noise를 T 스텝에 걸쳐 점진적으로 추가
- **Reverse Process**: 노이즈 x_T로부터 원본 데이터 x₀를 복원하는 과정을 학습
- **Score Function**: 데이터 분포의 gradient ∇_x log p(x)를 추정하여 reverse process 구현

### 1.2 역사적 배경

**주요 발전 과정:**

1. **Non-equilibrium Thermodynamics (2015)**
   - Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
   - 열역학의 확산 과정을 머신러닝에 적용
   - Forward/Reverse Markov Chain 개념 도입

2. **Score Matching (2019)**
   - Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution"
   - Score function ∇_x log p(x) 추정을 통한 생성 모델
   - Langevin dynamics를 이용한 샘플링

3. **DDPM (2020)**
   - Ho et al., "Denoising Diffusion Probabilistic Models"
   - 단순화된 목적 함수와 효율적인 학습 방법 제시
   - 이미지 생성에서 GAN 수준의 품질 달성

4. **Score SDE (2021)**
   - Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations"
   - 이산 시간 diffusion을 연속 시간 SDE로 일반화
   - Probability flow ODE를 통한 deterministic sampling

5. **DALL-E 2, Stable Diffusion (2022)**
   - Text-to-image generation의 혁신
   - Latent diffusion을 통한 효율성 개선
   - Classifier-free guidance로 제어 가능성 향상

6. **Consistency Models, Flow Matching (2023-2024)**
   - 1-step 또는 few-step generation
   - ODE/SDE의 통합 이론
   - 샘플링 속도의 획기적 개선

### 1.3 다른 생성 모델과의 비교

| 특성 | VAE | GAN | Diffusion Models |
|------|-----|-----|------------------|
| **학습 안정성** | 높음 | 낮음 (mode collapse) | 높음 |
| **샘플 품질** | 중간 (흐릿함) | 높음 | 매우 높음 |
| **다양성** | 높음 | 낮음 (mode dropping) | 매우 높음 |
| **Likelihood 계산** | Approximate | 불가능 | Exact (ODE) |
| **샘플링 속도** | 빠름 (1 step) | 빠름 (1 step) | 느림 (100-1000 steps) |
| **학습 비용** | 낮음 | 중간 | 높음 |

**Diffusion Models의 장점:**
- 학습이 안정적이고 mode collapse가 없음
- 높은 샘플 품질과 다양성
- 이론적으로 엄밀한 수학적 기초
- Exact likelihood 계산 가능 (Probability Flow ODE)

**Diffusion Models의 단점:**
- 샘플링이 매우 느림 (수백~수천 스텝)
- 학습에 많은 계산 자원 필요
- 고해상도 이미지 생성 시 메모리 요구량이 큼

---

## 2. Forward Diffusion Process의 수학적 기초

### 2.1 Markov Chain으로서의 Forward Process

Forward diffusion process는 원본 데이터 x₀에 점진적으로 Gaussian noise를 추가하는 Markov chain입니다.

**정의 2.1 (Forward Process)**

시간 단계 t = 0, 1, ..., T에 대해, forward process는 다음과 같이 정의됩니다:

```
q(x₁, ..., x_T | x₀) = ∏_{t=1}^T q(x_t | x_{t-1})
```

여기서 각 전이 확률은 Gaussian distribution입니다:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

**노이즈 스케줄 (Noise Schedule):**
- β₁, β₂, ..., β_T는 미리 정의된 variance schedule
- 일반적으로 0 < β₁ < β₂ < ... < β_T < 1
- 선형 스케줄: β_t = β_min + (β_max - β_min) · t/T
- 코사인 스케줄: 더 완만한 노이즈 추가

### 2.2 임의의 시간 t에서의 분포

Forward process의 핵심 성질은 임의의 시간 t에서 x_t를 x₀로부터 직접 계산할 수 있다는 것입니다.

**정리 2.1 (Closed-form Forward)**

α_t = 1 - β_t, ᾱ_t = ∏_{s=1}^t α_s라고 정의하면:

```
q(x_t | x₀) = N(x_t; √ᾱ_t x₀, (1-ᾱ_t)I)
```

**증명:**

재매개변수화를 사용하여 귀납법으로 증명합니다.

*Base case (t=1):*
```
x₁ = √(1-β₁) x₀ + √β₁ ε₁,  ε₁ ~ N(0,I)
   = √α₁ x₀ + √(1-α₁) ε₁
   = √ᾱ₁ x₀ + √(1-ᾱ₁) ε₁  (∵ ᾱ₁ = α₁)

∴ q(x₁ | x₀) = N(x₁; √ᾱ₁ x₀, (1-ᾱ₁)I) ✓
```

*Inductive step:*

q(x_t | x₀) = N(x_t; √ᾱ_t x₀, (1-ᾱ_t)I)가 성립한다고 가정하면:

```
x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε_t,  ε_t ~ N(0,I)

x_{t+1} = √α_{t+1} x_t + √(1-α_{t+1}) ε_{t+1}
        = √α_{t+1} (√ᾱ_t x₀ + √(1-ᾱ_t) ε_t) + √(1-α_{t+1}) ε_{t+1}
        = √(α_{t+1} ᾱ_t) x₀ + √(α_{t+1}(1-ᾱ_t)) ε_t + √(1-α_{t+1}) ε_{t+1}
```

두 독립적인 Gaussian noise의 합:
```
√(α_{t+1}(1-ᾱ_t)) ε_t + √(1-α_{t+1}) ε_{t+1} ~ N(0, α_{t+1}(1-ᾱ_t)I + (1-α_{t+1})I)
                                                = N(0, (α_{t+1} - α_{t+1}ᾱ_t + 1 - α_{t+1})I)
                                                = N(0, (1 - α_{t+1}ᾱ_t)I)
```

따라서:
```
x_{t+1} = √(α_{t+1}ᾱ_t) x₀ + √(1-α_{t+1}ᾱ_t) ε
        = √ᾱ_{t+1} x₀ + √(1-ᾱ_{t+1}) ε  (∵ ᾱ_{t+1} = α_{t+1}ᾱ_t)

∴ q(x_{t+1} | x₀) = N(x_{t+1}; √ᾱ_{t+1} x₀, (1-ᾱ_{t+1})I) ✓
```

**재매개변수화 표현:**

```python
def q_sample(x_0, t, noise=None):
    """
    Forward diffusion: x_t를 x_0로부터 직접 샘플링

    Args:
        x_0: 원본 데이터 [B, C, H, W]
        t: 시간 단계 [B]
        noise: 노이즈 (옵션) [B, C, H, W]

    Returns:
        x_t: 노이즈가 추가된 데이터 [B, C, H, W]
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

def extract(a, t, x_shape):
    """
    시간 t에 해당하는 계수를 추출하고 x_shape에 맞게 broadcast
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
```

### 2.3 Forward Process의 극한 행동

**정리 2.2 (Asymptotic Gaussian)**

적절한 노이즈 스케줄 하에서, T → ∞일 때:

```
lim_{T→∞} q(x_T | x₀) = N(0, I)
```

**증명:**

ᾱ_T = ∏_{t=1}^T (1-β_t)라고 하면:

```
log ᾱ_T = ∑_{t=1}^T log(1-β_t)
         ≈ -∑_{t=1}^T β_t  (Taylor expansion: log(1-x) ≈ -x)
```

β_t가 충분히 작고 ∑_{t=1}^∞ β_t = ∞이면:

```
lim_{T→∞} log ᾱ_T = -∞
∴ lim_{T→∞} ᾱ_T = 0
```

따라서:
```
q(x_T | x₀) = N(x_T; √ᾱ_T x₀, (1-ᾱ_T)I)
            → N(x_T; 0, I)  as T → ∞
```

이는 충분히 많은 스텝 후에는 원본 데이터의 정보가 완전히 사라지고 순수한 Gaussian noise가 됨을 의미합니다.

**실용적 의미:**
- T를 충분히 크게 설정하면 x_T ~ N(0, I)로 근사 가능
- Reverse process의 시작점으로 표준 정규분포 사용 가능
- 일반적으로 T = 1000 정도면 충분히 근사됨

### 2.4 Noise Schedule 설계

Noise schedule β₁, ..., β_T의 선택은 모델 성능에 큰 영향을 미칩니다.

**선형 스케줄 (Linear Schedule):**

DDPM 원논문에서 사용:

```
β_t = β_min + (β_max - β_min) · (t-1)/(T-1)
```

일반적 설정: β_min = 0.0001, β_max = 0.02, T = 1000

```python
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    선형 noise schedule
    """
    return torch.linspace(beta_start, beta_end, timesteps)
```

**코사인 스케줄 (Cosine Schedule):**

Nichol & Dhariwal (2021)에서 제안, 더 완만한 노이즈 추가:

```
ᾱ_t = f(t) / f(0),  where f(t) = cos²((t/T + s)/(1+s) · π/2)
```

s는 작은 offset (일반적으로 0.008)

```python
def cosine_beta_schedule(timesteps, s=0.008):
    """
    코사인 noise schedule (Improved DDPM)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

**스케줄 비교:**

| Schedule | 초반 β_t | 후반 β_t | 장점 | 단점 |
|----------|---------|---------|-----|-----|
| Linear | 작음 | 큼 | 구현 간단 | 후반부에서 급격한 변화 |
| Cosine | 매우 작음 | 완만하게 증가 | 안정적 학습 | 약간 복잡 |

### 2.5 Posterior Distribution q(x_{t-1} | x_t, x₀)

Reverse process를 유도하기 위해 필요한 핵심 분포입니다.

**정리 2.3 (Posterior is Gaussian)**

Forward process의 posterior는 Gaussian입니다:

```
q(x_{t-1} | x_t, x₀) = N(x_{t-1}; μ̃_t(x_t, x₀), β̃_t I)
```

여기서:

```
μ̃_t(x_t, x₀) = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x₀ + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t

β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t
```

**증명:**

Bayes' theorem을 적용:

```
q(x_{t-1} | x_t, x₀) = q(x_t | x_{t-1}, x₀) q(x_{t-1} | x₀) / q(x_t | x₀)
                      = q(x_t | x_{t-1}) q(x_{t-1} | x₀) / q(x_t | x₀)  (Markov property)
```

각 항은:
```
q(x_t | x_{t-1}) = N(x_t; √α_t x_{t-1}, β_t I)
                 ∝ exp(-1/(2β_t) ||x_t - √α_t x_{t-1}||²)

q(x_{t-1} | x₀) = N(x_{t-1}; √ᾱ_{t-1} x₀, (1-ᾱ_{t-1})I)
                ∝ exp(-1/(2(1-ᾱ_{t-1})) ||x_{t-1} - √ᾱ_{t-1} x₀||²)

q(x_t | x₀) = N(x_t; √ᾱ_t x₀, (1-ᾱ_t)I)
            ∝ exp(-1/(2(1-ᾱ_t)) ||x_t - √ᾱ_t x₀||²)
```

Posterior:
```
log q(x_{t-1} | x_t, x₀) = -1/(2β_t) ||x_t - √α_t x_{t-1}||²
                           -1/(2(1-ᾱ_{t-1})) ||x_{t-1} - √ᾱ_{t-1} x₀||²
                           + C

                         = -1/(2β_t) (x_t² - 2√α_t x_t·x_{t-1} + α_t x_{t-1}²)
                           -1/(2(1-ᾱ_{t-1})) (x_{t-1}² - 2√ᾱ_{t-1} x_{t-1}·x₀ + ᾱ_{t-1} x₀²)
                           + C
```

x_{t-1}에 대한 2차 항을 모으면:
```
coefficient of x_{t-1}²: -1/(2β_t) α_t - 1/(2(1-ᾱ_{t-1}))
                        = -1/2 · (α_t(1-ᾱ_{t-1}) + β_t) / (β_t(1-ᾱ_{t-1}))
                        = -1/2 · (α_t - α_t ᾱ_{t-1} + β_t) / (β_t(1-ᾱ_{t-1}))
                        = -1/2 · (α_t - ᾱ_t + β_t) / (β_t(1-ᾱ_{t-1}))
                        = -1/2 · (1 - ᾱ_t) / (β_t(1-ᾱ_{t-1}))
                        = -1/(2β̃_t)

where β̃_t = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)
```

x_{t-1}의 1차 항을 모으면:
```
coefficient of x_{t-1}: √α_t x_t / β_t + √ᾱ_{t-1} x₀ / (1-ᾱ_{t-1})
```

Gaussian의 canonical form ∝ exp(-1/(2σ²)(x - μ)²) = exp(-1/(2σ²)x² + (μ/σ²)x + ...)에서:

```
μ/σ² = √α_t x_t / β_t + √ᾱ_{t-1} x₀ / (1-ᾱ_{t-1})

μ = β̃_t · (√α_t x_t / β_t + √ᾱ_{t-1} x₀ / (1-ᾱ_{t-1}))
  = β̃_t √α_t x_t / β_t + β̃_t √ᾱ_{t-1} x₀ / (1-ᾱ_{t-1})
  = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t · √α_t x_t / β_t + (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t · √ᾱ_{t-1} x₀ / (1-ᾱ_{t-1})
  = √α_t (1-ᾱ_{t-1})/(1-ᾱ_t) x_t + √ᾱ_{t-1} β_t/(1-ᾱ_t) x₀
```

∴ **Posterior mean:**
```
μ̃_t(x_t, x₀) = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x₀ + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t
```

**Posterior variance:**
```
β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t
```

**실용적 의미:**
- x₀와 x_t를 알면 x_{t-1}의 분포를 정확히 계산 가능
- 학습 시 이 posterior를 타겟으로 사용
- Reverse process 설계의 이론적 기초

```python
def posterior_mean_variance(x_0, x_t, t):
    """
    q(x_{t-1} | x_t, x_0)의 평균과 분산 계산

    Args:
        x_0: 원본 데이터
        x_t: 시간 t에서의 노이즈 데이터
        t: 시간 단계

    Returns:
        posterior_mean, posterior_variance
    """
    posterior_mean = (
        extract(sqrt_alphas_cumprod_prev, t, x_t.shape) * extract(betas, t, x_t.shape) /
        extract(one_minus_alphas_cumprod, t, x_t.shape) * x_0 +
        extract(sqrt_alphas, t, x_t.shape) * extract(one_minus_alphas_cumprod_prev, t, x_t.shape) /
        extract(one_minus_alphas_cumprod, t, x_t.shape) * x_t
    )

    posterior_variance = (
        extract(one_minus_alphas_cumprod_prev, t, x_t.shape) /
        extract(one_minus_alphas_cumprod, t, x_t.shape) *
        extract(betas, t, x_t.shape)
    )

    return posterior_mean, posterior_variance
```

### 2.6 Forward Process의 정보 이론적 해석

**Information Loss:**

Forward process는 점진적으로 정보를 손실하는 과정으로 볼 수 있습니다.

Mutual information I(x₀; x_t)를 고려하면:

```
I(x₀; x_t) = H(x_t) - H(x_t | x₀)
           = H(x_t) - d/2 · log(2πe(1-ᾱ_t))
```

여기서 d는 데이터 차원입니다.

t가 증가하면:
- ᾱ_t → 0
- (1-ᾱ_t) → 1
- H(x_t | x₀) → d/2 · log(2πe)  (최대 엔트로피)
- I(x₀; x_t) → 0

즉, 시간이 지남에 따라 x₀에 대한 정보가 점진적으로 손실됩니다.

---

## 3. Reverse Diffusion Process와 Score Function

### 3.1 Reverse Process의 정의

Forward process가 데이터에 노이즈를 추가했다면, reverse process는 이를 역으로 제거하여 데이터를 생성합니다.

**정의 3.1 (Reverse Process)**

Reverse process는 노이즈 x_T ~ N(0, I)로부터 시작하여 데이터를 생성하는 Markov chain입니다:

```
p_θ(x_{0:T}) = p(x_T) ∏_{t=1}^T p_θ(x_{t-1} | x_t)
```

여기서:
```
p(x_T) = N(x_T; 0, I)
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**핵심 질문:** μ_θ와 Σ_θ를 어떻게 모델링할 것인가?

### 3.2 Reverse Process도 Gaussian이다

놀랍게도, 적절한 조건 하에서 reverse process도 Gaussian입니다.

**정리 3.1 (Reverse Process is Gaussian for small β_t)**

β_t가 충분히 작으면, reverse process conditional p(x_{t-1} | x_t)는 다음과 근사됩니다:

```
p(x_{t-1} | x_t) ≈ N(x_{t-1}; μ̃(x_t), β̃_t I)
```

여기서 μ̃와 β̃_t는 forward posterior의 평균과 분산과 동일한 형태입니다.

**직관적 설명:**

1. β_t가 작으면 한 스텝의 변화가 미소합니다
2. Forward와 reverse는 시간 방향만 다를 뿐 본질적으로 동일한 Markov process
3. 국소적으로는 Gaussian approximation이 정확함

**Feller (1949)의 Diffusion Theory:**

이산 시간 Markov chain의 연속 극한에서 forward/reverse process는 모두 diffusion process가 되며, transition kernel이 Gaussian입니다.

### 3.3 Score Function의 등장

Reverse process를 실제로 구현하려면 **score function**이 핵심입니다.

**정의 3.2 (Score Function)**

확률 분포 p(x)의 score function은:

```
s(x) = ∇_x log p(x)
```

즉, 로그 확률의 gradient입니다.

**직관적 의미:**
- Score는 확률 밀도가 증가하는 방향을 가리킴
- 데이터 manifold를 향하는 "gradient field"
- 높은 확률 영역으로 이동하는 방향을 알려줌

**Reverse Process와의 연결:**

Reverse conditional의 평균은 score function으로 표현 가능합니다.

**정리 3.2 (Reverse Process via Score)**

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)
```

에서, optimal μ_θ는:

```
μ_θ(x_t, t) = 1/√α_t (x_t + β_t · s_θ(x_t, t))
```

여기서 s_θ(x_t, t) = ∇_{x_t} log p(x_t)는 marginal distribution의 score입니다.

**증명 (핵심 아이디어):**

Bayes' rule과 Tweedie's formula를 사용합니다.

Forward posterior의 평균은:
```
μ̃_t(x_t, x₀) = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x₀ + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t
```

x₀를 x_t의 함수로 표현하면:
```
x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε

∴ x₀ = (x_t - √(1-ᾱ_t) ε) / √ᾱ_t
```

Tweedie's formula (지수족 분포의 성질):
```
E[x₀ | x_t] = x_t/√ᾱ_t + (1-ᾱ_t)/√ᾱ_t · ∇_{x_t} log p(x_t)
            = 1/√ᾱ_t (x_t + (1-ᾱ_t) s(x_t, t))
```

이를 대입하면:
```
μ̃_t ≈ (√ᾱ_{t-1} β_t)/(1-ᾱ_t) · 1/√ᾱ_t (x_t + (1-ᾱ_t) s(x_t, t)) + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t

    = 1/√α_t (x_t + β_t s(x_t, t))
```

따라서 **score function s_θ(x_t, t)를 추정하면 reverse process를 구현할 수 있습니다!**

### 3.4 Score Estimation as Denoising

Score function을 직접 추정하는 대신, **denoising** 문제로 변환할 수 있습니다.

**정리 3.3 (Score via Denoising)**

Forward process에서:
```
x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε,  ε ~ N(0, I)
```

이면:
```
∇_{x_t} log p(x_t) = -ε / √(1-ᾱ_t) = -(x_t - √ᾱ_t x₀) / (1-ᾱ_t)
```

**증명:**

p(x_t)의 score를 계산하기 위해 marginalization:

```
p(x_t) = ∫ p(x_t | x₀) p_data(x₀) dx₀
```

Log를 취하고 gradient:
```
∇_{x_t} log p(x_t) = ∇_{x_t} log ∫ p(x_t | x₀) p_data(x₀) dx₀

                    = (∫ ∇_{x_t} p(x_t | x₀) p_data(x₀) dx₀) / p(x_t)

                    = ∫ (∇_{x_t} p(x_t | x₀) / p(x_t | x₀)) · (p(x_t | x₀) p_data(x₀) / p(x_t)) dx₀

                    = ∫ ∇_{x_t} log p(x_t | x₀) · p(x₀ | x_t) dx₀

                    = E_{x₀ ~ p(x₀|x_t)} [∇_{x_t} log p(x_t | x₀)]
```

p(x_t | x₀) = N(x_t; √ᾱ_t x₀, (1-ᾱ_t)I)이므로:

```
log p(x_t | x₀) = -1/(2(1-ᾱ_t)) ||x_t - √ᾱ_t x₀||² + const

∇_{x_t} log p(x_t | x₀) = -1/(1-ᾱ_t) (x_t - √ᾱ_t x₀)
                         = -(x_t - √ᾱ_t x₀) / (1-ᾱ_t)
                         = -ε / √(1-ᾱ_t)
```

**실용적 의미:**

Score function을 추정하는 대신 **noise predictor ε_θ(x_t, t)**를 학습하면 됩니다!

```
s_θ(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)
```

### 3.5 Langevin Dynamics

Score function을 이용한 샘플링 방법입니다.

**정의 3.3 (Langevin Dynamics)**

주어진 score function s(x) = ∇_x log p(x)에 대해, 다음 반복을 수행:

```
x_{i+1} = x_i + δ · s(x_i) + √(2δ) · z_i,  z_i ~ N(0, I)
```

δ → 0이면, 이 과정은 p(x)로부터의 샘플을 생성합니다.

**이론적 근거 (Langevin MCMC):**

다음 SDE의 해는 정상 분포로 p(x)를 가집니다:

```
dx = ∇_x log p(x) dt + √2 dW
```

여기서 W는 Wiener process (Brownian motion)입니다.

**Diffusion Models에서의 활용:**

Reverse process의 각 스텝이 Langevin dynamics의 한 스텝으로 볼 수 있습니다:

```
x_{t-1} = x_t + β_t · s_θ(x_t, t) + √β_t · z,  z ~ N(0, I)
        = 1/√α_t (x_t + β_t · s_θ(x_t, t)) + √β̃_t · z  (reparameterization)
```

---

## 4. Score Matching Theory의 완전한 증명

### 4.1 Score Matching의 동기

Score function s(x) = ∇_x log p(x)를 직접 추정하려면 normalizing constant가 필요합니다:

```
p(x) = exp(f(x)) / Z,  where Z = ∫ exp(f(x)) dx
```

Z를 계산하는 것은 고차원에서 불가능합니다.

**핵심 아이디어:** Score는 Z에 무관합니다!

```
∇_x log p(x) = ∇_x f(x) - ∇_x log Z = ∇_x f(x)
```

따라서 normalizing constant 없이 score를 추정할 수 있습니다.

### 4.2 Fisher Divergence

Score matching의 목적 함수는 Fisher divergence입니다.

**정의 4.1 (Fisher Divergence)**

두 분포 p(x)와 q_θ(x)의 Fisher divergence는:

```
D_F(p || q_θ) = 1/2 E_{x~p} [||∇_x log p(x) - ∇_x log q_θ(x)||²]
              = 1/2 E_{x~p} [||s(x) - s_θ(x)||²]
```

**Score Matching 목적 함수:**

```
J_SM(θ) = 1/2 E_{x~p_data} [||∇_x log p_data(x) - s_θ(x)||²]
```

**문제점:** p_data의 score를 알 수 없습니다!

### 4.3 Denoising Score Matching

**핵심 아이디어:** 데이터에 노이즈를 추가하면 score를 계산할 수 있습니다!

**정리 4.1 (Denoising Score Matching, Vincent 2011)**

노이즈를 추가한 분포를:
```
q_σ(x̃ | x) = N(x̃; x, σ²I)
q_σ(x̃) = ∫ q_σ(x̃ | x) p_data(x) dx
```

라고 하면, 다음 목적 함수는 동등합니다:

```
J_DSM(θ) = 1/2 E_{x~p_data} E_{x̃~q_σ(·|x)} [||∇_{x̃} log q_σ(x̃ | x) - s_θ(x̃)||²]
         ∝ 1/2 E_{x~p_data} E_{x̃~q_σ(·|x)} [||s_θ(x̃) - ∇_{x̃} log q_σ(x̃)||²] + const
```

**증명:**

q_σ(x̃ | x) = N(x̃; x, σ²I)이므로:

```
log q_σ(x̃ | x) = -1/(2σ²) ||x̃ - x||² + const

∇_{x̃} log q_σ(x̃ | x) = -1/σ² (x̃ - x) = -(x̃ - x)/σ²
```

x̃ = x + σε, ε ~ N(0, I)로 재매개변수화하면:

```
∇_{x̃} log q_σ(x̃ | x) = -ε/σ
```

따라서:
```
J_DSM(θ) = 1/2 E_x E_ε [||s_θ(x + σε) + ε/σ||²]
         = 1/2 E_x E_ε [||σ·s_θ(x + σε) + ε||²]
```

**Diffusion Models에서의 적용:**

Diffusion models는 여러 noise level σ_t = √(1-ᾱ_t)에 대해 denoising score matching을 수행합니다!

### 4.4 Multi-scale Denoising Score Matching

**정의 4.2 (Noise Conditional Score Networks)**

여러 noise level {σ_t}_{t=1}^T에 대해:

```
J_MDSM(θ) = E_t E_{x~p_data} E_{x_t~N(x, σ_t²I)} [λ(t) ||s_θ(x_t, t) - ∇_{x_t} log p(x_t | x)||²]
```

여기서 λ(t)는 weighting function입니다.

**Diffusion Models의 목적 함수:**

Forward process에서 x_t = √ᾱ_t x + √(1-ᾱ_t) ε이므로:

```
∇_{x_t} log p(x_t | x) = -ε / √(1-ᾱ_t)
```

따라서:
```
J_DSM(θ) = E_t E_{x~p_data} E_{ε~N(0,I)} [λ(t) ||s_θ(√ᾱ_t x + √(1-ᾱ_t) ε, t) + ε/√(1-ᾱ_t)||²]
```

Noise predictor ε_θ를 사용하면:

```
s_θ(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)
```

목적 함수는:
```
L_simple(θ) = E_t E_{x~p_data} E_{ε~N(0,I)} [||ε - ε_θ(√ᾱ_t x + √(1-ᾱ_t) ε, t)||²]
```

이것이 DDPM의 단순화된 목적 함수입니다!

### 4.5 Variational Lower Bound 유도

Score matching 외에도 variational inference 관점에서 유도할 수 있습니다.

**정리 4.2 (Variational Lower Bound for Diffusion)**

```
log p_θ(x₀) ≥ E_q [log p_θ(x₀ | x₁)] - ∑_{t=2}^T D_KL(q(x_{t-1}|x_t,x₀) || p_θ(x_{t-1}|x_t)) - D_KL(q(x_T|x₀) || p(x_T))
```

**증명:**

Jensen's inequality를 적용:

```
log p_θ(x₀) = log ∫ p_θ(x_{0:T}) dx_{1:T}
            = log ∫ p_θ(x_{0:T}) · q(x_{1:T}|x₀)/q(x_{1:T}|x₀) dx_{1:T}
            = log E_q [p_θ(x_{0:T}) / q(x_{1:T}|x₀)]
            ≥ E_q [log p_θ(x_{0:T}) / q(x_{1:T}|x₀)]
            = E_q [log p_θ(x_{0:T}) - log q(x_{1:T}|x₀)]
```

전개하면:
```
= E_q [log p(x_T) + ∑_{t=1}^T log p_θ(x_{t-1}|x_t) - ∑_{t=1}^T log q(x_t|x_{t-1})]

= E_q [log p(x_T) - log q(x_T|x₀)]
  + E_q [∑_{t=2}^T log p_θ(x_{t-1}|x_t) - log q(x_{t-1}|x_t,x₀)]
  + E_q [log p_θ(x₀|x₁)]

= -D_KL(q(x_T|x₀) || p(x_T))
  - ∑_{t=2}^T E_q [D_KL(q(x_{t-1}|x_t,x₀) || p_θ(x_{t-1}|x_t))]
  + E_q [log p_θ(x₀|x₁)]
```

**각 항의 의미:**

1. **Reconstruction term**: E_q [log p_θ(x₀ | x₁)]
   - VAE의 reconstruction loss와 유사

2. **Denoising matching terms**: ∑_{t=2}^T D_KL(q(x_{t-1}|x_t,x₀) || p_θ(x_{t-1}|x_t))
   - Forward posterior와 reverse process의 매칭
   - 두 Gaussian의 KL divergence로 계산 가능

3. **Prior matching**: D_KL(q(x_T|x₀) || p(x_T))
   - x_T가 표준 정규분포에 가까울수록 작음
   - T가 충분히 크면 무시 가능

### 4.6 KL Divergence의 명시적 계산

두 Gaussian의 KL divergence:

```
q(x_{t-1}|x_t,x₀) = N(x_{t-1}; μ̃_t, β̃_t I)
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ, σ_t² I)
```

에 대해:

```
D_KL(q || p_θ) = 1/(2σ_t²) ||μ̃_t - μ_θ||² + d/2 (β̃_t/σ_t² - 1 - log(β̃_t/σ_t²))
```

σ_t² = β̃_t로 설정하면 (variance를 고정):

```
D_KL(q || p_θ) = 1/(2β̃_t) ||μ̃_t - μ_θ||²
```

μ̃_t와 μ_θ를 ε-prediction으로 표현:

```
μ̃_t = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε)
μ_θ = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t))
```

따라서:
```
||μ̃_t - μ_θ||² = (β_t² / (α_t(1-ᾱ_t))) ||ε - ε_θ(x_t, t)||²
```

결국:
```
D_KL(q || p_θ) = (β_t² / (2α_t β̃_t(1-ᾱ_t))) ||ε - ε_θ(x_t, t)||²
```

**DDPM의 단순화:**

Ho et al.은 복잡한 weighting을 무시하고:

```
L_simple(θ) = E_t E_{x₀,ε} [||ε - ε_θ(x_t, t)||²]
```

를 사용했고, 이것이 더 좋은 성능을 보였습니다!

---

## 5. DDPM의 수학적 기초

### 5.1 DDPM의 알고리즘

**Denoising Diffusion Probabilistic Models (DDPM)**은 Ho et al. (2020)이 제안한 모델로, diffusion models를 실용화했습니다.

**핵심 기여:**

1. **단순화된 학습 목적 함수**: 복잡한 weighting 제거
2. **U-Net 기반 아키텍처**: 시간 조건부 노이즈 예측
3. **안정적 학습**: VAE와 유사한 수준의 안정성

**DDPM 학습 알고리즘:**

```
Algorithm 1: DDPM Training

1. repeat
2.   x₀ ~ q(x₀)                         // 데이터 샘플링
3.   t ~ Uniform({1, ..., T})          // 시간 단계 무작위 선택
4.   ε ~ N(0, I)                       // 노이즈 샘플링
5.   Take gradient descent step on:
     ∇_θ ||ε - ε_θ(√ᾱ_t x₀ + √(1-ᾱ_t) ε, t)||²
6. until converged
```

**DDPM 샘플링 알고리즘:**

```
Algorithm 2: DDPM Sampling

1. x_T ~ N(0, I)
2. for t = T, ..., 1 do
3.   z ~ N(0, I) if t > 1, else z = 0
4.   x_{t-1} = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t, t)) + √β_t z
5. end for
6. return x₀
```

### 5.2 U-Net 아키텍처

DDPM은 **U-Net**을 noise predictor ε_θ(x_t, t)로 사용합니다.

**구조:**

```
Input: x_t (noisy image) + t (time embedding)
       ↓
    [Encoder]
       ↓
  ResNet blocks with time embedding
  + Downsampling
       ↓
    [Bottleneck]
       ↓
  Attention layers
       ↓
    [Decoder]
       ↓
  ResNet blocks with time embedding
  + Upsampling
  + Skip connections from encoder
       ↓
    Output: ε_θ(x_t, t)
```

**시간 임베딩 (Sinusoidal Position Encoding):**

```python
def timestep_embedding(timesteps, dim):
    """
    Sinusoidal timestep embeddings (Transformer와 동일)

    Args:
        timesteps: [B] 시간 단계
        dim: 임베딩 차원

    Returns:
        [B, dim] 임베딩
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb
```

**ResNet Block with Time Embedding:**

```python
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Time embedding projection
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, time_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.act(self.time_emb_proj(time_emb))
        h = h + time_emb[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.shortcut(x)
```

### 5.3 DDPM의 완전한 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """코사인 noise schedule"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model  # U-Net ε_θ
        self.timesteps = timesteps

        # Noise schedule
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))

        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: x_0 -> x_t"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        """학습 loss 계산"""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)

        # Simple MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def forward(self, x, *args, **kwargs):
        """학습 시 호출"""
        b = x.shape[0]
        device = x.device
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

    @torch.no_grad()
    def p_sample(self, x, t):
        """Reverse diffusion: x_t -> x_{t-1}"""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # ε_θ(x_t, t) 예측
        predicted_noise = self.model(x, t)

        # 평균 계산: μ_θ = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape):
        """x_T ~ N(0,I)로부터 샘플링"""
        device = next(self.model.parameters()).device
        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.timesteps)):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long))

        return x

def extract(a, t, x_shape):
    """시간 t에 해당하는 계수를 추출하고 broadcast"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
```

### 5.4 Loss Weighting의 효과

DDPM 논문은 복잡한 weighting을 제거한 **L_simple**을 사용했습니다:

```
L_simple = E_t E_{x₀,ε} [||ε - ε_θ(x_t, t)||²]
```

하지만 이론적으로는 다음 weighting이 더 정확합니다:

```
L_vlb = E_t E_{x₀,ε} [λ(t) ||ε - ε_θ(x_t, t)||²]

where λ(t) = (1-ᾱ_t) / (2α_t β̃_t √(1-ᾱ_t))
```

**실험 결과 (Ho et al.):**

- **L_simple**: FID (Fréchet Inception Distance)가 더 좋음
- **L_vlb**: Negative log-likelihood가 더 좋음 (더 정확한 density model)

**직관적 설명:**

- L_simple은 모든 시간 단계를 동등하게 취급
- 이는 샘플 품질에 더 유리 (perceptual quality 중시)
- L_vlb는 likelihood 최적화에 유리 (확률적 정확성 중시)

### 5.5 Improved DDPM (Nichol & Dhariwal 2021)

DDPM의 개선 사항들:

**1. Learned Variance:**

원래 DDPM은 variance를 고정했지만, 학습 가능하게 변경:

```python
def p_sample_learned_variance(self, x, t):
    """Learned variance를 사용한 샘플링"""
    # 모델이 평균과 variance를 모두 예측
    model_output = self.model(x, t)

    # 채널을 절반으로 나눠서 평균과 variance로 사용
    model_mean, model_var_values = torch.split(model_output, x.shape[1], dim=1)

    # Variance interpolation: v ∈ [0,1]로 β_t와 β̃_t 사이를 보간
    min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
    max_log = extract(torch.log(self.betas), t, x.shape)
    frac = (model_var_values + 1) / 2  # [-1, 1] -> [0, 1]
    model_log_variance = frac * max_log + (1 - frac) * min_log
    model_variance = torch.exp(model_log_variance)

    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(model_variance) * noise
```

**2. Cosine Noise Schedule:**

선형 스케줄 대신 cosine 스케줄 사용 (더 안정적)

**3. Hybrid Loss:**

```
L_hybrid = L_simple + λ · L_vlb
```

**4. Importance Sampling:**

시간 단계 t를 균등하게 샘플링하지 않고, loss가 큰 시간대에 집중

**결과:**
- ImageNet 64x64에서 FID 2.07 달성 (당시 SOTA)
- GAN과 경쟁 가능한 수준

---

## 6. DDIM과 Deterministic Sampling

### 6.1 DDIM의 동기

**DDPM의 한계:**

- 샘플링에 1000 스텝 필요 → 매우 느림
- Stochastic process → 동일한 latent에서 다른 샘플 생성
- 중간 보간이 어려움

**DDIM의 아이디어 (Song et al. 2021):**

- **Non-Markovian** forward process 정의
- **Deterministic** sampling 가능
- **Accelerated** sampling (10-100 스텝으로 감소)

### 6.2 DDIM의 수학적 유도

**핵심 통찰:** Forward process는 marginal q(x_t | x₀)만 일치하면 됩니다!

DDPM은 Markovian이었지만:
```
q(x_t | x_{t-1}, x₀) = q(x_t | x_{t-1})  (Markov)
```

DDIM은 non-Markovian:
```
q(x_t | x_{t-1}, x₀) ≠ q(x_t | x_{t-1})
```

하지만 marginal은 동일:
```
q(x_t | x₀) = N(x_t; √ᾱ_t x₀, (1-ᾱ_t)I)  (DDPM과 동일)
```

**정리 6.1 (DDIM Forward Process)**

다음 forward process를 정의:

```
q_σ(x_{t-1} | x_t, x₀) = N(x_{t-1}; √ᾱ_{t-1} x₀ + √(1-ᾱ_{t-1}-σ_t²) · (x_t - √ᾱ_t x₀)/√(1-ᾱ_t), σ_t² I)
```

여기서 σ_t는 조절 가능한 noise 파라미터입니다.

**특수 케이스:**

1. σ_t² = β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t → **DDPM** (stochastic)
2. σ_t = 0 → **DDIM** (deterministic)

**증명 (Marginal 불변성):**

x_{t-1}을 재매개변수화:
```
x_{t-1} = √ᾱ_{t-1} x₀ + √(1-ᾱ_{t-1}-σ_t²) · (x_t - √ᾱ_t x₀)/√(1-ᾱ_t) + σ_t ε
```

x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε'를 대입하면, marginalization over ε'에 의해:

```
q(x_{t-1} | x₀) = N(x_{t-1}; √ᾱ_{t-1} x₀, (1-ᾱ_{t-1})I)  ✓
```

### 6.3 DDIM Reverse Process

x₀를 ε_θ로 예측:

```
x₀ ≈ (x_t - √(1-ᾱ_t) ε_θ(x_t, t)) / √ᾱ_t
```

이를 forward process 식에 대입:

```
x_{t-1} = √ᾱ_{t-1} · (x_t - √(1-ᾱ_t) ε_θ(x_t, t)) / √ᾱ_t
        + √(1-ᾱ_{t-1}-σ_t²) · ε_θ(x_t, t)
        + σ_t ε
```

정리하면:

```
x_{t-1} = √ᾱ_{t-1}/√ᾱ_t · x_t
        + (√(1-ᾱ_{t-1}-σ_t²) - √ᾱ_{t-1}/√ᾱ_t · √(1-ᾱ_t)) · ε_θ(x_t, t)
        + σ_t ε
```

**Deterministic case (σ_t = 0):**

```
x_{t-1} = √ᾱ_{t-1}/√ᾱ_t · x_t + (√(1-ᾱ_{t-1}) - √ᾱ_{t-1}/√ᾱ_t · √(1-ᾱ_t)) · ε_θ(x_t, t)
```

이것은 **ODE**입니다! (노이즈 항이 없음)

### 6.4 DDIM 샘플링 알고리즘

```python
@torch.no_grad()
def ddim_sample(self, x, t, t_next, eta=0.0):
    """
    DDIM 샘플링: x_t -> x_{t_next}

    Args:
        x: x_t
        t: 현재 시간
        t_next: 다음 시간 (t_next < t)
        eta: stochasticity 조절 (0 = deterministic, 1 = DDPM)

    Returns:
        x_{t_next}
    """
    # 현재 노이즈 예측
    predicted_noise = self.model(x, t)

    # 현재 α, ᾱ 값
    alpha_t = extract(self.alphas, t, x.shape)
    alpha_bar_t = extract(self.alphas_cumprod, t, x.shape)
    alpha_bar_t_next = extract(self.alphas_cumprod, t_next, x.shape) if t_next.max() >= 0 else torch.ones_like(alpha_bar_t)

    # x₀ 예측
    pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

    # Variance 계산
    sigma_t = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_next)

    # Direction pointing to x_t
    dir_xt = torch.sqrt(1 - alpha_bar_t_next - sigma_t**2) * predicted_noise

    # DDIM 업데이트
    x_next = torch.sqrt(alpha_bar_t_next) * pred_x0 + dir_xt

    if eta > 0:
        noise = torch.randn_like(x)
        x_next = x_next + sigma_t * noise

    return x_next

@torch.no_grad()
def ddim_sample_loop(self, shape, ddim_timesteps=50, eta=0.0):
    """
    Accelerated DDIM 샘플링

    Args:
        shape: 출력 shape
        ddim_timesteps: 사용할 시간 단계 수
        eta: stochasticity

    Returns:
        생성된 샘플
    """
    device = next(self.model.parameters()).device
    b = shape[0]

    # Subset of timesteps
    step = self.timesteps // ddim_timesteps
    timesteps = list(range(0, self.timesteps, step))
    timesteps = list(reversed(timesteps))

    x = torch.randn(shape, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        t_next_batch = torch.full((b,), timesteps[i+1], device=device, dtype=torch.long) if i < len(timesteps) - 1 else torch.full((b,), -1, device=device, dtype=torch.long)

        x = self.ddim_sample(x, t_batch, t_next_batch, eta=eta)

    return x
```

### 6.5 DDIM의 장점

**1. Fast Sampling:**

- DDPM: 1000 스텝
- DDIM: 10-100 스텝
- **10-100배 빠름!**

**2. Deterministic Mapping:**

동일한 x_T에서 항상 동일한 x₀ 생성

```python
# 동일한 latent로 재생성
x_T = torch.randn(1, 3, 64, 64)
x_0_1 = ddim_sample_loop(x_T, eta=0.0)
x_0_2 = ddim_sample_loop(x_T, eta=0.0)
# x_0_1 == x_0_2  (deterministic)
```

**3. Interpolation in Latent Space:**

```python
# 두 이미지의 latent를 보간
x_T_1 = encode_to_latent(img1)  # x₀ -> x_T (DDIM inversion)
x_T_2 = encode_to_latent(img2)

# Spherical interpolation
x_T_interp = slerp(x_T_1, x_T_2, alpha=0.5)

# Decode
img_interp = ddim_sample_loop(x_T_interp, eta=0.0)
```

**4. Semantic Editing:**

Latent space에서 방향 찾기:

```python
# "smile" 방향 찾기
direction = avg_latent(smiling_imgs) - avg_latent(neutral_imgs)

# 적용
x_T = encode_to_latent(img)
x_T_smile = x_T + alpha * direction
img_smile = ddim_sample_loop(x_T_smile)
```

### 6.6 DDIM Inversion

DDIM의 deterministic 성질을 이용해 **역방향 인코딩**이 가능합니다.

```python
@torch.no_grad()
def ddim_invert(self, x, ddim_timesteps=50):
    """
    DDIM Inversion: x₀ -> x_T

    Args:
        x: x₀ (원본 이미지)
        ddim_timesteps: 시간 단계 수

    Returns:
        x_T (latent code)
    """
    device = x.device
    b = x.shape[0]

    # Forward direction
    step = self.timesteps // ddim_timesteps
    timesteps = list(range(0, self.timesteps, step))

    for i, t in enumerate(timesteps[:-1]):
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        t_next_batch = torch.full((b,), timesteps[i+1], device=device, dtype=torch.long)

        # DDIM forward step (deterministic)
        predicted_noise = self.model(x, t_batch)
        alpha_bar_t = extract(self.alphas_cumprod, t_batch, x.shape)
        alpha_bar_t_next = extract(self.alphas_cumprod, t_next_batch, x.shape)

        # x₀ 예측
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        # x_{t+1} 계산 (forward direction, eta=0)
        dir_xt = torch.sqrt(1 - alpha_bar_t_next) * predicted_noise
        x = torch.sqrt(alpha_bar_t_next) * pred_x0 + dir_xt

    return x
```

**활용 예:**

```python
# Real image editing
x_0 = real_image
x_T = ddim_invert(x_0)  # Encode to latent

# Edit in latent space
x_T_edited = x_T + some_direction

# Decode back
x_0_edited = ddim_sample_loop(x_T_edited, eta=0.0)
```

### 6.7 η 파라미터의 효과

η ∈ [0, 1]은 stochasticity를 조절합니다:

- **η = 0**: Deterministic (DDIM)
  - 빠른 샘플링
  - 재현 가능
  - 약간 낮은 다양성

- **η = 1**: Stochastic (DDPM)
  - 느린 샘플링
  - 높은 다양성
  - 더 나은 샘플 품질 (경우에 따라)

- **η ∈ (0, 1)**: Hybrid
  - 속도와 품질의 균형

**실험 결과:**

| η | Steps | FID | Time |
|---|-------|-----|------|
| 0 | 50 | 5.2 | 5s |
| 0.5 | 50 | 4.8 | 5s |
| 1.0 | 50 | 6.1 | 5s |
| 1.0 | 1000 | 3.2 | 100s |

---

## 7. 고급 기법들

### 7.1 Classifier Guidance

**동기:** Conditional generation의 품질 향상

**정의 7.1 (Classifier Guidance)**

Classifier p_φ(y|x_t)를 이용하여 조건부 샘플링:

```
∇_{x_t} log p(x_t | y) = ∇_{x_t} log p(x_t) + ∇_{x_t} log p_φ(y | x_t)
                        = s_θ(x_t, t) + ∇_{x_t} log p_φ(y | x_t)
```

Guidance scale w를 도입:

```
s̃(x_t, y) = s_θ(x_t, t) + w · ∇_{x_t} log p_φ(y | x_t)
```

**유도:**

Bayes' rule:
```
p(x_t | y) = p(y | x_t) p(x_t) / p(y)

log p(x_t | y) = log p(y | x_t) + log p(x_t) - log p(y)

∇_{x_t} log p(x_t | y) = ∇_{x_t} log p(y | x_t) + ∇_{x_t} log p(x_t)
```

**구현:**

```python
@torch.no_grad()
def sample_with_classifier_guidance(self, y, classifier, guidance_scale=1.0):
    """
    Classifier guidance를 사용한 조건부 샘플링

    Args:
        y: 클래스 레이블
        classifier: 학습된 classifier p_φ(y|x_t)
        guidance_scale: guidance 강도 w

    Returns:
        조건부 샘플
    """
    device = next(self.model.parameters()).device
    shape = (y.shape[0], 3, 64, 64)
    x = torch.randn(shape, device=device)

    for i in reversed(range(self.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        # Unconditional score
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            predicted_noise = self.model(x_in, t)

            # Classifier gradient
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            classifier_grad = torch.autograd.grad(selected.sum(), x_in)[0]

        # Guided score
        guided_noise = predicted_noise - guidance_scale * torch.sqrt(1 - self.alphas_cumprod[i]) * classifier_grad

        # Sampling step (DDPM)
        x = self.p_sample_guided(x, t, guided_noise)

    return x
```

**문제점:**

- Noisy image에 대한 classifier 학습 필요
- 각 노이즈 레벨에서 classifier 성능이 달라짐

### 7.2 Classifier-Free Guidance

**핵심 아이디어:** Classifier 없이 guidance!

**정의 7.2 (Classifier-Free Guidance, Ho & Salimans 2022)**

Conditional과 unconditional score를 모두 학습:

```
s̃_θ(x_t, y) = s_θ(x_t, ∅) + w · (s_θ(x_t, y) - s_θ(x_t, ∅))
             = (1 - w) · s_θ(x_t, ∅) + w · s_θ(x_t, y)
```

여기서:
- s_θ(x_t, y): Conditional score
- s_θ(x_t, ∅): Unconditional score
- w: Guidance scale (일반적으로 w > 1)

**학습 방법:**

```python
def train_classifier_free(self, x, y, p_uncond=0.1):
    """
    Classifier-free guidance를 위한 학습

    Args:
        x: 이미지 배치
        y: 조건 (클래스, 텍스트 등)
        p_uncond: Unconditional로 학습할 확률

    Returns:
        loss
    """
    b = x.shape[0]
    t = torch.randint(0, self.timesteps, (b,), device=x.device).long()
    noise = torch.randn_like(x)

    # 일정 확률로 조건을 null로 대체
    mask = torch.rand(b, device=x.device) < p_uncond
    y_input = y.clone()
    y_input[mask] = self.null_token  # null token (∅)

    x_noisy = self.q_sample(x, t, noise)
    predicted_noise = self.model(x_noisy, t, y_input)

    loss = F.mse_loss(predicted_noise, noise)
    return loss
```

**샘플링:**

```python
@torch.no_grad()
def sample_classifier_free(self, y, guidance_scale=7.5):
    """
    Classifier-free guidance 샘플링

    Args:
        y: 조건
        guidance_scale: w (일반적으로 7.5)

    Returns:
        조건부 샘플
    """
    device = next(self.model.parameters()).device
    shape = (y.shape[0], 3, 64, 64)
    x = torch.randn(shape, device=device)

    for i in reversed(range(self.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        # Conditional과 unconditional을 동시에 계산
        x_input = torch.cat([x, x], dim=0)
        t_input = torch.cat([t, t], dim=0)
        y_input = torch.cat([y, torch.full_like(y, self.null_token)], dim=0)

        noise_pred = self.model(x_input, t_input, y_input)
        noise_cond, noise_uncond = noise_pred.chunk(2, dim=0)

        # Classifier-free guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Sampling step
        x = self.p_sample_with_noise(x, t, noise_pred)

    return x
```

**장점:**

- Classifier 학습 불필요
- 더 나은 샘플 품질
- Text-to-image (DALL-E 2, Stable Diffusion) 등에 필수

**Guidance Scale의 효과:**

| w | 품질 | 다양성 | 용도 |
|---|------|-------|-----|
| 1.0 | 중간 | 높음 | 다양한 샘플 |
| 3.0-5.0 | 높음 | 중간 | 균형잡힌 생성 |
| 7.5-10.0 | 매우 높음 | 낮음 | 정확한 조건 반영 (Stable Diffusion 기본값) |
| >15.0 | 과포화 | 매우 낮음 | 비권장 |

### 7.3 Latent Diffusion Models (Stable Diffusion)

**동기:** 고해상도 이미지는 계산 비용이 매우 큼!

**핵심 아이디어:** Pixel space 대신 **latent space**에서 diffusion!

**구조:**

```
Real Image x ∈ ℝ^(H×W×3)
      ↓ Encoder E
Latent z ∈ ℝ^(h×w×c)  (h,w << H,W)
      ↓ Diffusion Process
Generated Latent z'
      ↓ Decoder D
Generated Image x' ∈ ℝ^(H×W×3)
```

**정의 7.3 (Latent Diffusion Model, Rombach et al. 2022)**

1. **Autoencoder 학습:**
   ```
   Encoder: E: ℝ^(H×W×3) → ℝ^(h×w×c)
   Decoder: D: ℝ^(h×w×c) → ℝ^(H×W×3)

   Loss: L_AE = ||x - D(E(x))||² + KL regularization
   ```

2. **Latent space에서 diffusion:**
   ```
   z = E(x)
   z_t = √ᾱ_t z + √(1-ᾱ_t) ε

   Loss: L_LDM = E_{z,ε,t} [||ε - ε_θ(z_t, t, c)||²]
   ```

   여기서 c는 conditioning (text, class 등)

**구현:**

```python
class LatentDiffusion(nn.Module):
    def __init__(self, autoencoder, unet, timesteps=1000):
        super().__init__()
        self.autoencoder = autoencoder  # VAE
        self.unet = unet  # U-Net in latent space
        self.ddpm = DDPM(unet, timesteps)

    def encode(self, x):
        """이미지를 latent로 인코딩"""
        with torch.no_grad():
            z = self.autoencoder.encode(x).sample()  # E(x)
        return z

    def decode(self, z):
        """Latent를 이미지로 디코딩"""
        with torch.no_grad():
            x = self.autoencoder.decode(z)  # D(z)
        return x

    def forward(self, x, condition=None):
        """학습"""
        # Encode to latent
        z = self.encode(x)

        # Diffusion in latent space
        loss = self.ddpm(z, condition)
        return loss

    @torch.no_grad()
    def sample(self, condition, num_samples=1):
        """조건부 샘플링"""
        # Sample in latent space
        z = self.ddpm.sample((num_samples, 4, 64, 64), condition)

        # Decode to image
        x = self.decode(z)
        return x
```

**장점:**

1. **계산 효율성:**
   - 512×512 이미지 → 64×64 latent (8배 downsampling)
   - 64배 적은 연산량!

2. **메모리 효율성:**
   - 더 큰 배치 크기
   - 더 깊은 U-Net 사용 가능

3. **더 나은 학습:**
   - Perceptual loss 효과 (autoencoder가 이미 학습됨)
   - 더 적은 diffusion 스텝 필요

**Stable Diffusion Architecture:**

```
Text: "a photo of a cat"
       ↓ CLIP Text Encoder
    Text Embedding c_text ∈ ℝ^(77×768)
       ↓
    Cross-Attention in U-Net
       ↓
Latent z_T ~ N(0, I) ∈ ℝ^(64×64×4)
       ↓ DDIM (50 steps)
Latent z_0 ∈ ℝ^(64×64×4)
       ↓ VAE Decoder
Image x ∈ ℝ^(512×512×3)
```

**Cross-Attention for Conditioning:**

```python
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Query from image features, Key/Value from text
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)

    def forward(self, x, context):
        """
        Args:
            x: image features [B, H*W, dim]
            context: text features [B, seq_len, context_dim]
        """
        # Cross-attention
        q = self.to_q(self.norm1(x))
        k = self.to_k(context)
        v = self.to_v(context)

        attn_out, _ = self.attn(q, k, v)
        x = x + attn_out

        return self.norm2(x)
```

### 7.4 SDE와 Probability Flow ODE

**Song et al. (2021)의 통합 이론**

**정의 7.4 (Forward SDE)**

이산 시간 diffusion을 연속 시간 SDE로 일반화:

```
dx = f(x, t) dt + g(t) dW
```

여기서:
- f(x, t): drift coefficient
- g(t): diffusion coefficient
- W: Wiener process (Brownian motion)

**DDPM의 SDE 표현:**

```
dx = -1/2 β(t) x dt + √β(t) dW
```

**Reverse SDE:**

Anderson (1982)의 결과:

```
dx = [f(x, t) - g(t)² ∇_x log p_t(x)] dt + g(t) dW̄
```

여기서 W̄는 reverse time Wiener process입니다.

**정리 7.1 (Probability Flow ODE)**

다음 ODE는 reverse SDE와 **동일한 marginal distribution**을 가집니다:

```
dx = [f(x, t) - 1/2 g(t)² ∇_x log p_t(x)] dt
```

**증명 (Sketch):**

Fokker-Planck equation을 사용하여 두 process의 marginal evolution이 동일함을 보일 수 있습니다.

**실용적 의미:**

1. **Exact Likelihood 계산:**
   ```python
   def compute_likelihood(self, x_0):
       """ODE를 이용한 정확한 likelihood 계산"""
       # x_0 -> x_T via ODE (deterministic)
       # Instantaneous change of variables formula

       def ode_func(t, state):
           x, logp = state[0], state[1]
           with torch.enable_grad():
               x.requires_grad_(True)
               drift = self.probability_flow_drift(x, t)
               div = self.divergence(drift, x)
           dlogp = -div
           return drift, dlogp

       # Integrate ODE
       x_T, delta_logp = odeint(ode_func, (x_0, 0), t_span)

       # Prior log-likelihood
       logp_T = -0.5 * x_T.pow(2).sum() - 0.5 * np.log(2*np.pi) * np.prod(x_T.shape)

       return logp_T + delta_logp
   ```

2. **Flexible Sampling:**
   - SDE solver (stochastic, 다양성 높음)
   - ODE solver (deterministic, 빠름)

**Variance Exploding (VE) vs Variance Preserving (VP):**

**VP SDE (DDPM):**
```
dx = -1/2 β(t) x dt + √β(t) dW

Variance: Var[x_t] → 1  (bounded)
```

**VE SDE:**
```
dx = √(dσ²(t)/dt) dW

Variance: Var[x_t] → ∞  (exploding)
```

**Sub-VP SDE:**
```
dx = -1/2 β(t) x dt + √(β(t)(1 - e^{-2∫β(s)ds})) dW
```

### 7.5 Consistency Models (Song et al. 2023)

**동기:** DDPM/DDIM은 여전히 느림 (10-100 스텝)

**핵심 아이디어:** **1-step generation**을 가능하게!

**정의 7.5 (Consistency Model)**

Consistency function f:

```
f(x_t, t) = x_0  for all t ∈ [0, T]
```

즉, 같은 trajectory의 모든 점이 동일한 x_0로 매핑됩니다.

**Self-Consistency Property:**

```
f(x_t, t) = f(x_s, s)  for all t, s
```

**학습 방법 1: Consistency Distillation**

Pre-trained diffusion model로부터 distillation:

```python
def consistency_distillation_loss(self, x_0):
    """
    사전 학습된 diffusion model로부터 consistency model 증류
    """
    # 연속된 두 시간 단계 샘플링
    n = torch.randint(0, self.num_steps - 1, (x_0.shape[0],))
    t_n = self.timesteps[n]
    t_n1 = self.timesteps[n + 1]

    # x_{t_n} 샘플링
    noise = torch.randn_like(x_0)
    x_tn = self.q_sample(x_0, t_n, noise)

    # Pretrained diffusion으로 x_{t_{n+1}} -> x_{t_n} 한 스텝 이동
    with torch.no_grad():
        x_tn_target = self.ddim_step(x_tn, t_n, t_n1)

    # Consistency: f(x_{t_{n+1}}, t_{n+1}) = f(x_{t_n}, t_n)
    pred_tn1 = self.consistency_function(x_tn, t_n1)
    pred_tn = self.consistency_function(x_tn_target, t_n)

    loss = F.mse_loss(pred_tn1, pred_tn.detach())
    return loss
```

**학습 방법 2: Consistency Training**

Diffusion model 없이 직접 학습:

```python
def consistency_training_loss(self, x_0):
    """
    Score function만 사용하여 consistency model 직접 학습
    """
    n = torch.randint(0, self.num_steps - 1, (x_0.shape[0],))
    t_n = self.timesteps[n]
    t_n1 = self.timesteps[n + 1]

    noise = torch.randn_like(x_0)
    x_tn1 = self.q_sample(x_0, t_n1, noise)

    # Score function으로 Euler step
    with torch.no_grad():
        score = self.score_model(x_tn1, t_n1)
        x_tn_target = x_tn1 + (t_n - t_n1) * score

    # Consistency
    pred_tn1 = self.consistency_function(x_tn1, t_n1)
    pred_tn = self.consistency_function(x_tn_target, t_n)

    loss = F.mse_loss(pred_tn1, pred_tn.detach())
    return loss
```

**1-Step Sampling:**

```python
@torch.no_grad()
def sample_consistency(self, shape):
    """1-step generation!"""
    x_T = torch.randn(shape, device=self.device)
    x_0 = self.consistency_function(x_T, self.T)
    return x_0
```

**Multi-Step Sampling (optional):**

더 나은 품질을 위해:

```python
@torch.no_grad()
def sample_consistency_multistep(self, shape, steps=2):
    """Few-step generation for better quality"""
    x = torch.randn(shape, device=self.device)
    timesteps = torch.linspace(self.T, 0, steps+1)

    for i in range(steps):
        t = timesteps[i]
        # Consistency function
        x_0_pred = self.consistency_function(x, t)

        if i < steps - 1:
            # Add noise and continue
            t_next = timesteps[i+1]
            noise = torch.randn_like(x)
            x = self.q_sample(x_0_pred, t_next, noise)

    return x_0_pred
```

**성능:**

- **1-step**: FID ~10 (ImageNet 64x64)
- **2-step**: FID ~5
- **4-step**: FID ~3 (DDIM 50-step과 비슷)

### 7.6 Flow Matching (Lipman et al. 2023)

**동기:** SDE/ODE는 복잡하고 학습이 어려움

**핵심 아이디어:** **Continuous Normalizing Flow**를 직접 학습!

**정의 7.6 (Flow Matching)**

Velocity field v_t(x)를 학습하여:

```
dx/dt = v_t(x)

x(0) ~ p_data
x(1) ~ N(0, I)
```

**Conditional Flow:**

각 데이터 x_0에 대해 conditional path:

```
x_t = α_t x_0 + σ_t ε,  ε ~ N(0, I)
```

예: Linear interpolation
```
α_t = 1 - t,  σ_t = t
x_t = (1-t) x_0 + t ε
```

**Conditional Velocity:**

```
v_t(x_t | x_0) = dx_t/dt = -x_0 + ε
```

**Flow Matching Objective:**

```
L_FM(θ) = E_{t,x_0,ε} [||v_θ(x_t, t) - v_t(x_t | x_0)||²]
         = E_{t,x_0,ε} [||v_θ((1-t)x_0 + tε, t) - (ε - x_0)||²]
```

**구현:**

```python
class FlowMatching(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Velocity network v_θ

    def forward(self, x_0):
        """Flow matching loss"""
        b = x_0.shape[0]
        t = torch.rand(b, device=x_0.device)[:, None, None, None]
        eps = torch.randn_like(x_0)

        # Conditional path: x_t = (1-t)x_0 + t*eps
        x_t = (1 - t) * x_0 + t * eps

        # Target velocity: ε - x_0
        v_target = eps - x_0

        # Predicted velocity
        v_pred = self.model(x_t, t.squeeze())

        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def sample(self, shape, steps=100):
        """ODE integration for sampling"""
        x = torch.randn(shape, device=next(self.model.parameters()).device)
        dt = 1.0 / steps

        for i in range(steps):
            t = i / steps
            t_batch = torch.full((shape[0],), t, device=x.device)

            # Euler step
            v = self.model(x, t_batch)
            x = x + v * dt

        return x
```

**장점:**

- **간단한 목적 함수**: Diffusion의 복잡한 weighting 불필요
- **빠른 샘플링**: ODE로 직접 적분 (10-20 스텝)
- **학습 안정성**: Diffusion보다 안정적

**Optimal Transport Flow:**

더 나은 성능을 위해 OT flow 사용:

```
Minimize: W_2(p_0, p_1)  (Wasserstein-2 distance)

Result: Straight line paths in Wasserstein space
```

---

## 8. 최신 연구 동향 (2023-2025)

### 8.1 Fast Sampling의 발전

**1. DPM-Solver++ (Lu et al. 2022)**

고차 ODE solver로 5-10 스텝 샘플링:

```python
def dpm_solver_step(x, t, t_next, order=2):
    """
    DPM-Solver++ 2nd order step
    """
    h = t_next - t

    # 1st order (Exponential integrator)
    lambda_t = -torch.log(alphas_cumprod[t])
    lambda_next = -torch.log(alphas_cumprod[t_next])
    h_lambda = lambda_next - lambda_t

    noise_pred = model(x, t)
    x_pred = (alphas_cumprod[t_next] / alphas_cumprod[t]) * x - \
             sigma[t_next] * torch.expm1(h_lambda) * noise_pred

    if order == 2:
        # 2nd order correction
        noise_pred_next = model(x_pred, t_next)
        x_pred = x_pred - sigma[t_next] * torch.expm1(h_lambda) / h_lambda * \
                 (noise_pred_next - noise_pred) * 0.5

    return x_pred
```

**2. LCM (Latent Consistency Models, Luo et al. 2023)**

Stable Diffusion을 1-4 스텝으로 압축:

- Consistency distillation + latent space
- Real-time image generation 가능!

### 8.2 Text-to-Image의 발전

**1. DALL-E 3 (2023)**

- Improved captioning: 더 정확한 text-image alignment
- Synthetic data: CLIP으로 재캡셔닝
- Safety: Harmful content 필터링 강화

**2. Midjourney v6 (2024)**

- Prompt adherence 개선
- Photorealism 향상
- Consistent characters

**3. Stable Diffusion 3 (2024)**

- **Multimodal Diffusion Transformer (MMDiT)**:
  ```
  Text + Image를 joint latent space에서 처리
  Separate streams → Cross-attention → Merged processing
  ```

- **Rectified Flow**:
  Flow matching 기반, ODE로 빠른 샘플링

### 8.3 Video Diffusion

**1. Imagen Video (Google, 2022)**

Cascaded video diffusion:

```
Text → 16 frames @ 24×40
     → 16 frames @ 48×80
     → 128 frames @ 192×320
     → 128 frames @ 768×1280 (temporal SR + spatial SR)
```

**2. Stable Video Diffusion (Stability AI, 2023)**

Image-to-video:

```python
class VideoUNet(nn.Module):
    """
    3D U-Net with temporal attention
    """
    def forward(self, x, t, cond_image):
        """
        Args:
            x: [B, C, F, H, W] (frames)
            t: timestep
            cond_image: conditioning first frame
        """
        # Temporal conv3d
        h = self.temp_conv(x)

        # Spatial attention (per frame)
        h = rearrange(h, 'b c f h w -> (b f) c h w')
        h = self.spatial_attn(h)
        h = rearrange(h, '(b f) c h w -> b c f h w', b=b)

        # Temporal attention (across frames)
        h = rearrange(h, 'b c f h w -> (b h w) f c')
        h = self.temporal_attn(h)
        h = rearrange(h, '(b h w) f c -> b c f h w', b=b, h=h, w=w)

        return h
```

**3. Sora (OpenAI, 2024)**

- **Patches in spacetime**: Video를 spacetime patches로 분할
- **Transformer**: Diffusion Transformer 사용
- **Variable duration/resolution**: 유연한 생성

### 8.4 3D Generation

**1. DreamFusion (Poole et al. 2022)**

Text-to-3D via **Score Distillation Sampling (SDS)**:

```python
def score_distillation_sampling(nerf, text, camera):
    """
    SDS loss for 3D generation

    Args:
        nerf: NeRF model
        text: text prompt
        camera: camera parameters

    Returns:
        SDS loss
    """
    # Render image from NeRF
    img = nerf.render(camera)

    # Add noise
    t = torch.randint(0, timesteps, (1,))
    noise = torch.randn_like(img)
    img_noisy = sqrt_alphas_cumprod[t] * img + sqrt_one_minus_alphas_cumprod[t] * noise

    # Predict noise with text conditioning
    with torch.no_grad():
        noise_pred = diffusion_model(img_noisy, t, text)

    # SDS gradient
    w = (1 - alphas_cumprod[t])
    grad = w * (noise_pred - noise)

    # Backprop through NeRF
    loss = (grad * img).sum()
    return loss
```

**2. Magic3D, Fantasia3D (2023)**

- Coarse-to-fine 3D generation
- Mesh optimization
- Texture refinement

### 8.5 Personalization

**1. DreamBooth (Ruiz et al. 2022)**

Few-shot personalization:

```python
def dreambooth_finetuning(diffusion_model, images, prompt_template):
    """
    DreamBooth fine-tuning

    Args:
        diffusion_model: pretrained model
        images: 3-5 images of subject
        prompt_template: "a [V] dog" ([V] is rare token)
    """
    # Freeze most layers, finetune only subset
    for param in diffusion_model.parameters():
        param.requires_grad = False
    for param in diffusion_model.final_layers.parameters():
        param.requires_grad = True

    # Prior preservation loss
    for epoch in range(1000):
        # Subject loss
        loss_subject = diffusion_loss(images, prompt_template)

        # Class loss (prevent forgetting)
        loss_class = diffusion_loss(generated_class_images, "a dog")

        loss = loss_subject + lambda_prior * loss_class
        loss.backward()
```

**2. LoRA (Low-Rank Adaptation)**

Parameter-efficient fine-tuning:

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 1.0 / rank

    def forward(self, x, W_pretrained):
        # W_pretrained: frozen pretrained weight
        # LoRA: W = W_pretrained + α * A * B
        return F.linear(x, W_pretrained) + \
               self.scaling * (x @ self.lora_A @ self.lora_B)
```

**3. Textual Inversion**

Optimize embedding only:

```python
# Freeze entire model
for param in model.parameters():
    param.requires_grad = False

# Optimize only the new token embedding
new_token_embedding = nn.Parameter(torch.randn(768))  # CLIP dim

for epoch in range(5000):
    # Replace [V] with learned embedding
    text_embedding[V_token_idx] = new_token_embedding

    loss = diffusion_loss(images, text_embedding)
    loss.backward()
    optimizer.step()
```

### 8.6 Controllability

**1. ControlNet (Zhang et al. 2023)**

추가적인 spatial control:

```python
class ControlNet(nn.Module):
    def __init__(self, unet):
        super().__init__()
        # Copy encoder from U-Net
        self.control_encoder = copy.deepcopy(unet.encoder)

        # Trainable copy
        for param in self.control_encoder.parameters():
            param.requires_grad = True

        # Zero convolutions (initialize to zero)
        self.zero_convs = nn.ModuleList([
            nn.Conv2d(ch, ch, 1) for ch in encoder_channels
        ])
        for conv in self.zero_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x, control_input, unet):
        """
        Args:
            control_input: edge map, depth, pose, etc.
        """
        # Encode control
        control_features = self.control_encoder(control_input)

        # Apply zero convs
        control_features = [conv(f) for conv, f in zip(self.zero_convs, control_features)]

        # Add to U-Net features
        out = unet(x, additional_features=control_features)
        return out
```

**Controls:**
- Canny edge
- Depth map
- Normal map
- Pose (OpenPose)
- Segmentation
- Scribble

**2. T2I-Adapter**

Lightweight control:

```python
class T2IAdapter(nn.Module):
    """
    Lightweight adapter for control
    """
    def __init__(self):
        super().__init__()
        # Only 77M params vs ControlNet's 1.2B
        self.adapter = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            ResBlock(64, 128),
            ResBlock(128, 256),
            ResBlock(256, 512),
        )

    def forward(self, control_input):
        features = self.adapter(control_input)
        return features  # Add to U-Net via residual
```

### 8.7 Efficiency Improvements

**1. Progressive Distillation (Salimans & Ho, 2022)**

Halve steps iteratively:

```
Teacher (1000 steps) → Student (500 steps)
→ Student (250 steps) → Student (125 steps)
→ ... → Student (4 steps)
```

**2. Token Merging for Diffusion**

Merge redundant tokens in attention:

```python
def token_merging(x, similarity_threshold=0.9):
    """
    Merge similar tokens to reduce computation
    """
    # Compute similarity
    sim = x @ x.T / (x.norm(dim=-1, keepdim=True) @ x.norm(dim=-1, keepdim=True).T)

    # Find pairs to merge
    merge_pairs = (sim > similarity_threshold).nonzero()

    # Merge tokens
    for i, j in merge_pairs:
        x[i] = (x[i] + x[j]) / 2
        x[j] = 0  # Mark for removal

    # Remove merged tokens
    x = x[x.sum(dim=-1) != 0]
    return x
```

**3. Quantization & Pruning**

- INT8 quantization: 4배 메모리 절약
- Structured pruning: 불필요한 레이어 제거

### 8.8 Theoretical Understanding

**1. Diffusion as Denoising Autoencoders**

Connection to score matching and denoising:

```
Score matching ≈ Denoising autoencoder ≈ Diffusion models
```

**2. Blessing of Dimensionality**

고차원에서 diffusion이 잘 작동하는 이유:

- Manifold hypothesis
- Concentration of measure
- Score matching의 안정성

**3. Mode Coverage Analysis**

Diffusion vs GAN:

```
Precision: How realistic samples are
Recall: How diverse samples are

GAN: High precision, low recall (mode collapse)
Diffusion: High precision, high recall
```

### 8.9 Open Problems

**1. Sampling Speed**

- 여전히 real-time에는 부족
- 1-step generation의 품질 개선 필요

**2. Controllability**

- Fine-grained control 어려움
- Compositional generation 미흡

**3. 3D & Video**

- Temporal consistency
- View consistency
- Long video generation

**4. Efficiency**

- Large-scale 모델의 계산 비용
- Edge device deployment

**5. Safety & Ethics**

- Deepfake 방지
- Copyright 문제
- Bias 완화

---

## 9. 결론 및 향후 연구 방향

### 9.1 Diffusion Models의 성과

Diffusion Models는 2020년대 생성 모델 연구의 패러다임을 완전히 바꾸었습니다.

**주요 성과:**

1. **이미지 생성의 혁신**
   - DALL-E 2, Midjourney, Stable Diffusion
   - Photorealistic 수준의 품질
   - Text-to-image의 대중화

2. **이론적 엄밀성**
   - Score matching theory
   - SDE/ODE 통합 framework
   - Exact likelihood 계산 가능

3. **학습 안정성**
   - GAN의 mode collapse 문제 해결
   - VAE의 posterior collapse 문제 회피
   - 안정적이고 재현 가능한 학습

4. **다양한 응용 분야**
   - Image: super-resolution, inpainting, editing
   - Video: generation, interpolation
   - 3D: text-to-3D, novel view synthesis
   - Audio: music generation, speech synthesis
   - Scientific: protein design, molecular generation

### 9.2 주요 한계와 도전 과제

**1. 계산 비용**

- **문제:**
  - 학습에 수백만 GPU 시간 필요
  - 샘플링이 느림 (1-100 스텝)

- **현재 해결책:**
  - Latent diffusion (계산량 64배 감소)
  - Fast samplers (DDIM, DPM-Solver)
  - Consistency models (1-step generation)

- **미해결 문제:**
  - Real-time generation (30+ FPS)
  - Edge device deployment

**2. 제어 가능성**

- **문제:**
  - Fine-grained control 어려움
  - Compositional generation 미흡
  - Attribute binding problem ("red cube and blue sphere")

- **현재 해결책:**
  - Classifier-free guidance
  - ControlNet, T2I-Adapter
  - Attention manipulation

- **미해결 문제:**
  - Multi-object scene composition
  - Consistent character generation
  - Long-range coherence

**3. 3D 및 Video 생성**

- **문제:**
  - Temporal consistency 부족
  - View consistency 문제
  - Long video generation 어려움

- **현재 해결책:**
  - Temporal attention layers
  - Multi-view consistency loss
  - Cascaded generation

- **미해결 문제:**
  - Hour-long video generation
  - Interactive 3D scenes
  - Physical plausibility

**4. 안전성 및 윤리**

- **문제:**
  - Deepfake 생성
  - Copyright 침해
  - Bias 증폭

- **현재 해결책:**
  - Watermarking
  - Content filtering
  - Bias mitigation techniques

- **미해결 문제:**
  - Foolproof detection
  - Fair use vs copyright
  - Universal ethical standards

### 9.3 향후 연구 방향

**1. Unified Generative Models**

다양한 modality를 하나의 모델로 통합:

```
Text + Image + Video + Audio + 3D
           ↓
    Unified Diffusion Model
           ↓
  Any-to-Any Generation
```

**주요 연구:**
- Meta's ImageBind
- Google's Gemini
- OpenAI's GPT-4V

**2. Efficient Diffusion Models**

```
목표: 1-step, real-time, on-device generation

방향:
- Consistency models 개선
- Progressive distillation
- Neural architecture search
- Quantization & pruning
```

**3. Controllable Generation**

```
Fine-grained control:
- Object-level editing
- Attribute disentanglement
- Physics-aware generation
- Interactive editing tools
```

**4. Theoretical Understanding**

미해결 이론적 질문:

- **Why do diffusion models work so well?**
  - Manifold hypothesis
  - Inductive bias of neural networks
  - Blessing of dimensionality

- **Optimal noise schedule**
  - 어떤 schedule이 최적인가?
  - Task-dependent schedule?

- **Sample quality vs diversity trade-off**
  - Pareto frontier 특성화
  - Adaptive guidance

**5. Applications in Science**

```
Protein Design:
- Diffusion models for protein structure prediction
- Antibody design
- Enzyme engineering

Drug Discovery:
- Molecular generation
- Property optimization
- Synthesis planning

Materials Science:
- Crystal structure prediction
- Materials design
```

### 9.4 마무리

Diffusion Models는 단순히 좋은 이미지를 생성하는 도구를 넘어, **확률론적 생성 모델의 새로운 패러다임**을 제시했습니다.

**핵심 통찰:**

1. **점진적 변환의 힘**: 복잡한 분포를 작은 스텝들로 분해
2. **Score function의 중요성**: Normalizing constant 없이 학습 가능
3. **이론과 실용의 조화**: 수학적으로 엄밀하면서도 실용적

**연구자를 위한 조언:**

1. **수학적 기초를 탄탄히**: Score matching, SDE, ODE 이론 이해
2. **코드로 검증**: 이론을 직접 구현하며 이해
3. **최신 연구 팔로우**: 빠르게 발전하는 분야
4. **응용 분야 탐색**: 자신의 도메인에 어떻게 적용할지 고민

Diffusion models는 아직 발전 초기 단계입니다. 앞으로 더 놀라운 발전이 기대됩니다!

---

## 10. 부록

### 10.1 수학적 배경

#### A. Gaussian Distribution

**정의:**

```
N(x; μ, σ²) = 1/√(2πσ²) exp(-1/(2σ²) (x-μ)²)
```

**다변량 Gaussian:**

```
N(x; μ, Σ) = 1/√((2π)^d |Σ|) exp(-1/2 (x-μ)ᵀ Σ⁻¹ (x-μ))
```

**성질:**

1. **선형 변환:**
   ```
   X ~ N(μ, Σ)
   Y = AX + b
   ⇒ Y ~ N(Aμ + b, AΣAᵀ)
   ```

2. **Marginalization:**
   ```
   [X₁]   [μ₁]   [Σ₁₁  Σ₁₂]
   [X₂] ~ N([μ₂], [Σ₂₁  Σ₂₂])

   ⇒ X₁ ~ N(μ₁, Σ₁₁)
   ```

3. **Conditioning:**
   ```
   X₁|X₂ ~ N(μ₁ + Σ₁₂Σ₂₂⁻¹(X₂ - μ₂), Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁)
   ```

4. **Sum of independent Gaussians:**
   ```
   X ~ N(μ₁, σ₁²), Y ~ N(μ₂, σ₂²)
   ⇒ X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²)
   ```

#### B. KL Divergence

**정의:**

```
D_KL(P || Q) = ∫ p(x) log(p(x)/q(x)) dx
```

**성질:**

1. **Non-negativity:** D_KL(P || Q) ≥ 0, 등호는 P = Q
2. **Asymmetry:** D_KL(P || Q) ≠ D_KL(Q || P)

**Gaussian KL:**

```
P = N(μ₁, Σ₁), Q = N(μ₂, Σ₂)

D_KL(P || Q) = 1/2 [tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - d + log(|Σ₂|/|Σ₁|)]
```

**Diagonal Gaussian:**

```
D_KL(N(μ₁, σ₁²I) || N(μ₂, σ₂²I)) = d/2 [σ₁²/σ₂² + ||μ₁-μ₂||²/σ₂² - 1 - log(σ₁²/σ₂²)]
```

#### C. Stochastic Differential Equations (SDE)

**일반 형태:**

```
dx = f(x, t) dt + g(t) dW
```

- f(x, t): drift
- g(t): diffusion coefficient
- dW: Wiener process (Brownian motion)

**Itô's Lemma:**

Y = h(X_t, t)일 때:

```
dY = (∂h/∂t + f ∂h/∂x + 1/2 g² ∂²h/∂x²) dt + g ∂h/∂x dW
```

**Fokker-Planck Equation:**

확률 밀도 p(x, t)의 evolution:

```
∂p/∂t = -∂/∂x [f(x,t)p] + 1/2 ∂²/∂x² [g(t)²p]
```

**Reverse SDE (Anderson 1982):**

Forward SDE의 reverse:

```
dx = [f(x,t) - g(t)² ∇_x log p_t(x)] dt + g(t) dW̄
```

#### D. Ordinary Differential Equations (ODE)

**Initial Value Problem:**

```
dx/dt = f(x, t),  x(t₀) = x₀
```

**Euler Method:**

```
x_{n+1} = x_n + h f(x_n, t_n)
```

**Runge-Kutta 4th Order:**

```
k₁ = f(x_n, t_n)
k₂ = f(x_n + h/2 k₁, t_n + h/2)
k₃ = f(x_n + h/2 k₂, t_n + h/2)
k₄ = f(x_n + h k₃, t_n + h)

x_{n+1} = x_n + h/6 (k₁ + 2k₂ + 2k₃ + k₄)
```

#### E. Variational Inference

**ELBO (Evidence Lower Bound):**

```
log p(x) ≥ E_q [log p(x,z)] - E_q [log q(z)]
         = E_q [log p(x|z)] - D_KL(q(z) || p(z))
```

**Mean Field Approximation:**

```
q(z) = ∏ᵢ q_i(z_i)
```

### 10.2 구현 팁

#### A. 수치적 안정성

**1. Log-space 계산:**

```python
# Bad: overflow/underflow
p = torch.exp(log_p1) + torch.exp(log_p2)

# Good: log-sum-exp trick
log_p = torch.logsumexp(torch.stack([log_p1, log_p2]), dim=0)
```

**2. Gradient clipping:**

```python
# Prevent gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**3. Noise scheduling:**

```python
# Clip betas to avoid numerical issues
betas = torch.clip(betas, min=1e-4, max=0.999)
```

#### B. 효율적인 구현

**1. Mixed precision training:**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**2. Gradient checkpointing:**

```python
# Trade compute for memory
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(module, x):
    return checkpoint(module, x)
```

**3. Efficient attention:**

```python
# Use Flash Attention for large sequences
from flash_attn import flash_attn_qkvpacked_func

# Standard attention: O(n²) memory
# Flash attention: O(n) memory
```

#### C. 디버깅 팁

**1. Visualization:**

```python
# Visualize forward process
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
timesteps = torch.linspace(0, model.timesteps-1, 10).long()

for i, t in enumerate(timesteps):
    x_t = model.q_sample(x_0, t.unsqueeze(0))
    axes[i].imshow(x_t[0].permute(1, 2, 0).cpu())
    axes[i].set_title(f't={t}')
    axes[i].axis('off')
```

**2. Loss monitoring:**

```python
# Track different loss components
losses = {
    'total': [],
    'mse': [],
    'vlb': [],
}

# Check for NaN/Inf
assert not torch.isnan(loss).any()
assert not torch.isinf(loss).any()
```

**3. Gradient monitoring:**

```python
# Check gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f'{name}: {grad_norm:.4f}')
```

### 10.3 추가 자료

#### A. 필수 논문

**Foundation Papers:**

1. **Sohl-Dickstein et al. (2015)**
   - "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
   - 최초의 diffusion models

2. **Ho et al. (2020)**
   - "Denoising Diffusion Probabilistic Models"
   - DDPM, 실용화의 시작

3. **Song et al. (2021)**
   - "Score-Based Generative Modeling through SDEs"
   - SDE framework, 이론적 통합

4. **Song et al. (2021)**
   - "Denoising Diffusion Implicit Models"
   - DDIM, fast sampling

**Advanced Topics:**

5. **Dhariwal & Nichol (2021)**
   - "Diffusion Models Beat GANs on Image Synthesis"
   - Improved DDPM

6. **Rombach et al. (2022)**
   - "High-Resolution Image Synthesis with Latent Diffusion Models"
   - Stable Diffusion

7. **Ho & Salimans (2022)**
   - "Classifier-Free Diffusion Guidance"
   - 조건부 생성의 표준

8. **Song et al. (2023)**
   - "Consistency Models"
   - 1-step generation

#### B. 구현 참고 자료

**Official Implementations:**

- OpenAI DDPM: https://github.com/openai/improved-diffusion
- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- Diffusers (Hugging Face): https://github.com/huggingface/diffusers

**Educational Resources:**

- Lilian Weng's Blog: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- Yang Song's Blog: https://yang-song.net/blog/2021/score/
- Hugging Face Diffusion Course: https://huggingface.co/docs/diffusers/

#### C. 데이터셋

**이미지:**
- CIFAR-10: 60K 32×32 images
- ImageNet: 14M images, 1000 classes
- LAION-5B: 5.8B image-text pairs
- CelebA-HQ: 30K high-quality faces

**기타:**
- AudioSet: Audio classification
- ShapeNet: 3D models
- KITTI: Autonomous driving

### 10.4 용어 정리

**한영 대조:**

| 한국어 | English |
|--------|---------|
| 확산 모델 | Diffusion Models |
| 점수 함수 | Score Function |
| 순방향 과정 | Forward Process |
| 역방향 과정 | Reverse Process |
| 노이즈 스케줄 | Noise Schedule |
| 재매개변수화 | Reparameterization |
| 변분 하한 | Variational Lower Bound (ELBO) |
| 가이던스 | Guidance |
| 잠재 확산 | Latent Diffusion |

**약어:**

- DDPM: Denoising Diffusion Probabilistic Models
- DDIM: Denoising Diffusion Implicit Models
- SDE: Stochastic Differential Equation
- ODE: Ordinary Differential Equation
- ELBO: Evidence Lower Bound
- VAE: Variational Autoencoder
- GAN: Generative Adversarial Network
- FID: Fréchet Inception Distance
- IS: Inception Score
- CFG: Classifier-Free Guidance

---

## 참고문헌

**주요 논문:**

1. Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *ICML*.

2. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.

3. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*.

4. Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR*.

5. Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *ICML*.

6. Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. *NeurIPS*.

7. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *CVPR*.

8. Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *NeurIPS Workshop*.

9. Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency models. *ICML*.

10. Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow matching for generative modeling. *ICLR*.

**추가 참고자료:**

11. Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Computation*.

12. Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *JMLR*.

13. Anderson, B. D. (1982). Reverse-time diffusion equation models. *Stochastic Processes and their Applications*.

14. Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *NeurIPS*.

15. Salimans, T., & Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. *ICLR*.

**최신 응용:**

16. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). Hierarchical text-conditional image generation with CLIP latents. *arXiv* (DALL-E 2).

17. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., ... & Norouzi, M. (2022). Photorealistic text-to-image diffusion models with deep language understanding. *NeurIPS* (Imagen).

18. Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2022). DreamFusion: Text-to-3D using 2D diffusion. *ICLR*.

19. Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. *ICCV* (ControlNet).

20. Brooks, T., Holynski, A., & Efros, A. A. (2023). InstructPix2Pix: Learning to follow image editing instructions. *CVPR*.

---

**끝.**

이 문서는 Diffusion Models의 수학적 기초부터 최신 응용까지 석사 수준의 깊이로 다루었습니다. 질문이나 피드백은 언제든 환영합니다!
