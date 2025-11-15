# 생성 모델 학습 경로 가이드

> **목표**: 자신의 수준에 맞는 최적의 학습 경로 제시
>
> **업데이트**: 2025-11-15

---

## 📚 목차

1. [레벨 진단 테스트](#1-레벨-진단-테스트)
2. [초급 경로 (Beginner)](#2-초급-경로-beginner)
3. [중급 경로 (Intermediate)](#3-중급-경로-intermediate)
4. [고급 경로 (Advanced)](#4-고급-경로-advanced)
5. [체크포인트 퀴즈](#5-체크포인트-퀴즈)
6. [학습 로드맵 시각화](#6-학습-로드맵-시각화)

---

## 1. 레벨 진단 테스트

### 자가 진단 질문

다음 질문에 "예/아니오"로 답하세요:

#### 기초 지식
- [ ] Python 기본 문법을 알고 있다
- [ ] NumPy 배열 연산을 할 수 있다
- [ ] Matplotlib으로 그래프를 그릴 수 있다
- [ ] 미적분(도함수, 편미분) 기본 개념을 안다
- [ ] 선형대수(행렬, 벡터) 기본을 안다
- [ ] 확률론(확률 분포, 기댓값) 기본을 안다

#### 딥러닝 기초
- [ ] PyTorch로 간단한 신경망을 만들 수 있다
- [ ] 역전파 알고리즘의 원리를 설명할 수 있다
- [ ] CNN의 작동 방식을 안다
- [ ] Optimizer(Adam, SGD)를 사용해본 적 있다
- [ ] TensorBoard로 학습을 모니터링해본 적 있다

#### 생성 모델
- [ ] Autoencoder의 구조를 설명할 수 있다
- [ ] VAE의 ELBO를 유도할 수 있다
- [ ] GAN의 minimax game을 이해한다
- [ ] Diffusion Model의 forward process를 설명할 수 있다
- [ ] Transformer의 attention 메커니즘을 안다

### 레벨 판정

**점수 계산:**
- 기초 지식 (6문항): 각 1점
- 딥러닝 기초 (5문항): 각 2점
- 생성 모델 (5문항): 각 3점

**레벨:**
- **0-8점**: 초급 (Beginner) - 기초부터 시작
- **9-20점**: 중급 (Intermediate) - 생성 모델 집중
- **21-31점**: 고급 (Advanced) - 최신 연구 및 실전

---

## 2. 초급 경로 (Beginner)

> **대상**: 딥러닝 초보자, 수학/프로그래밍 기초만 있는 분
>
> **목표**: 생성 모델의 기초를 탄탄히 다지기
>
> **예상 기간**: 3-4개월

### 학습 경로

```
Week 1-2: Python & 수학 기초
    ↓
Week 3-4: PyTorch 기초
    ↓
Week 5-6: 신경망 기초 (MLP, CNN)
    ↓
Week 7-8: Autoencoder
    ↓
Week 9-11: VAE (핵심!)
    ↓
Week 12-14: GAN 기초 (DCGAN)
    ↓
Week 15-16: 프로젝트 & 복습
```

### 상세 일정표

#### Week 1-2: Python & 수학 기초 (14시간)

**학습 자료:**
- Python: [Python Tutorial](https://docs.python.org/3/tutorial/)
- NumPy: [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- 미적분: Khan Academy - Calculus
- 선형대수: 3Blue1Brown - Essence of Linear Algebra

**실습:**
```python
# NumPy 기본 연산
import numpy as np

# 행렬 연산
A = np.random.randn(3, 3)
B = np.random.randn(3, 3)
C = A @ B  # 행렬 곱

# Broadcasting
x = np.array([1, 2, 3])
y = x + 10  # [11, 12, 13]

# 그래프 그리기
import matplotlib.pyplot as plt
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x))
plt.show()
```

**체크포인트:** [퀴즈 1 - 기초](#퀴즈-1-python--수학-기초)

#### Week 3-4: PyTorch 기초 (14시간)

**학습 자료:**
- [PyTorch 60분 튜토리얼](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- `code/hello_pytorch.py` 실행

**실습:**
```python
import torch
import torch.nn as nn

# Tensor 기본
x = torch.randn(3, 4)
y = x * 2

# 간단한 신경망
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# 학습
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    # Forward
    pred = model(x)
    loss = (pred - y).pow(2).mean()

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**과제:**
- MNIST 분류기 만들기 (MLP)
- Loss curve 그리기

**체크포인트:** [퀴즈 2 - PyTorch 기초](#퀴즈-2-pytorch-기초)

#### Week 5-6: 신경망 기초 (14시간)

**학습 자료:**
- README.md - Phase 2 (딥러닝 기초)
- `docs/phase2-deep-learning.md`

**실습:**
```python
# CNN으로 MNIST 분류
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**과제:**
- CIFAR-10 분류 (>70% accuracy)

**체크포인트:** [퀴즈 3 - CNN](#퀴즈-3-cnn)

#### Week 7-8: Autoencoder (14시간)

**학습 자료:**
- README.md - Week 10-11: Autoencoders

**실습:**
```python
# 실행
cd code/vae
python train_vae.py --model vae --dataset mnist --epochs 50 --latent-dim 20
```

**이해해야 할 것:**
- Encoder-Decoder 구조
- Reconstruction Loss
- Latent Space의 의미

**과제:**
- MNIST Autoencoder 직접 구현
- Latent space 시각화 (t-SNE)

**체크포인트:** [퀴즈 4 - Autoencoder](#퀴즈-4-autoencoder)

#### Week 9-11: VAE (핵심!) (21시간)

**학습 자료:**
- `docs/phase3-vae-detailed.md` 정독
- 논문: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)

**실습:**
```python
# code/vae/basic_vae.py 분석
# 핵심 3가지:
# 1. Reparameterization Trick
# 2. ELBO 계산
# 3. KL Divergence

# 학습
python train_vae.py --model vae --dataset mnist --epochs 50

# β-VAE 실험
python train_vae.py --model beta-vae --beta 4.0 --dataset mnist
```

**이해해야 할 것:**
- ELBO 유도 과정
- KL Divergence의 의미
- Reparameterization Trick 필요성

**과제:**
- VAE 처음부터 구현 (코드 보지 않고)
- Latent space interpolation
- β값 변화에 따른 영향 분석

**체크포인트:** [퀴즈 5 - VAE](#퀴즈-5-vae)

#### Week 12-14: GAN 기초 (21시간)

**학습 자료:**
- `docs/phase3-gan-detailed.md` - DCGAN 부분
- 논문: "Unsupervised Representation Learning with DCGAN"

**실습:**
```python
# DCGAN 학습
cd code/gan
python train_gan.py --model dcgan --dataset mnist --epochs 100

# 생성 샘플 확인
python generate.py --checkpoint checkpoints/dcgan_final.pth
```

**이해해야 할 것:**
- Minimax game
- Generator vs Discriminator
- Mode Collapse 문제
- DCGAN 5가지 아키텍처 규칙

**과제:**
- DCGAN 직접 구현
- CelebA로 얼굴 생성 (64x64)

**체크포인트:** [퀴즈 6 - GAN](#퀴즈-6-gan-기초)

#### Week 15-16: 프로젝트 & 복습 (14시간)

**프로젝트 선택:**
1. VAE로 이미지 압축 시스템
2. DCGAN으로 특정 도메인 이미지 생성
3. Autoencoder로 Denoising

**발표 준비:**
- 프로젝트 README 작성
- 결과 시각화 (샘플, 그래프)
- GitHub 업로드

**체크포인트:** [최종 평가 - 초급](#최종-평가-초급)

### 주간 학습 시간

| 주차 | 이론 | 실습 | 과제 | 합계 |
|------|------|------|------|------|
| 1-2 | 4h | 6h | 4h | 14h |
| 3-4 | 4h | 6h | 4h | 14h |
| 5-6 | 4h | 6h | 4h | 14h |
| 7-8 | 4h | 6h | 4h | 14h |
| 9-11 | 6h | 9h | 6h | 21h |
| 12-14 | 6h | 9h | 6h | 21h |
| 15-16 | 3h | 8h | 3h | 14h |

**총 학습 시간: 112시간 (주당 7시간 × 16주)**

---

## 3. 중급 경로 (Intermediate)

> **대상**: PyTorch 기초는 있고, 생성 모델을 본격적으로 배우고 싶은 분
>
> **목표**: 주요 생성 모델 마스터 및 실전 적용
>
> **예상 기간**: 2-3개월

### 학습 경로

```
Week 1-2: VAE 심화
    ↓
Week 3-4: GAN 심화 (WGAN-GP, StyleGAN)
    ↓
Week 5-7: Diffusion Models (DDPM, DDIM)
    ↓
Week 8-9: Normalizing Flows
    ↓
Week 10-11: Transformer 생성 모델
    ↓
Week 12: 프로젝트
```

### 상세 일정표

#### Week 1-2: VAE 심화 (14시간)

**학습 자료:**
- `docs/masters-level/vae-theory.md`
- β-VAE, VQ-VAE 논문

**실습:**
```python
# β-VAE 실험
cd code/experiments
python run_vae_experiments.py --experiments 1  # β 비교

# VQ-VAE 구현 (선택)
```

**심화 주제:**
- Disentangled Representation
- Posterior Collapse
- Hierarchical VAE

**과제:**
- β값에 따른 disentanglement 분석
- CIFAR-10으로 VAE 학습 (FID < 50)

**체크포인트:** [퀴즈 7 - VAE 심화](#퀴즈-7-vae-심화)

#### Week 3-4: GAN 심화 (14시간)

**학습 자료:**
- `docs/phase3-gan-detailed.md` - WGAN-GP
- `docs/masters-level/gan-theory.md`

**실습:**
```python
# WGAN-GP
python train_gan.py --model wgan-gp --dataset celeba

# FID 계산
cd code/experiments
python run_gan_experiments.py --experiments 1  # DCGAN vs WGAN-GP
```

**심화 주제:**
- Wasserstein Distance
- Gradient Penalty
- Progressive GAN (개념)
- StyleGAN (개념)

**과제:**
- WGAN-GP 직접 구현
- Mode Collapse 해결 실험

**체크포인트:** [퀴즈 8 - GAN 심화](#퀴즈-8-gan-심화)

#### Week 5-7: Diffusion Models (21시간)

**학습 자료:**
- `docs/phase4-diffusion-detailed.md`
- `docs/masters-level/diffusion-theory.md`
- 논문: DDPM, DDIM

**실습:**
```python
# DDPM 이해
cd code/diffusion
# 코드 분석: noise_schedule.py, ddpm.py, ddim.py

# 학습 (작은 데이터셋부터)
cd code/experiments
python run_diffusion_experiments.py --experiments 1 2
```

**심화 주제:**
- Forward/Reverse Process 수학
- Score Matching
- DDIM 빠른 샘플링
- Classifier-Free Guidance

**과제:**
- DDPM 처음부터 구현
- DDIM 50 steps vs DDPM 1000 steps 비교

**체크포인트:** [퀴즈 9 - Diffusion](#퀴즈-9-diffusion-models)

#### Week 8-9: Normalizing Flows (14시간)

**학습 자료:**
- `docs/phase3-flows-detailed.md`
- 논문: RealNVP, Glow

**실습:**
```python
# 2D Toy Data로 시작
# RealNVP 구현 (문서 참고)

# Coupling Layer 이해
```

**심화 주제:**
- Change of Variables Theorem
- Jacobian Determinant
- Affine Coupling Layer

**과제:**
- 2D Gaussian Mixture 학습
- MNIST로 RealNVP 학습

**체크포인트:** [퀴즈 10 - Flows](#퀴즈-10-normalizing-flows)

#### Week 10-11: Transformer 생성 모델 (14시간)

**학습 자료:**
- `docs/masters-level/transformer-generative-theory.md`
- 논문: "Attention Is All You Need", GPT-2

**실습:**
```python
# Mini GPT 구현 (문서의 코드)
# Character-level Language Model
```

**심화 주제:**
- Autoregressive Modeling
- Self-Attention
- In-Context Learning

**과제:**
- GPT로 텍스트 생성
- PixelCNN 개념 이해

**체크포인트:** [퀴즈 11 - Transformer](#퀴즈-11-transformer-생성-모델)

#### Week 12: 프로젝트 (14시간)

**프로젝트 선택:**
1. Custom Diffusion (프로젝트 예제 참고)
2. Style Transfer (CycleGAN)
3. Text Generation (GPT fine-tuning)

**deliverable:**
- GitHub repo
- README with results
- Gradio demo (선택)

**체크포인트:** [최종 평가 - 중급](#최종-평가-중급)

### 주간 학습 시간

**총 학습 시간: 126시간 (주당 10.5시간 × 12주)**

---

## 4. 고급 경로 (Advanced)

> **대상**: 생성 모델 경험자, 최신 연구 적용 및 실전 프로젝트
>
> **목표**: State-of-the-art 모델 구현 및 배포
>
> **예상 기간**: 2-3개월

### 학습 경로

```
Week 1-2: Stable Diffusion 심화
    ↓
Week 3-4: LLM Fine-tuning (LoRA, RLHF)
    ↓
Week 5-6: Multimodal Models
    ↓
Week 7-8: 최적화 & 배포
    ↓
Week 9-12: 대형 프로젝트
```

### 상세 일정표

#### Week 1-2: Stable Diffusion 심화 (14시간)

**학습 자료:**
- Stable Diffusion 논문
- ControlNet, LoRA for Diffusion

**실습:**
```python
# Hugging Face Diffusers
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Fine-tuning (DreamBooth, LoRA)
```

**과제:**
- Custom dataset으로 fine-tuning
- ControlNet 사용

#### Week 3-4: LLM Fine-tuning (14시간)

**학습 자료:**
- LoRA 논문
- RLHF 개념

**실습:**
```python
# LLaMA 2 + LoRA
from peft import LoraConfig, get_peft_model

# Fine-tuning
```

**과제:**
- 도메인 특화 LLM fine-tuning
- RAG 시스템 구축 (프로젝트 예제)

#### Week 5-6: Multimodal Models (14시간)

**학습 자료:**
- CLIP, LLaVA 논문

**실습:**
```python
# CLIP zero-shot classification
# LLaVA for VQA
```

#### Week 7-8: 최적화 & 배포 (14시간)

**학습 자료:**
- `docs/OPTIMIZATION-GUIDE.md`

**실습:**
```python
# Mixed Precision Training
# Distributed Training (DDP)
# Quantization
# TorchScript/ONNX
```

**과제:**
- 모델을 TensorRT로 최적화
- Gradio 앱 배포

#### Week 9-12: 대형 프로젝트 (28시간)

**프로젝트 예시:**
1. Production-ready Image Generation System
2. Multi-lingual Chatbot (RAG + LLM)
3. Video Generation Pipeline

**요구사항:**
- GitHub repo with CI/CD
- Docker containerization
- API 서버 (FastAPI)
- Monitoring (Prometheus, Grafana)
- Documentation

**체크포인트:** [최종 평가 - 고급](#최종-평가-고급)

### 주간 학습 시간

**총 학습 시간: 112시간 (주당 9.3시간 × 12주)**

---

## 5. 체크포인트 퀴즈

### 퀴즈 1: Python & 수학 기초

**문제 1:** NumPy로 3×3 단위 행렬을 만드는 방법은?
<details>
<summary>정답 보기</summary>

```python
np.eye(3)
# 또는
np.identity(3)
```
</details>

**문제 2:** f(x) = x² 의 도함수는?
<details>
<summary>정답 보기</summary>

f'(x) = 2x
</details>

**문제 3:** 확률 변수 X ~ N(0, 1)의 기댓값과 분산은?
<details>
<summary>정답 보기</summary>

E[X] = 0, Var[X] = 1
</details>

---

### 퀴즈 2: PyTorch 기초

**문제 1:** Tensor의 gradient를 계산하려면 어떤 속성을 True로 설정해야 하나?
<details>
<summary>정답 보기</summary>

```python
x = torch.randn(3, 4, requires_grad=True)
```
</details>

**문제 2:** loss.backward() 후 optimizer.step() 전에 반드시 해야 할 것은?
<details>
<summary>정답 보기</summary>

```python
optimizer.zero_grad()
```
(이전 gradient 초기화)
</details>

---

### 퀴즈 3: CNN

**문제 1:** Convolution의 출력 크기 공식은? (padding=0, stride=1)
<details>
<summary>정답 보기</summary>

Output = Input - Kernel + 1

예: (28, 28) 이미지에 (3, 3) kernel → (26, 26)
</details>

**문제 2:** Max Pooling의 역할은?
<details>
<summary>정답 보기</summary>

- Spatial dimension 축소
- Translation invariance
- 파라미터 없음
- 가장 강한 activation 선택
</details>

---

### 퀴즈 4: Autoencoder

**문제 1:** Autoencoder의 손실 함수는?
<details>
<summary>정답 보기</summary>

Reconstruction Loss:
```
L = ||x - decoder(encoder(x))||²
```
보통 MSE Loss 사용
</details>

**문제 2:** Latent space의 차원이 입력보다 작은 이유는?
<details>
<summary>정답 보기</summary>

- 차원 축소 (dimensionality reduction)
- 중요한 feature만 학습
- Regularization 효과
- Overcomplete 방지
</details>

---

### 퀴즈 5: VAE

**문제 1:** ELBO 공식을 쓰시오.
<details>
<summary>정답 보기</summary>

$$
\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

또는:
```
ELBO = Reconstruction Loss - KL Divergence
```
</details>

**문제 2:** Reparameterization Trick이 필요한 이유는?
<details>
<summary>정답 보기</summary>

- Sampling operation은 미분 불가능
- z = μ + σ ⊙ ε (ε ~ N(0,1))로 변환
- ε는 상수처럼 취급하여 gradient가 μ, σ로 흐르게 함
</details>

**문제 3:** β-VAE의 β가 커지면 어떻게 되나?
<details>
<summary>정답 보기</summary>

- KL divergence에 더 큰 가중치
- Reconstruction quality 감소
- Disentanglement 증가
- Latent space가 더 prior에 가까워짐
</details>

---

### 퀴즈 6: GAN 기초

**문제 1:** GAN의 minimax 목적 함수를 쓰시오.
<details>
<summary>정답 보기</summary>

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]
$$
</details>

**문제 2:** Mode Collapse란?
<details>
<summary>정답 보기</summary>

Generator가 다양한 샘플을 생성하지 못하고 특정 패턴만 반복하는 현상
</details>

**문제 3:** DCGAN의 5가지 아키텍처 규칙 중 3가지는?
<details>
<summary>정답 보기</summary>

1. Pooling 제거 (Strided Conv 사용)
2. BatchNorm 사용
3. FC layer 제거
4. Generator: ReLU + Tanh
5. Discriminator: LeakyReLU
</details>

---

### 퀴즈 7: VAE 심화

**문제 1:** Posterior Collapse란?
<details>
<summary>정답 보기</summary>

- q(z|x)가 p(z)와 동일해지는 현상
- KL divergence → 0
- Latent code가 사용되지 않음
- Decoder가 latent 무시하고 marginal distribution 학습
</details>

---

### 퀴즈 8: GAN 심화

**문제 1:** Wasserstein Distance의 장점은?
<details>
<summary>정답 보기</summary>

- 분포가 겹치지 않아도 의미있는 gradient
- 학습 안정성 향상
- Mode collapse 감소
- Loss가 실제 거리를 반영
</details>

**문제 2:** Gradient Penalty의 공식은?
<details>
<summary>정답 보기</summary>

$$
\lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$

여기서 $\hat{x} = \epsilon x + (1-\epsilon) G(z)$
</details>

---

### 퀴즈 9: Diffusion Models

**문제 1:** DDPM의 학습 목적 함수는?
<details>
<summary>정답 보기</summary>

$$
\mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]
$$

Noise prediction objective
</details>

**문제 2:** DDIM이 DDPM보다 빠른 이유는?
<details>
<summary>정답 보기</summary>

- Non-Markovian process
- Skip steps 가능 (1000 → 50)
- Deterministic sampling (η=0)
- 품질 유지하면서 20배 빠름
</details>

---

### 퀴즈 10: Normalizing Flows

**문제 1:** Change of Variables 공식은?
<details>
<summary>정답 보기</summary>

$$
p_x(x) = p_z(f^{-1}(x)) \left|\det \frac{\partial f^{-1}}{\partial x}\right|
$$

또는 log 형태:
$$
\log p_x(x) = \log p_z(z) - \log \left|\det \frac{\partial f}{\partial z}\right|
$$
</details>

**문제 2:** Coupling Layer에서 왜 일부는 그대로 두나?
<details>
<summary>정답 보기</summary>

- Invertibility 보장
- Jacobian이 triangular해짐 (det 계산 O(d))
- x₁은 identity, x₂만 변환
- Mask를 교대로 사용하여 모든 차원 변환
</details>

---

### 퀴즈 11: Transformer 생성 모델

**문제 1:** Self-Attention의 Query, Key, Value는 무엇인가?
<details>
<summary>정답 보기</summary>

```
Q = X W^Q  (현재 위치의 query)
K = X W^K  (모든 위치의 key)
V = X W^V  (모든 위치의 value)

Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
</details>

**문제 2:** Causal Masking이 필요한 이유는?
<details>
<summary>정답 보기</summary>

- Autoregressive generation에서 미래 토큰을 보면 안됨
- 학습 시 teacher forcing 사용하므로 mask 필요
- 삼각 행렬로 구현 (upper triangle = -∞)
</details>

---

### 최종 평가: 초급

**프로젝트 평가 기준:**
- [ ] 코드 실행 가능 (에러 없음)
- [ ] README 작성 (목표, 방법, 결과)
- [ ] 시각화 (샘플 이미지, loss curve)
- [ ] 결과 분석 (정량적 + 정성적)
- [ ] 코드 주석 (핵심 부분)

**합격 기준:** 5개 중 4개 이상

---

### 최종 평가: 중급

**프로젝트 평가 기준:**
- [ ] 초급 기준 모두 충족
- [ ] 고급 기법 사용 (AMP, Gradient Accumulation 등)
- [ ] 평가 지표 계산 (FID, IS)
- [ ] Ablation study
- [ ] GitHub repo 공개

**합격 기준:** 5개 모두

---

### 최종 평가: 고급

**프로젝트 평가 기준:**
- [ ] 중급 기준 모두 충족
- [ ] Production-ready (Docker, API, Monitoring)
- [ ] 최적화 (TensorRT, Quantization)
- [ ] 문서화 (API docs, Tutorial)
- [ ] 배포 (Hugging Face Spaces, 또는 클라우드)

**합격 기준:** 5개 모두

---

## 6. 학습 로드맵 시각화

### 초급 → 중급 → 고급 전체 경로

```
                    입문
                     |
            [Python & 수학 기초]
                     |
                [PyTorch 기초]
                     |
              [신경망 기초 (CNN)]
                     |
        ┌────────────┴────────────┐
        |                         |
   [Autoencoder]              [초급 완료]
        |                         |
      [VAE]                  [프로젝트 1]
        |
    [GAN 기초]
        |
        └────────────┬────────────┘
                     |
                 [중급 시작]
                     |
        ┌────────────┼────────────┐
        |            |            |
   [VAE 심화]   [GAN 심화]  [Diffusion]
        |            |            |
        └────────────┼────────────┘
                     |
           [Normalizing Flows]
                     |
            [Transformer 생성]
                     |
               [프로젝트 2]
                     |
                 [고급 시작]
                     |
        ┌────────────┼────────────┐
        |            |            |
  [Stable Diffusion] [LLM] [Multimodal]
        |            |            |
        └────────────┼────────────┘
                     |
             [최적화 & 배포]
                     |
              [대형 프로젝트]
                     |
                  [완료!]
```

### 모델별 학습 순서

```
생성 모델 계보:

Autoencoder (가장 기초)
    ↓
VAE (확률적 생성)
    ↓
    ├─→ GAN (적대적 학습)
    │      ↓
    │   DCGAN → WGAN → StyleGAN
    │
    ├─→ Normalizing Flows (정확한 likelihood)
    │      ↓
    │   NICE → RealNVP → Glow
    │
    └─→ Diffusion Models (노이즈 제거)
           ↓
        DDPM → DDIM → Stable Diffusion

Transformer (별도 계보)
    ↓
GPT (Autoregressive)
    ↓
DALL-E, Imagen (Text-to-Image)
```

---

## 📚 추가 리소스

### 학습 커뮤니티
- Reddit: r/MachineLearning
- Discord: PyTorch, Hugging Face
- 논문 스터디 그룹

### 추천 도구
- **Colab/Kaggle**: GPU 무료
- **Weights & Biases**: 실험 추적
- **Hugging Face**: 모델 공유

---

**최종 업데이트**: 2025-11-15
**라이선스**: MIT
