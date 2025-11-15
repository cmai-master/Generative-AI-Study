# GAN (Generative Adversarial Networks) 중급 가이드

> **학습 목표**: GAN의 핵심 원리를 완전히 이해하고, DCGAN과 WGAN-GP를 직접 구현할 수 있다
>
> **난이도**: 중급 (Intermediate)
> **예상 학습 시간**: 2-3주
> **선수 지식**: PyTorch 기초, CNN, 확률론 기초

---

## 📚 목차

1. [GAN 기초 복습](#1-gan-기초-복습)
2. [DCGAN 완전 정복](#2-dcgan-완전-정복)
3. [Wasserstein GAN (WGAN)](#3-wasserstein-gan-wgan)
4. [WGAN-GP 구현](#4-wgan-gp-구현)
5. [Mode Collapse 문제와 해결](#5-mode-collapse-문제와-해결)
6. [학습 안정화 기법](#6-학습-안정화-기법)
7. [평가 지표](#7-평가-지표)
8. [실전 학습 팁](#8-실전-학습-팁)
9. [실습 프로젝트](#9-실습-프로젝트)

---

## 1. GAN 기초 복습

### 1.1 GAN이란?

GAN(Generative Adversarial Network)은 2014년 Ian Goodfellow가 제안한 생성 모델입니다. 핵심 아이디어는 **두 개의 신경망을 경쟁시켜 학습**하는 것입니다.

#### 두 플레이어 게임

```
Generator (G)    vs    Discriminator (D)
    |                        |
    |---> 가짜 이미지 생성    |
                             |---> 진짜/가짜 판별
```

- **Generator (생성자)**: 랜덤 노이즈 z를 입력받아 가짜 이미지 G(z)를 생성
- **Discriminator (판별자)**: 이미지를 받아 진짜(1) 또는 가짜(0)로 분류

### 1.2 수학적 정의

GAN의 목적 함수는 **Minimax Game**으로 표현됩니다:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

**해석:**
- **D의 목표 (Maximize)**:
  - 진짜 이미지 x에 대해 D(x) → 1 (높은 확률)
  - 가짜 이미지 G(z)에 대해 D(G(z)) → 0 (낮은 확률)
- **G의 목표 (Minimize)**:
  - 가짜 이미지가 D를 속이도록: D(G(z)) → 1

### 1.3 학습 알고리즘

```python
for epoch in range(num_epochs):
    for real_images in dataloader:
        # 1. Discriminator 학습
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)

        real_loss = -log(D(real_images))      # 진짜를 진짜로
        fake_loss = -log(1 - D(fake_images))  # 가짜를 가짜로
        d_loss = real_loss + fake_loss

        update_D(d_loss)

        # 2. Generator 학습
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)
        g_loss = -log(D(fake_images))  # 가짜를 진짜처럼

        update_G(g_loss)
```

### 1.4 왜 GAN이 어려운가?

**주요 문제점:**

1. **Mode Collapse**: Generator가 다양성을 잃고 특정 샘플만 생성
2. **학습 불안정성**: D와 G의 균형이 깨지면 학습 실패
3. **Vanishing Gradient**: D가 너무 강해지면 G가 학습 못함
4. **평가 어려움**: 생성 품질을 정량적으로 측정하기 어려움

이제 이 문제들을 해결하는 **DCGAN**과 **WGAN-GP**를 배워봅시다!

---

## 2. DCGAN 완전 정복

### 2.1 DCGAN이란?

**Deep Convolutional GAN (DCGAN)**은 2015년 Radford 등이 제안한 CNN 기반 GAN입니다.

**핵심 기여:**
- GAN 학습을 안정화하는 아키텍처 가이드라인 제시
- 고품질 이미지 생성 가능
- Latent space에 의미있는 구조 발견

### 2.2 DCGAN 아키텍처 가이드라인

#### 5가지 핵심 규칙

1. **Pooling 층 제거**
   - Max Pooling 대신 **Strided Convolution** 사용 (D)
   - **Transposed Convolution** 사용 (G)

2. **Batch Normalization 사용**
   - Generator: 모든 층에 BatchNorm (출력 층 제외)
   - Discriminator: 모든 층에 BatchNorm (입력 층 제외)

3. **Fully Connected 층 제거**
   - 완전 연결 층을 제거하고 전부 Convolutional 층으로

4. **활성화 함수**
   - Generator: ReLU (출력 층만 Tanh)
   - Discriminator: LeakyReLU (모든 층)

5. **가중치 초기화**
   - 평균 0, 표준편차 0.02인 정규분포

### 2.3 Generator 아키텍처

#### 개념도

```
z (100,)
    ↓
Linear + Reshape → (512, 4, 4)
    ↓
ConvTranspose2d (stride=2) → (256, 8, 8)
    ↓ BatchNorm + ReLU
ConvTranspose2d (stride=2) → (128, 16, 16)
    ↓ BatchNorm + ReLU
ConvTranspose2d (stride=2) → (64, 32, 32)
    ↓ BatchNorm + ReLU
ConvTranspose2d (stride=2) → (3, 64, 64)
    ↓ Tanh
Generated Image (3, 64, 64)
```

#### PyTorch 구현

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, feature_map_size=64, num_channels=3):
        """
        DCGAN Generator

        Args:
            latent_dim: 잠재 벡터 차원 (보통 100)
            feature_map_size: Feature map 기본 크기 (보통 64)
            num_channels: 출력 이미지 채널 (RGB=3, Grayscale=1)
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        ngf = feature_map_size  # Number of Generator Features

        # 1. Linear Projection: z(100) → (512*4*4)
        self.project = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # 2. Transposed Convolutions
        self.convs = nn.Sequential(
            # (512, 4, 4) → (256, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (256, 8, 8) → (128, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (128, 16, 16) → (64, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (64, 32, 32) → (3, 64, 64)
            nn.ConvTranspose2d(ngf, num_channels,
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # 출력 범위: [-1, 1]
        )

        # 가중치 초기화
        self.apply(weights_init)

    def forward(self, z):
        """
        Args:
            z: (batch_size, latent_dim) 랜덤 노이즈
        Returns:
            img: (batch_size, num_channels, 64, 64) 생성된 이미지
        """
        # Linear projection
        x = self.project(z)  # (B, 512*4*4)
        x = x.view(-1, 512, 4, 4)  # (B, 512, 4, 4)

        # Transposed convolutions
        img = self.convs(x)  # (B, 3, 64, 64)

        return img

def weights_init(m):
    """DCGAN 논문의 가중치 초기화"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

#### Transposed Convolution 이해하기

**일반 Convolution (Downsampling):**
```
(64, 64) --[Conv, stride=2]--> (32, 32)
```

**Transposed Convolution (Upsampling):**
```
(32, 32) --[ConvTranspose, stride=2]--> (64, 64)
```

**계산 공식:**
$$
\text{output\_size} = (\text{input\_size} - 1) \times \text{stride} + \text{kernel\_size} - 2 \times \text{padding}
$$

**예시:**
```python
# (4, 4) → (8, 8)
# output = (4 - 1) * 2 + 4 - 2 * 1 = 3 * 2 + 4 - 2 = 8
nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
```

### 2.4 Discriminator 아키텍처

#### 개념도

```
Image (3, 64, 64)
    ↓
Conv2d (stride=2) → (64, 32, 32)
    ↓ LeakyReLU
Conv2d (stride=2) → (128, 16, 16)
    ↓ BatchNorm + LeakyReLU
Conv2d (stride=2) → (256, 8, 8)
    ↓ BatchNorm + LeakyReLU
Conv2d (stride=2) → (512, 4, 4)
    ↓ BatchNorm + LeakyReLU
Conv2d → (1, 1, 1)
    ↓ Sigmoid
Probability (진짜일 확률)
```

#### PyTorch 구현

```python
class Discriminator(nn.Module):
    def __init__(self, feature_map_size=64, num_channels=3):
        """
        DCGAN Discriminator

        Args:
            feature_map_size: Feature map 기본 크기
            num_channels: 입력 이미지 채널
        """
        super(Discriminator, self).__init__()

        ndf = feature_map_size  # Number of Discriminator Features

        self.main = nn.Sequential(
            # (3, 64, 64) → (64, 32, 32)
            # 첫 층은 BatchNorm 없음!
            nn.Conv2d(num_channels, ndf,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(ndf, ndf * 2,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (256, 8, 8) → (512, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (512, 4, 4) → (1, 1, 1)
            nn.Conv2d(ndf * 8, 1,
                     kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # 출력: 0~1 (진짜일 확률)
        )

        self.apply(weights_init)

    def forward(self, img):
        """
        Args:
            img: (batch_size, num_channels, 64, 64)
        Returns:
            prob: (batch_size, 1, 1, 1) 진짜일 확률
        """
        prob = self.main(img)
        return prob.view(-1, 1)  # (B, 1)
```

### 2.5 DCGAN 학습

#### 손실 함수

```python
import torch.nn as nn
import torch.optim as optim

# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Optimizers (Adam with β1=0.5)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
```

#### 학습 루프

```python
import torch
from torchvision.utils import save_image

def train_dcgan(generator, discriminator, dataloader, num_epochs=100, device='cuda'):
    """
    DCGAN 학습 함수
    """
    generator.to(device)
    discriminator.to(device)

    # 고정된 노이즈 (샘플링용)
    fixed_noise = torch.randn(64, latent_dim, device=device)

    # 레이블
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # ========== Discriminator 학습 ==========
            discriminator.zero_grad()

            # 1-1. 진짜 이미지
            label = torch.full((batch_size, 1), real_label, device=device)
            output = discriminator(real_images)
            d_loss_real = criterion(output, label)
            d_loss_real.backward()

            # 1-2. 가짜 이미지
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_images.detach())  # detach() 중요!
            d_loss_fake = criterion(output, label)
            d_loss_fake.backward()

            # Discriminator 업데이트
            d_loss = d_loss_real + d_loss_fake
            optimizer_D.step()

            # ========== Generator 학습 ==========
            generator.zero_grad()

            label.fill_(real_label)  # Generator는 D를 속이려고 함
            output = discriminator(fake_images)  # 이번엔 detach 안함!
            g_loss = criterion(output, label)
            g_loss.backward()

            # Generator 업데이트
            optimizer_G.step()

            # ========== 로깅 ==========
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f} '
                      f'D(x): {output.mean().item():.4f}')

        # Epoch마다 샘플 저장
        with torch.no_grad():
            fake = generator(fixed_noise)
            save_image(fake, f'samples/epoch_{epoch}.png', normalize=True)

        # 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch
            }, f'checkpoints/dcgan_epoch_{epoch}.pth')
```

#### 중요한 학습 팁

1. **detach() 사용**
   ```python
   # Discriminator 학습 시
   fake_images = generator(noise)
   output = discriminator(fake_images.detach())  # ✅ detach() 필수!
   # detach()를 하지 않으면 Generator까지 gradient가 흘러감
   ```

2. **레이블 스무딩 (Label Smoothing)**
   ```python
   # 원래: real_label = 1.0, fake_label = 0.0
   # 스무딩: real_label = 0.9, fake_label = 0.1
   # 학습을 더 안정적으로 만듦
   real_label = 0.9
   fake_label = 0.1
   ```

3. **학습률 조정**
   ```python
   # D가 너무 강하면 G 학습 안됨
   # G 학습률을 D보다 높게 설정할 수도 있음
   optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
   optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)
   ```

### 2.6 DCGAN 결과 분석

#### 성공 지표

1. **Loss가 균형있게 유지**
   - D_loss와 G_loss가 모두 log(2) ≈ 0.693 근처
   - 한쪽이 0에 가까우면 실패

2. **D(x)와 D(G(z)) 추이**
   - D(x) ≈ 0.7~0.8 (진짜를 진짜로 잘 판별)
   - D(G(z)) ≈ 0.3~0.4 (가짜를 어느 정도 속임)

3. **시각적 품질**
   - Epoch이 지날수록 이미지가 선명해짐
   - 다양한 샘플 생성

#### 실패 사례

**Mode Collapse:**
```
Epoch 1-10: 다양한 얼굴
Epoch 11-20: 비슷한 얼굴만 반복
```
→ Section 5에서 해결법 다룸

---

## 3. Wasserstein GAN (WGAN)

### 3.1 기존 GAN의 문제점

#### Vanishing Gradient 문제

기존 GAN의 Generator 손실:
$$
L_G = \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

**문제:** D가 완벽해지면 (D(G(z)) → 0):
$$
\log(1 - 0) = \log(1) = 0 \implies \text{Gradient vanishes!}
$$

#### 해결책: Non-Saturating Loss

$$
L_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$

하지만 여전히 학습이 불안정!

### 3.2 Wasserstein Distance

WGAN의 핵심 아이디어: **더 나은 거리 측정 방법 사용**

#### Earth Mover's Distance (EMD)

두 분포 P_r (real)과 P_g (generated) 사이의 **Wasserstein-1 distance**:

$$
W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]
$$

**직관:**
- 흙더미 P_r을 P_g로 옮기는 최소 비용
- "지구를 움직이는 거리"라는 의미

#### JS Divergence vs Wasserstein Distance

**Jensen-Shannon Divergence (기존 GAN):**
- 분포가 겹치지 않으면 상수
- Gradient가 사라짐

**Wasserstein Distance (WGAN):**
- 분포가 떨어져 있어도 의미있는 gradient
- 학습 안정성 ↑

### 3.3 WGAN의 목적 함수

**Kantorovich-Rubinstein Duality**를 이용하면:

$$
W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)]
$$

여기서 f는 **1-Lipschitz 함수**.

WGAN의 목적 함수:

$$
\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

**주의:**
- D는 이제 **Critic**이라 부름 (확률이 아닌 점수 출력)
- Sigmoid 없음!

### 3.4 1-Lipschitz Constraint

**Lipschitz 연속 조건:**
$$
|f(x_1) - f(x_2)| \leq K \|x_1 - x_2\|
$$

K=1일 때 "1-Lipschitz"

**WGAN의 해결책: Weight Clipping**
```python
# 각 업데이트 후
for p in critic.parameters():
    p.data.clamp_(-0.01, 0.01)
```

**문제점:**
- Gradient가 explode/vanish할 수 있음
- 표현력 제한

→ WGAN-GP에서 해결!

---

## 4. WGAN-GP 구현

### 4.1 Gradient Penalty

**WGAN-GP의 아이디어:** Weight clipping 대신 **Gradient Penalty** 사용

#### Gradient Penalty 정의

$$
\text{GP} = \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$

여기서 $\hat{x}$는 real과 fake 사이의 **보간(interpolation)**:
$$
\hat{x} = \epsilon x + (1 - \epsilon) G(z), \quad \epsilon \sim U[0, 1]
$$

**직관:**
- Gradient의 norm이 1에 가깝도록 regularize
- 1-Lipschitz 조건을 soft하게 만족

### 4.2 WGAN-GP 목적 함수

$$
L_D = \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})] - \mathbb{E}_{x \sim P_r}[D(x)] + \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$

$$
L_G = -\mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})]
$$

### 4.3 Critic (Discriminator) 구현

```python
class Critic(nn.Module):
    def __init__(self, feature_map_size=64, num_channels=3):
        """
        WGAN-GP Critic (DCGAN과 거의 동일)

        차이점:
        1. 출력에 Sigmoid 없음 (점수 출력)
        2. BatchNorm 사용하지 않음 (LayerNorm 또는 없음)
        """
        super(Critic, self).__init__()

        ndf = feature_map_size

        self.main = nn.Sequential(
            # (3, 64, 64) → (64, 32, 32)
            nn.Conv2d(num_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LayerNorm([ndf * 2, 16, 16]),  # LayerNorm 사용
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LayerNorm([ndf * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),

            # (256, 8, 8) → (512, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LayerNorm([ndf * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),

            # (512, 4, 4) → (1, 1, 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            # Sigmoid 없음!
        )

    def forward(self, img):
        return self.main(img).view(-1, 1)
```

**왜 BatchNorm을 쓰지 않나?**
- BatchNorm은 배치 간 의존성을 만듦
- Gradient Penalty 계산에 문제 발생
- LayerNorm 또는 InstanceNorm 사용

### 4.4 Gradient Penalty 계산

```python
def compute_gradient_penalty(critic, real_images, fake_images, device):
    """
    WGAN-GP의 Gradient Penalty 계산

    Args:
        critic: Critic 네트워크
        real_images: 진짜 이미지 (B, C, H, W)
        fake_images: 가짜 이미지 (B, C, H, W)
        device: 'cuda' or 'cpu'

    Returns:
        gradient_penalty: GP 값 (scalar)
    """
    batch_size = real_images.size(0)

    # 1. 랜덤 보간 계수 생성
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # 2. 보간된 이미지 생성
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated = interpolated.requires_grad_(True)  # gradient 계산을 위해

    # 3. Critic 평가
    critic_interpolated = critic(interpolated)

    # 4. Gradient 계산
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,  # 2차 미분을 위해 필요
        retain_graph=True,
        only_inputs=True
    )[0]

    # 5. Gradient의 L2 norm 계산
    gradients = gradients.view(batch_size, -1)  # (B, C*H*W)
    gradient_norm = gradients.norm(2, dim=1)    # (B,)

    # 6. Gradient Penalty
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty
```

**중요 포인트:**

1. **create_graph=True**: 2차 미분을 위해 필요
2. **보간 이미지**: Real과 fake 사이의 선형 보간
3. **Gradient norm**: L2 norm을 1에 가깝게

### 4.5 WGAN-GP 학습 루프

```python
def train_wgan_gp(generator, critic, dataloader,
                  num_epochs=100,
                  n_critic=5,
                  lambda_gp=10,
                  device='cuda'):
    """
    WGAN-GP 학습

    Args:
        n_critic: Critic을 몇 번 업데이트할지 (보통 5)
        lambda_gp: Gradient Penalty 가중치 (보통 10)
    """
    generator.to(device)
    critic.to(device)

    # Optimizers (RMSprop 또는 Adam)
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.0, 0.9))

    fixed_noise = torch.randn(64, latent_dim, device=device)

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # ========== Critic 학습 (n_critic 번) ==========
            for _ in range(n_critic):
                critic.zero_grad()

                # 1. Real images
                critic_real = critic(real_images)
                loss_real = -critic_real.mean()  # Maximize D(x)

                # 2. Fake images
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake_images = generator(noise).detach()
                critic_fake = critic(fake_images)
                loss_fake = critic_fake.mean()  # Minimize D(G(z))

                # 3. Gradient Penalty
                gp = compute_gradient_penalty(critic, real_images, fake_images, device)

                # 4. Total Critic Loss
                critic_loss = loss_real + loss_fake + lambda_gp * gp

                critic_loss.backward()
                optimizer_C.step()

            # ========== Generator 학습 ==========
            generator.zero_grad()

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            gen_fake = critic(fake_images)
            gen_loss = -gen_fake.mean()  # Maximize D(G(z))

            gen_loss.backward()
            optimizer_G.step()

            # ========== 로깅 ==========
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_C: {critic_loss.item():.4f} '
                      f'Loss_G: {gen_loss.item():.4f} '
                      f'D(x): {-loss_real.item():.4f} '
                      f'D(G(z)): {loss_fake.item():.4f} '
                      f'GP: {gp.item():.4f}')

        # 샘플 저장
        with torch.no_grad():
            fake = generator(fixed_noise)
            save_image(fake, f'samples_wgan/epoch_{epoch}.png', normalize=True)
```

### 4.6 DCGAN vs WGAN-GP 비교

| 특징 | DCGAN | WGAN-GP |
|------|-------|---------|
| **손실 함수** | Binary Cross-Entropy | Wasserstein Distance |
| **Discriminator 출력** | 확률 (Sigmoid) | 점수 (linear) |
| **BatchNorm** | 사용 (D 제외 첫 층) | 사용 안함 (LayerNorm) |
| **Critic 업데이트** | 1회 | 5회 (n_critic) |
| **학습 안정성** | 중간 | 높음 ⭐ |
| **Mode Collapse** | 발생 가능 | 적음 ⭐ |
| **학습 속도** | 빠름 | 느림 (5배 Critic) |

**언제 WGAN-GP를 써야 하나?**
- ✅ 학습 안정성이 중요할 때
- ✅ Mode collapse가 심할 때
- ✅ Loss를 신뢰할 수 있는 지표로 쓰고 싶을 때
- ❌ 빠른 프로토타이핑 (DCGAN이 더 빠름)

---

## 5. Mode Collapse 문제와 해결

### 5.1 Mode Collapse란?

**정의:** Generator가 다양한 샘플을 생성하지 못하고 특정 패턴만 반복

#### 예시

**정상:**
```
생성된 얼굴: 👨 👩 👴 👧 👦 🧔 👱 ...
```

**Mode Collapse:**
```
생성된 얼굴: 👨 👨 👨 👨 👨 👨 👨 ...
```

### 5.2 왜 발생하나?

1. **Generator의 최적화 문제**
   - G가 D를 속이기 쉬운 특정 샘플만 생성
   - 다양성보다 "D를 속이기"에만 집중

2. **Discriminator가 너무 강함**
   - D가 완벽해지면 G가 학습 못함
   - G가 local minimum에 빠짐

3. **목적 함수의 한계**
   - JS Divergence는 다양성을 직접 측정하지 않음

### 5.3 Mode Collapse 감지

#### 1. 시각적 검사
```python
# 고정된 노이즈로 여러 epoch 샘플 비교
fixed_noise = torch.randn(100, latent_dim)

for epoch in [10, 20, 30, 40, 50]:
    samples = generator(fixed_noise)
    save_image(samples, f'epoch_{epoch}.png')
# 점점 비슷해지면 Mode Collapse!
```

#### 2. 정량적 측정
```python
def measure_diversity(samples):
    """
    생성된 샘플의 다양성 측정 (간단한 방법)
    """
    # 1. 샘플 간 평균 거리 계산
    samples_flat = samples.view(samples.size(0), -1)
    distances = torch.cdist(samples_flat, samples_flat)
    avg_distance = distances.sum() / (len(samples) * (len(samples) - 1))

    return avg_distance.item()

# 사용
diversity = measure_diversity(generated_samples)
print(f"Diversity: {diversity}")  # 낮아지면 Mode Collapse
```

### 5.4 해결 방법

#### 방법 1: WGAN-GP 사용

이미 다뤘듯이, WGAN-GP는 Mode Collapse를 크게 줄입니다.

#### 방법 2: Minibatch Discrimination

```python
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        # x: (batch_size, in_features)
        matrices = torch.mm(x, self.T.view(self.T.size(0), -1))
        matrices = matrices.view(-1, self.T.size(1), self.T.size(2))

        # 배치 내 샘플 간 유사도 계산
        M = matrices.unsqueeze(0)  # (1, batch, out, kernel)
        M_T = M.permute(1, 0, 2, 3)  # (batch, 1, out, kernel)

        # L1 distance
        out = torch.sum(torch.abs(M - M_T), dim=3)  # (batch, batch, out)
        out = torch.sum(torch.exp(-out), dim=0)  # (batch, out)

        return torch.cat([x, out], dim=1)
```

**아이디어:** D가 개별 샘플뿐만 아니라 배치 전체의 다양성도 고려

#### 방법 3: Unrolled GAN

```python
# Discriminator 업데이트를 k번 "미리" 해보고 그 결과로 G 업데이트
# 구현이 복잡하지만 효과적
```

#### 방법 4: Feature Matching

```python
# Generator 손실을 출력 확률이 아닌 중간 feature의 유사도로
def feature_matching_loss(real_images, fake_images, discriminator):
    # Discriminator의 중간 layer 출력 사용
    real_features = discriminator.get_features(real_images)
    fake_features = discriminator.get_features(fake_images)

    # Feature의 평균 차이 최소화
    loss = torch.mean((real_features.mean(0) - fake_features.mean(0)) ** 2)
    return loss
```

#### 방법 5: 다양한 Latent Code 사용

```python
# 단순 Gaussian 대신 다양한 분포 시도
z = torch.randn(batch_size, latent_dim)  # Gaussian

# 또는
z = torch.rand(batch_size, latent_dim) * 2 - 1  # Uniform [-1, 1]

# 또는 Mixture of Gaussians
```

---

## 6. 학습 안정화 기법

### 6.1 Two Time-Scale Update Rule (TTUR)

D와 G의 학습 속도를 다르게:

```python
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.0, 0.9))
# D를 4배 빠르게
```

### 6.2 Label Smoothing

```python
# Hard labels
real_labels = 1.0
fake_labels = 0.0

# Smoothed labels
real_labels = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0)
fake_labels = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.2)
```

### 6.3 Noisy Labels

```python
# 레이블을 일정 확률로 뒤집기
if np.random.random() < 0.05:  # 5% 확률
    real_labels, fake_labels = fake_labels, real_labels
```

### 6.4 Spectral Normalization

```python
from torch.nn.utils import spectral_norm

# Discriminator의 모든 Conv layer에 적용
self.conv1 = spectral_norm(nn.Conv2d(3, 64, 4, 2, 1))
self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
# ...
```

**효과:** Lipschitz 상수를 제한하여 학습 안정화

---

## 7. 평가 지표

### 7.1 Inception Score (IS)

#### 정의

$$
IS(G) = \exp(\mathbb{E}_{x \sim P_g}[D_{KL}(p(y|x) \| p(y))])
$$

**해석:**
- p(y|x): 생성 이미지 x의 클래스 분포 (Inception 모델 사용)
- p(y): 전체 생성 이미지의 평균 클래스 분포

**좋은 생성 모델:**
- p(y|x)는 명확 (높은 품질) → entropy 낮음
- p(y)는 uniform (다양성) → entropy 높음

#### 구현

```python
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.stats import entropy

def calculate_inception_score(images, batch_size=32, splits=10):
    """
    Inception Score 계산

    Args:
        images: (N, 3, H, W) 생성된 이미지 (N >= 50000 권장)
        batch_size: 배치 크기
        splits: 평균을 낼 splits 수

    Returns:
        is_mean: IS 평균
        is_std: IS 표준편차
    """
    # Inception V3 모델 로드
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()
    inception_model.fc = torch.nn.Identity()  # 마지막 FC 제거

    # 이미지를 Inception 모델에 통과
    N = len(images)
    preds = []

    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            pred = inception_model(batch)
            preds.append(F.softmax(pred, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N, 1000)

    # Inception Score 계산
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)  # p(y)
        scores = []

        for i in range(part.shape[0]):
            pyx = part[i, :]  # p(y|x)
            scores.append(entropy(pyx, py))

        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
```

**좋은 IS:**
- ImageNet: 10+
- CIFAR-10: 8+
- MNIST: 3+

### 7.2 Fréchet Inception Distance (FID)

#### 정의

실제 이미지와 생성 이미지의 feature 분포 거리:

$$
FID = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

- μ: 평균
- Σ: 공분산 행렬
- Tr: Trace

**낮을수록 좋음!**

#### 구현

```python
import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(real_images, fake_images, inception_model):
    """
    FID 계산

    Args:
        real_images: (N, 3, H, W) 실제 이미지
        fake_images: (N, 3, H, W) 생성 이미지
        inception_model: Feature 추출용 모델

    Returns:
        fid: FID 점수 (낮을수록 좋음)
    """
    # 1. Feature 추출
    def get_features(images):
        with torch.no_grad():
            features = inception_model(images)
        return features.cpu().numpy()

    real_features = get_features(real_images)
    fake_features = get_features(fake_images)

    # 2. 평균과 공분산 계산
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # 3. FID 계산
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))

    # 수치 안정성
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return fid
```

**좋은 FID:**
- CelebA 64x64: < 10
- CIFAR-10: < 30
- 낮을수록 좋음

### 7.3 Precision and Recall

**Precision:** 생성 샘플이 real 분포에 얼마나 가까운가?
**Recall:** real 분포의 mode를 얼마나 커버하는가?

```python
def calculate_precision_recall(real_features, fake_features, k=3):
    """
    Precision and Recall for GANs
    """
    # k-NN 기반 계산
    # 구현 생략 (복잡)
    pass
```

---

## 8. 실전 학습 팁

### 8.1 하이퍼파라미터 선택

| 하이퍼파라미터 | 권장 값 | 설명 |
|--------------|--------|------|
| **Learning Rate** | 0.0002 | Adam with β1=0.5 |
| **Batch Size** | 64-128 | 클수록 안정적 |
| **Latent Dim** | 100 | 너무 작으면 다양성 ↓ |
| **Feature Map** | 64 | GPU 메모리에 따라 조정 |

### 8.2 학습 모니터링

```python
# TensorBoard 로깅
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/gan_experiment')

for epoch in range(num_epochs):
    # ...학습...

    writer.add_scalar('Loss/Generator', g_loss, global_step)
    writer.add_scalar('Loss/Discriminator', d_loss, global_step)
    writer.add_scalar('Scores/D(x)', d_x, global_step)
    writer.add_scalar('Scores/D(G(z))', d_g_z, global_step)

    # 이미지 로깅
    writer.add_images('Generated', fake_images, global_step)
```

### 8.3 체크리스트

**학습 전:**
- [ ] 데이터 정규화 (-1, 1)
- [ ] 가중치 초기화 (mean=0, std=0.02)
- [ ] Adam optimizer (lr=0.0002, β1=0.5)

**학습 중:**
- [ ] D와 G loss 모니터링 (균형 유지)
- [ ] 샘플 품질 시각적 확인
- [ ] Mode collapse 체크

**학습 후:**
- [ ] FID/IS 계산
- [ ] 다양한 latent code로 샘플 생성
- [ ] Latent space interpolation 테스트

---

## 9. 실습 프로젝트

### 프로젝트 1: MNIST DCGAN

**목표:** DCGAN으로 손글씨 숫자 생성

```bash
# 1. 구현된 코드 실행
cd code/gan
python train_gan.py --model dcgan --dataset mnist --epochs 50

# 2. 결과 확인
# samples/ 디렉토리에서 생성 이미지 확인

# 3. FID 계산
python evaluate_gan.py --checkpoint checkpoints/dcgan_epoch_50.pth
```

### 프로젝트 2: CelebA WGAN-GP

**목표:** 고품질 얼굴 이미지 생성

```bash
python train_gan.py --model wgan-gp --dataset celeba --epochs 100 --n-critic 5
```

### 프로젝트 3: Latent Space 탐색

**목표:** Latent space interpolation 및 arithmetic

```python
# Interpolation
z1 = torch.randn(1, 100)
z2 = torch.randn(1, 100)

for alpha in np.linspace(0, 1, 10):
    z = alpha * z1 + (1 - alpha) * z2
    img = generator(z)
    save_image(img, f'interpolation_{alpha:.1f}.png')

# Arithmetic (예: 미소 - 무표정 + 남자 = 미소짓는 남자)
z_smile = find_direction_vector('smile')
z_neutral = find_direction_vector('neutral')
z_man = sample_latent_code()

z_smiling_man = z_man + (z_smile - z_neutral)
```

---

## 📚 참고 자료

### 필수 논문

1. **GAN** - Goodfellow et al., 2014
   - [arXiv](https://arxiv.org/abs/1406.2661)
   - 원조 GAN 논문

2. **DCGAN** - Radford et al., 2015
   - [arXiv](https://arxiv.org/abs/1511.06434)
   - CNN 기반 GAN의 표준

3. **WGAN** - Arjovsky et al., 2017
   - [arXiv](https://arxiv.org/abs/1701.07875)
   - Wasserstein Distance 도입

4. **WGAN-GP** - Gulrajani et al., 2017
   - [arXiv](https://arxiv.org/abs/1704.00028)
   - Gradient Penalty로 개선

### 추가 학습 자료

- **Lil'Log - GAN 시리즈**: https://lilianweng.github.io/posts/2017-08-20-gan/
- **Distill.pub - GAN Lab**: https://poloclub.github.io/ganlab/
- **Ian Goodfellow NIPS Tutorial**: https://arxiv.org/abs/1701.00160

---

## ✅ 학습 체크리스트

- [ ] GAN의 minimax game 이해
- [ ] DCGAN 5가지 아키텍처 규칙 암기
- [ ] DCGAN Generator/Discriminator 직접 구현
- [ ] Transposed Convolution 계산 방법 이해
- [ ] Wasserstein Distance의 장점 설명 가능
- [ ] Gradient Penalty 코드 작성 가능
- [ ] WGAN-GP 학습 루프 구현
- [ ] Mode Collapse 감지 및 해결
- [ ] FID, IS 계산 코드 작성
- [ ] MNIST/CelebA로 GAN 학습 성공

---

**최종 업데이트**: 2025-11-15
**작성자**: Claude
**라이선스**: MIT
