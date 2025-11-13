# Phase 4: Diffusion Models - 완전 정복

## 🎯 Diffusion Model이란?

**Diffusion Model**은 현재 가장 강력한 생성 모델입니다:
- **DALL-E 2**, **Stable Diffusion**, **Imagen** 모두 Diffusion 기반
- GAN보다 학습 안정적
- VAE보다 고품질 생성

**핵심 아이디어**:
- **Forward Process**: 이미지에 점진적으로 노이즈 추가 (파괴)
- **Reverse Process**: 노이즈를 제거하여 이미지 복원 (생성)

---

## Week 24: DDPM (Denoising Diffusion Probabilistic Models)

### 💡 Step 1: 직관적 이해

#### 1.1 물리적 비유: 잉크 확산

```
깨끗한 물 + 잉크 한 방울
    ↓
시간이 지남 (확산)
    ↓
균일하게 퍼짐 (노이즈)
```

**Forward Process (확산)**:
- 이미지 → 순수 노이즈
- 정보가 점차 사라짐
- 되돌릴 수 없음 (열역학 제2법칙)

**Reverse Process (역확산)**:
- 순수 노이즈 → 이미지
- 정보를 되살림
- 이것을 **학습**해야 함!

#### 1.2 수학적 비유: Markov Chain

```
x₀ → x₁ → x₂ → ... → x_T
↑                     ↑
원본 이미지        순수 노이즈
```

**Forward**: q(xₜ|xₜ₋₁) = N(√(1-βₜ)·xₜ₋₁, βₜI)
- 이전 상태에 작은 노이즈 추가
- βₜ: 노이즈 양 (schedule)

**Reverse**: p(xₜ₋₁|xₜ) = ?
- 노이즈 제거
- **학습 목표**!

---

### 📐 Step 2: Forward Process (노이즈 추가)

#### 2.1 단계별 노이즈 추가

```python
# t번째 스텝의 노이즈 추가
xₜ = √(1-βₜ) · xₜ₋₁ + √βₜ · εₜ

where:
- βₜ: 노이즈 스케줄 (작은 값, 예: 0.0001 ~ 0.02)
- εₜ ~ N(0, I): 가우시안 노이즈
```

**왜 이렇게?**
1. **√(1-βₜ)**: 원본 이미지의 비율 (점점 감소)
2. **√βₜ**: 노이즈의 비율 (점점 증가)
3. **제곱근**: 분산을 일정하게 유지

#### 2.2 한 번에 계산하기 (Closed Form)

**중요한 성질**: 중간 단계를 건너뛸 수 있음!

```
x₀에서 xₜ로 한 번에:

xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε

where:
- ᾱₜ = ∏ᵢ₌₁ᵗ (1-βᵢ) = α₁ · α₂ · ... · αₜ
- αₜ = 1 - βₜ
- ε ~ N(0, I)
```

**왜 중요한가?**
- 학습 시 임의의 timestep t 선택 가능
- 중간 단계 계산 불필요
- 학습 효율적!

#### 💻 실습 1: Forward Process 시각화

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ForwardDiffusion:
    """
    Forward Diffusion Process

    점진적으로 노이즈를 추가하여 이미지를 파괴
    """

    def __init__(self, timesteps=1000):
        """
        Args:
            timesteps: 총 노이즈 추가 단계 (T)
        """
        self.timesteps = timesteps

        # β schedule 정의
        # Linear schedule: β₁ = 0.0001 → βₜ = 0.02
        self.betas = self.linear_beta_schedule(timesteps)

        # α = 1 - β
        self.alphas = 1.0 - self.betas

        # ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ (누적 곱)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱₜ₋₁
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])

        # √ᾱₜ
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # √(1 - ᾱₜ)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )

    @staticmethod
    def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
        """
        Linear schedule for β

        β₁, β₂, ..., βₜ를 선형적으로 증가

        Args:
            timesteps: T
            beta_start: β₁
            beta_end: βₜ
        Returns:
            betas: [T]
        """
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        Improved Cosine Schedule (Improved DDPM)

        더 부드러운 노이즈 추가

        수식:
        ᾱₜ = f(t) / f(0)
        f(t) = cos²((t/T + s) / (1+s) · π/2)
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: q(xₜ | x₀)

        xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε

        Args:
            x_start: 원본 이미지 x₀ [B, C, H, W]
            t: timestep [B]
            noise: 가우시안 노이즈 ε (None이면 자동 생성)
        Returns:
            x_t: 노이즈가 추가된 이미지 [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # timestep에 해당하는 계수 가져오기
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Broadcasting을 위해 reshape
        # [B] → [B, 1, 1, 1]
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)

        # xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε
        return sqrt_alphas_cumprod_t * x_start + \
               sqrt_one_minus_alphas_cumprod_t * noise


def visualize_forward_process():
    """Forward Process 시각화"""

    print("🎨 Forward Diffusion Process 시각화")
    print("=" * 70)

    # 이미지 로드
    from torchvision import transforms

    # MNIST 숫자 이미지 사용
    from torchvision.datasets import MNIST
    dataset = MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor())

    # 숫자 7 선택
    img, label = dataset[0]
    img = img.unsqueeze(0)  # [1, 1, 28, 28]

    print(f"원본 이미지: 레이블 = {label}")

    # Forward Process 초기화
    diffusion = ForwardDiffusion(timesteps=1000)

    # 여러 timestep에서 샘플링
    timesteps_to_show = [0, 50, 100, 200, 400, 600, 800, 999]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, t in enumerate(timesteps_to_show):
        # 노이즈 추가
        t_tensor = torch.tensor([t])
        noisy_img = diffusion.q_sample(img, t_tensor)

        # 시각화
        ax = axes[idx]
        ax.imshow(noisy_img[0, 0].numpy(), cmap='gray')
        ax.set_title(f't = {t}\n' +
                     f'√ᾱₜ = {diffusion.sqrt_alphas_cumprod[t]:.4f}',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Forward Diffusion Process: 이미지 → 노이즈',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('forward_diffusion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✅ 관찰:")
    print("• t=0: 원본 이미지 (노이즈 없음)")
    print("• t 증가: 점차 노이즈 추가")
    print("• t=999: 거의 순수 노이즈 (이미지 정보 소실)")

    # β schedule 비교
    print("\n\n📊 β Schedule 비교")
    print("=" * 70)

    linear_betas = ForwardDiffusion.linear_beta_schedule(1000)
    cosine_betas = ForwardDiffusion.cosine_beta_schedule(1000)

    # Linear schedule의 α_cumprod
    alphas_linear = 1.0 - linear_betas
    alphas_cumprod_linear = torch.cumprod(alphas_linear, dim=0)

    # Cosine schedule의 α_cumprod
    alphas_cosine = 1.0 - cosine_betas
    alphas_cumprod_cosine = torch.cumprod(alphas_cosine, dim=0)

    plt.figure(figsize=(14, 5))

    # β 비교
    plt.subplot(1, 2, 1)
    plt.plot(linear_betas.numpy(), label='Linear', linewidth=2)
    plt.plot(cosine_betas.numpy(), label='Cosine', linewidth=2)
    plt.xlabel('Timestep t', fontsize=12)
    plt.ylabel('βₜ', fontsize=12)
    plt.title('β Schedule 비교', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # ᾱₜ 비교
    plt.subplot(1, 2, 2)
    plt.plot(alphas_cumprod_linear.numpy(), label='Linear', linewidth=2)
    plt.plot(alphas_cumprod_cosine.numpy(), label='Cosine', linewidth=2)
    plt.xlabel('Timestep t', fontsize=12)
    plt.ylabel('ᾱₜ (signal strength)', fontsize=12)
    plt.title('Cumulative Product ᾱₜ', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('beta_schedules.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ Schedule 비교:")
    print("• Linear: 균일하게 증가, 끝부분에서 급격히 노이즈")
    print("• Cosine: 부드럽게 증가, 더 안정적인 학습")

# 실행
visualize_forward_process()
```

**코드 핵심 설명**:

1. **q_sample 함수**:
   ```python
   # 한 줄로 xₜ 계산!
   xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε
   ```
   - t=0: √ᾱ₀=1, √(1-ᾱ₀)=0 → xₜ = x₀ (원본)
   - t=T: √ᾱₜ≈0, √(1-ᾱₜ)≈1 → xₜ = ε (노이즈)

2. **β Schedule**:
   - **Linear**: 단순, 초기 DDPM에서 사용
   - **Cosine**: 더 나은 성능, Improved DDPM

3. **왜 √를 사용?**:
   ```python
   # 분산 유지
   Var[xₜ] = ᾱₜ·Var[x₀] + (1-ᾱₜ)·Var[ε]
            = ᾱₜ + (1-ᾱₜ)  # x₀, ε 모두 단위 분산
            = 1  # 일정!
   ```

---

### 📐 Step 3: Reverse Process (노이즈 제거)

#### 3.1 이론적 배경

**목표**: p(xₜ₋₁|xₜ)를 학습

**베이즈 정리**:
```
p(xₜ₋₁|xₜ, x₀) = q(xₜ|xₜ₋₁, x₀) · q(xₜ₋₁|x₀) / q(xₜ|x₀)
```

**가우시안의 곱은 가우시안**:
```
q(xₜ₋₁|xₜ, x₀) = N(xₜ₋₁; μ̃ₜ(xₜ, x₀), β̃ₜI)

where:
μ̃ₜ(xₜ, x₀) = (√ᾱₜ₋₁·βₜ·x₀ + √αₜ·(1-ᾱₜ₋₁)·xₜ) / (1-ᾱₜ)
β̃ₜ = (1-ᾱₜ₋₁)/(1-ᾱₜ) · βₜ
```

**문제**: x₀를 모름!

**해결**: x₀를 xₜ와 εₜ로 표현
```
xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε

∴ x₀ = (xₜ - √(1-ᾱₜ)·ε) / √ᾱₜ
```

**최종 목표**: εₜ를 예측하는 신경망 학습!
```
ε_θ(xₜ, t) ≈ ε
```

#### 3.2 손실 함수

**DDPM Loss**:
```
L_simple = E_t,x₀,ε [||ε - ε_θ(xₜ, t)||²]

where:
- t ~ Uniform(1, T)
- x₀ ~ q(x₀)  (데이터 분포)
- ε ~ N(0, I)
- xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε
```

**직관적 의미**:
- 노이즈가 추가된 이미지 xₜ가 주어짐
- 원래 추가된 노이즈 ε를 예측
- MSE Loss로 학습

**왜 이게 작동하나?**
- 노이즈를 정확히 예측하면 x₀도 복원 가능
- 모든 timestep t에 대해 학습
- 간단하고 효과적!

#### 💻 실습 2: U-Net 구조

```python
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Timestep t를 임베딩

    Transformer의 Positional Encoding과 유사
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: [B]
        Returns:
            embeddings: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2

        # 주파수 계산
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )

        # time과 곱하기
        embeddings = time[:, None] * embeddings[None, :]

        # sin, cos 적용
        embeddings = torch.cat((
            embeddings.sin(),
            embeddings.cos()
        ), dim=-1)

        return embeddings


class Block(nn.Module):
    """
    Residual Block with Time Embedding

    Conv → GroupNorm → SiLU → Conv
           ↓
    Time Embedding 추가
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        # Convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Normalization
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Activation
        self.act = nn.SiLU()  # Swish activation

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_emb_dim]
        Returns:
            out: [B, C, H, W]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Time embedding 추가
        # [B, C] → [B, C, 1, 1] → [B, C, H, W]
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[:, :, None, None]
        h = h + time_emb

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        # Residual connection
        return h + self.residual_conv(x)


class SimpleUNet(nn.Module):
    """
    간단한 U-Net for DDPM

    구조:
    Encoder: 28x28 → 14x14 → 7x7
    Bottleneck: 7x7
    Decoder: 7x7 → 14x14 → 28x28
    """

    def __init__(self, in_channels=1, out_channels=1,
                 time_emb_dim=128, base_channels=64):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Encoder (Downsampling)
        self.down1 = Block(in_channels, base_channels, time_emb_dim)
        self.down2 = Block(base_channels, base_channels * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = Block(base_channels * 2, base_channels * 2,
                               time_emb_dim)

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2,
                                     2, stride=2)
        self.up_block1 = Block(base_channels * 4, base_channels * 2,
                               time_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                     2, stride=2)
        self.up_block2 = Block(base_channels * 2, base_channels,
                               time_emb_dim)

        # Output
        self.output = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        """
        Args:
            x: noisy image [B, 1, 28, 28]
            t: timestep [B]
        Returns:
            noise prediction [B, 1, 28, 28]
        """
        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]

        # Encoder
        x1 = self.down1(x, t_emb)  # [B, 64, 28, 28]
        x1_pool = self.pool(x1)     # [B, 64, 14, 14]

        x2 = self.down2(x1_pool, t_emb)  # [B, 128, 14, 14]
        x2_pool = self.pool(x2)          # [B, 128, 7, 7]

        # Bottleneck
        b = self.bottleneck(x2_pool, t_emb)  # [B, 128, 7, 7]

        # Decoder with skip connections
        u1 = self.up1(b)  # [B, 128, 14, 14]
        u1 = torch.cat([u1, x2], dim=1)  # [B, 256, 14, 14]
        u1 = self.up_block1(u1, t_emb)   # [B, 128, 14, 14]

        u2 = self.up2(u1)  # [B, 64, 28, 28]
        u2 = torch.cat([u2, x1], dim=1)  # [B, 128, 28, 28]
        u2 = self.up_block2(u2, t_emb)   # [B, 64, 28, 28]

        # Output
        out = self.output(u2)  # [B, 1, 28, 28]

        return out


# 테스트
def test_unet():
    print("🧪 U-Net 테스트")
    print("=" * 70)

    model = SimpleUNet()

    # 입력
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.randint(0, 1000, (batch_size,))

    # 순전파
    noise_pred = model(x, t)

    print(f"입력 shape: {x.shape}")
    print(f"Timestep: {t}")
    print(f"출력 shape: {noise_pred.shape}")
    print("\n✅ U-Net 작동 확인!")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params:,}")

test_unet()
```

**코드 핵심**:

1. **Time Embedding**:
   ```python
   # Timestep t를 네트워크에 전달
   # Positional Encoding과 유사
   # sin, cos 함수로 임베딩
   ```

2. **Residual Block**:
   ```python
   # Conv → Norm → Act → Conv
   # + Time Embedding
   # + Residual Connection
   ```

3. **U-Net 구조**:
   ```
   Encoder: 점진적으로 해상도 감소
   Bottleneck: 가장 압축된 표현
   Decoder: 해상도 복원 (Skip Connection 사용)
   ```

---

이어서 학습과 샘플링 코드를 작성하고, 전체를 커밋하겠습니다!
