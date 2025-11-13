# Phase 3: VAE (Variational Autoencoder) - 완전 정복

## 🎯 VAE를 왜 배워야 하는가?

VAE는 **현대 Generative Model의 기초**입니다:
- Stable Diffusion의 **Latent Diffusion**은 VAE 기반
- DALL-E는 VQ-VAE 사용
- 많은 생성 모델이 VAE의 아이디어를 차용

VAE를 이해하면:
- ✅ Latent Space 개념 이해
- ✅ ELBO (Evidence Lower Bound) 최적화 이해
- ✅ Diffusion Model 이해의 기초
- ✅ 생성 모델의 학습 원리 이해

---

## Week 12: VAE 이론 - 단계별 완전 이해

### 💡 Step 1: 문제 정의

**우리가 하고 싶은 것**:
- 데이터 x (예: 이미지)의 분포 p(x)를 학습
- 새로운 샘플 생성

**Autoencoder만으로는 부족**:
- 단순 Autoencoder: 입력 → 압축 → 복원
- 문제점: 잠재 공간이 불연속적, 새로운 샘플 생성 불가

```python
# 일반 Autoencoder의 한계
encoder: x → z  # 압축
decoder: z → x̂  # 복원

# 문제: z 공간이 불연속적
# z1 = encoder(image1)
# z2 = encoder(image2)
# z_new = (z1 + z2) / 2  # 의미 없는 코드!
```

**VAE의 해결책**:
- 잠재 공간을 **확률 분포**로 모델링
- 연속적이고 구조화된 잠재 공간

---

### 📐 Step 2: 생성 모델의 확률적 관점

#### 2.1 잠재 변수 모델 (Latent Variable Model)

**아이디어**: 관측된 데이터 x는 숨겨진 변수 z로부터 생성됨

```
생성 과정:
1. z ~ p(z)          # 잠재 변수 샘플링
2. x ~ p(x|z)        # z로부터 x 생성

목표: p(x) = ∫ p(x|z) p(z) dz를 학습
```

**예시: 얼굴 이미지 생성**
```
z: [head_angle, smile, age, ...]  # 잠재 요인
x: 픽셀 값들  # 관측 데이터

p(z): 사전 분포 (Prior) - 잠재 요인이 어떻게 분포하는가?
p(x|z): 우도 (Likelihood) - 주어진 잠재 요인으로 어떤 얼굴이 생성되는가?
```

#### 2.2 왜 잠재 변수가 필요한가?

**직접 p(x)를 모델링하면?**
- 이미지는 고차원 (예: 64×64×3 = 12,288차원)
- 가능한 이미지 조합: 256^12,288 (천문학적!)
- 대부분은 의미 없는 노이즈

**잠재 변수 z를 사용하면?**
- z는 저차원 (예: 128차원)
- z는 데이터의 **본질적 특성**만 담음
- 생성 과정을 이해하고 제어 가능

---

### 📐 Step 3: ELBO 유도 - 수식의 의미 이해

#### 3.1 문제: p(z|x)를 구할 수 없다

**베이즈 정리**:
```
p(z|x) = p(x|z) p(z) / p(x)
```

**문제**: p(x) = ∫ p(x|z) p(z) dz를 계산할 수 없음 (적분 불가)

**해결**: 근사 분포 q(z|x)를 도입

```
q(z|x) ≈ p(z|x)

q(z|x): Encoder가 출력하는 분포
        (inference model, recognition network)
```

#### 3.2 ELBO 유도 - 단계별

**목표**: log p(x)를 최대화하고 싶음 (데이터의 로그 우도)

**Step 1**: q(z|x) 도입
```
log p(x) = log ∫ p(x,z) dz
         = log ∫ [p(x,z) / q(z|x)] q(z|x) dz
         = log E_q [p(x,z) / q(z|x)]
```

**Step 2**: Jensen's Inequality 적용
```
log E[X] ≥ E[log X]  (log는 concave 함수)

따라서:
log p(x) ≥ E_q [log p(x,z) / q(z|x)]
         = E_q [log p(x,z)] - E_q [log q(z|x)]
```

**Step 3**: 전개
```
ELBO = E_q [log p(x,z)] - E_q [log q(z|x)]
     = E_q [log p(x|z) + log p(z)] - E_q [log q(z|x)]
     = E_q [log p(x|z)] + E_q [log p(z)] - E_q [log q(z|x)]
     = E_q [log p(x|z)] - [E_q [log q(z|x)] - E_q [log p(z)]]
     = E_q [log p(x|z)] - KL(q(z|x) || p(z))
```

**최종 ELBO**:
```
ELBO = E_q(z|x) [log p(x|z)] - KL(q(z|x) || p(z))
       ↑                        ↑
  Reconstruction Term      Regularization Term
  (재구성 항)                (정규화 항)
```

#### 3.3 ELBO의 의미

```
log p(x) = ELBO + KL(q(z|x) || p(z|x))
          ↑      ↑           ↑
       목표    최적화      항상 ≥ 0
```

**두 항의 의미**:

1. **Reconstruction Term**: E_q [log p(x|z)]
   - 인코더로 z를 얻고, 디코더로 x를 복원
   - "얼마나 잘 복원하는가?"
   - 높을수록 좋음 (최대화)

2. **KL Divergence**: KL(q(z|x) || p(z))
   - q(z|x)가 사전 분포 p(z)와 얼마나 다른가?
   - "잠재 공간이 얼마나 구조화되어 있는가?"
   - 낮을수록 좋음 (최소화)

**Trade-off**:
```
ELBO = Reconstruction - β × KL

β > 1: 더 구조화된 잠재 공간 (β-VAE)
β < 1: 더 나은 재구성
```

---

### 💻 실습 1: ELBO 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def visualize_elbo():
    """ELBO의 두 항을 시각화"""

    # 가우시안 분포 설정
    x = np.linspace(-5, 8, 1000)

    # Prior: p(z) = N(0, 1)
    prior = norm.pdf(x, 0, 1)

    # 여러 Posterior: q(z|x) = N(μ, σ²)
    posteriors = [
        (0, 1, "Perfect match"),
        (2, 1, "Shifted mean"),
        (0, 2, "Larger variance"),
        (3, 0.5, "Very different")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (mu, sigma, desc) in enumerate(posteriors):
        ax = axes[idx]

        # Posterior
        posterior = norm.pdf(x, mu, sigma)

        # KL Divergence 계산 (가우시안의 해석적 해)
        kl = np.log(1/sigma) + (sigma**2 + mu**2) / 2 - 0.5

        # 시각화
        ax.plot(x, prior, label='Prior p(z)', linewidth=2, color='blue')
        ax.plot(x, posterior, label=f'Posterior q(z|x)', linewidth=2, color='red')
        ax.fill_between(x, prior, alpha=0.2, color='blue')
        ax.fill_between(x, posterior, alpha=0.2, color='red')

        ax.set_xlabel('z', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{desc}\nKL(q||p) = {kl:.4f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("📊 ELBO 분석")
    print("=" * 70)
    print("\nELBO = Reconstruction - KL Divergence")
    print("\n1. Perfect match (μ=0, σ=1):")
    print("   → KL = 0")
    print("   → Prior와 Posterior가 같음")
    print("   → 이상적인 경우 (실제로는 불가능)")
    print("\n2. Shifted mean (μ=2, σ=1):")
    print("   → KL > 0")
    print("   → Posterior가 Prior에서 멀어짐")
    print("   → 데이터마다 다른 잠재 코드 학습")
    print("\n3. Larger variance (μ=0, σ=2):")
    print("   → KL > 0")
    print("   → Posterior가 더 퍼짐")
    print("   → 불확실성 증가")
    print("\n4. Very different (μ=3, σ=0.5):")
    print("   → KL 매우 큼")
    print("   → Prior와 완전히 다름")
    print("   → Regularization이 강하게 작용")

visualize_elbo()
```

**코드 설명**:

1. **Prior p(z) = N(0,1)**:
   - 표준 정규 분포
   - 단순하고 샘플링하기 쉬움
   - 모든 VAE에서 동일

2. **Posterior q(z|x)**:
   - 데이터 x에 따라 다름
   - Encoder가 μ와 σ를 출력
   - N(μ(x), σ²(x))로 모델링

3. **KL Divergence**:
   - q가 p에서 멀어질수록 증가
   - 정규화 역할: 잠재 공간을 구조화

---

### 📐 Step 4: Reparameterization Trick

#### 4.1 문제: 그래디언트를 어떻게 계산하나?

**ELBO**:
```
ELBO = E_q(z|x) [log p(x|z)] - KL(q(z|x) || p(z))
```

**문제**:
```python
# q(z|x)에서 z를 샘플링
z = sample_from_q(x)  # 확률적 연산!

# log p(x|z) 계산
reconstruction = log_p(x, z)

# 어떻게 역전파?
# z는 랜덤 샘플링 → 미분 불가!
```

#### 4.2 해결: Reparameterization Trick

**아이디어**: 랜덤성을 분리

**Before** (미분 불가):
```
z ~ N(μ, σ²)  # 랜덤 샘플링
```

**After** (미분 가능):
```
ε ~ N(0, 1)      # 표준 정규 분포 (고정)
z = μ + σ × ε    # 결정론적 변환
```

**왜 작동하나?**
```
z = μ + σ × ε where ε ~ N(0,1)

E[z] = E[μ + σε] = μ + σE[ε] = μ
Var[z] = Var[σε] = σ² Var[ε] = σ²

∴ z ~ N(μ, σ²)  ✓
```

**그래디언트 계산**:
```python
# 이제 μ와 σ에 대해 미분 가능!
∂L/∂μ = ∂L/∂z × ∂z/∂μ = ∂L/∂z × 1
∂L/∂σ = ∂L/∂z × ∂z/∂σ = ∂L/∂z × ε
```

#### 💻 실습 2: Reparameterization Trick 시연

```python
import torch
import torch.nn as nn

class ReparameterizationDemo:
    """Reparameterization Trick 시연"""

    @staticmethod
    def wrong_way(mu, logvar, requires_grad=True):
        """
        잘못된 방법: 직접 샘플링

        문제: torch.randn은 그래디언트 추적 안 됨
        """
        sigma = torch.exp(0.5 * logvar)
        # 잘못된 방법!
        z = torch.randn_like(sigma) * sigma + mu
        return z

    @staticmethod
    def correct_way(mu, logvar):
        """
        올바른 방법: Reparameterization Trick

        z = μ + σ × ε, where ε ~ N(0,1)
        """
        # ε 샘플링 (그래디언트 추적 X)
        epsilon = torch.randn_like(mu)

        # σ = exp(0.5 × log(σ²))
        sigma = torch.exp(0.5 * logvar)

        # z = μ + σε (그래디언트 추적 O)
        z = mu + sigma * epsilon

        return z

# 테스트
print("🎲 Reparameterization Trick 테스트")
print("=" * 70)

# 파라미터 생성
mu = torch.tensor([1.0, 2.0], requires_grad=True)
logvar = torch.tensor([0.0, 0.5], requires_grad=True)

print(f"μ = {mu.data.numpy()}")
print(f"log(σ²) = {logvar.data.numpy()}")
print(f"σ = {torch.exp(0.5 * logvar).data.numpy()}")

# 샘플링
z = ReparameterizationDemo.correct_way(mu, logvar)
print(f"\n샘플 z = {z.data.numpy()}")

# 더미 손실 함수
loss = z.sum()

# 역전파
loss.backward()

print(f"\n그래디언트:")
print(f"∂L/∂μ = {mu.grad.numpy()}")
print(f"∂L/∂log(σ²) = {logvar.grad.numpy()}")

print("\n✅ Reparameterization Trick 덕분에 그래디언트 계산 가능!")

# 샘플링이 확률 분포를 따르는지 확인
print("\n\n📊 샘플링 분포 확인")
print("=" * 70)

n_samples = 10000
samples = []

for _ in range(n_samples):
    z = ReparameterizationDemo.correct_way(mu, logvar)
    samples.append(z.detach().numpy())

samples = np.array(samples)

print(f"이론적 평균: {mu.data.numpy()}")
print(f"실제 평균:   {samples.mean(axis=0)}")
print(f"\n이론적 표준편차: {torch.exp(0.5 * logvar).data.numpy()}")
print(f"실제 표준편차:   {samples.std(axis=0)}")

print("\n✅ 샘플이 N(μ, σ²) 분포를 따름!")
```

**코드 핵심**:

1. **잘못된 방법**:
   ```python
   z = torch.randn(...) * sigma + mu
   # torch.randn()은 그래디언트 추적 안 됨!
   ```

2. **올바른 방법**:
   ```python
   epsilon = torch.randn_like(mu)  # 상수 취급
   z = mu + sigma * epsilon        # μ, σ에 대해 미분 가능!
   ```

3. **검증**:
   - 많은 샘플을 생성
   - 평균과 표준편차 확인
   - 이론값과 일치하는지 확인

---

### 💻 실습 3: 완전한 VAE 구현 (MNIST)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class VAE(nn.Module):
    """
    Variational Autoencoder

    Encoder: x → [μ(x), log σ²(x)]
    Reparameterization: z = μ + σ × ε
    Decoder: z → x̂
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()

        # Encoder: x → μ, log(σ²)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z → x̂
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encoder: x → q(z|x) = N(μ(x), σ²(x))

        Args:
            x: input image [batch_size, 784]
        Returns:
            mu: mean [batch_size, latent_dim]
            logvar: log variance [batch_size, latent_dim]
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick

        z = μ + σ × ε, where ε ~ N(0,1)

        Args:
            mu: mean [batch_size, latent_dim]
            logvar: log variance [batch_size, latent_dim]
        Returns:
            z: latent code [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 × log σ²)
        eps = torch.randn_like(std)     # ε ~ N(0,1)
        z = mu + std * eps              # z = μ + σε
        return z

    def decode(self, z):
        """
        Decoder: z → p(x|z)

        Args:
            z: latent code [batch_size, latent_dim]
        Returns:
            x_recon: reconstructed image [batch_size, 784]
        """
        h = F.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(h))  # 픽셀 값: [0, 1]
        return x_recon

    def forward(self, x):
        """
        전체 VAE 순전파

        Args:
            x: input [batch_size, 784]
        Returns:
            x_recon: reconstruction [batch_size, 784]
            mu: mean [batch_size, latent_dim]
            logvar: log variance [batch_size, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def sample(self, num_samples, device):
        """
        새로운 이미지 생성

        1. z ~ p(z) = N(0, 1) 샘플링
        2. x = decode(z)

        Args:
            num_samples: 생성할 샘플 수
            device: cuda or cpu
        Returns:
            samples: generated images [num_samples, 784]
        """
        with torch.no_grad():
            # Prior에서 샘플링
            z = torch.randn(num_samples, self.fc3.in_features).to(device)
            # 디코드
            samples = self.decode(z)
        return samples


def vae_loss_function(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE 손실 함수

    L = Reconstruction Loss + β × KL Divergence

    Args:
        x_recon: 복원된 이미지 [batch_size, 784]
        x: 원본 이미지 [batch_size, 784]
        mu: 평균 [batch_size, latent_dim]
        logvar: log 분산 [batch_size, latent_dim]
        beta: KL 가중치 (β-VAE)
    """
    # 1. Reconstruction Loss: Binary Cross-Entropy
    # p(x|z)를 Bernoulli로 모델링
    recon_loss = F.binary_cross_entropy(
        x_recon, x, reduction='sum'
    )

    # 2. KL Divergence: KL(q(z|x) || p(z))
    # q(z|x) = N(μ(x), σ²(x))
    # p(z) = N(0, 1)
    #
    # 해석적 해:
    # KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    # 총 손실
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


# 학습
def train_vae():
    """VAE 학습 함수"""

    print("🎨 VAE 학습 시작")
    print("=" * 70)

    # 하이퍼파라미터
    batch_size = 128
    latent_dim = 20
    epochs = 10
    learning_rate = 1e-3
    beta = 1.0  # β-VAE

    # 디바이스
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터 로드
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        '../data', train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # 모델
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            # 데이터 준비
            data = data.view(-1, 784).to(device)  # [batch_size, 784]

            # 순전파
            x_recon, mu, logvar = model(data)

            # 손실 계산
            loss, recon, kl = vae_loss_function(
                x_recon, data, mu, logvar, beta=beta
            )

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 누적
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        # 평균 손실
        avg_loss = epoch_loss / len(train_dataset)
        avg_recon = epoch_recon / len(train_dataset)
        avg_kl = epoch_kl / len(train_dataset)

        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1:2d}: "
              f"Loss={avg_loss:.4f} "
              f"(Recon={avg_recon:.4f}, KL={avg_kl:.4f})")

    return model, train_losses


# 시각화
def visualize_vae(model, device):
    """VAE 결과 시각화"""

    model.eval()

    # 1. 재구성 (Reconstruction)
    test_dataset = datasets.MNIST(
        '../data', train=False, download=True,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    data, _ = next(iter(test_loader))
    data = data.to(device)

    with torch.no_grad():
        x_recon, _, _ = model(data.view(-1, 784))
        x_recon = x_recon.view(-1, 1, 28, 28)

    # 시각화: 원본 vs 재구성
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))

    for i in range(10):
        # 원본
        axes[0, i].imshow(data[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # 재구성
        axes[1, i].imshow(x_recon[i, 0].cpu(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    plt.savefig('vae_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 2. 생성 (Generation)
    samples = model.sample(25, device)
    samples = samples.view(-1, 1, 28, 28)

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(samples[i, 0].cpu(), cmap='gray')
        ax.axis('off')

    plt.suptitle('Generated Samples from VAE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vae_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 3. Latent Space Interpolation
    # 두 숫자 사이를 보간
    with torch.no_grad():
        # 숫자 0과 1 선택
        data1 = test_dataset[0][0].unsqueeze(0).to(device)
        data2 = test_dataset[2][0].unsqueeze(0).to(device)  # 다른 클래스

        # Encode
        mu1, _ = model.encode(data1.view(-1, 784))
        mu2, _ = model.encode(data2.view(-1, 784))

        # Interpolate
        n_steps = 10
        interpolations = []

        for alpha in torch.linspace(0, 1, n_steps):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            x_interp = model.decode(z_interp)
            interpolations.append(x_interp.view(28, 28).cpu())

    fig, axes = plt.subplots(1, n_steps, figsize=(15, 2))

    for i, ax in enumerate(axes):
        ax.imshow(interpolations[i], cmap='gray')
        ax.axis('off')

    plt.suptitle('Latent Space Interpolation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vae_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ 시각화 완료!")


# 실행
if __name__ == '__main__':
    model, losses = train_vae()
    visualize_vae(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```

**코드 전체 흐름**:

1. **Encode**:
   ```
   x → [μ, log(σ²)]
   ```

2. **Reparameterize**:
   ```
   z = μ + σ × ε
   ```

3. **Decode**:
   ```
   z → x̂
   ```

4. **Loss**:
   ```
   L = BCE(x, x̂) + KL(q||p)
   ```

---

이어서 Phase 4 (Diffusion Models)도 이렇게 상세하게 작성할까요? 아니면 현재까지 작성한 내용을 커밋하고 푸시할까요?
