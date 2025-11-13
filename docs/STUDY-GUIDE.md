# Generative AI 학습 가이드

이 가이드는 Generative AI를 처음 시작하는 분들부터 고급 연구자까지 모두를 위한 상세한 학습 로드맵입니다.

---

## 📖 학습 전략

### 1. 학습 접근 방법

#### Top-Down vs Bottom-Up
- **Bottom-Up (추천)**: 수학/이론 → 기초 → 고급 → 프로젝트
  - 장점: 탄탄한 기초, 깊은 이해
  - 단점: 초반에 동기부여 어려움

- **Top-Down**: 프로젝트 먼저 → 필요한 이론 학습
  - 장점: 빠른 실습, 높은 동기부여
  - 단점: 이론적 이해 부족 가능

**추천**: 하이브리드 접근
- Phase 1-2는 Bottom-Up으로 기초 다지기
- Phase 3부터는 실습과 이론을 병행

### 2. 시간 관리

#### 주간 학습 시간
- **최소**: 주 10시간 (평일 1시간, 주말 5시간)
- **권장**: 주 15-20시간 (평일 2시간, 주말 6-8시간)
- **집중**: 주 30시간+ (풀타임 학습)

#### 일일 루틴 예시
```
평일:
07:00-08:00  논문 읽기 / 이론 학습
19:00-21:00  코딩 실습

주말:
09:00-12:00  프로젝트 작업
14:00-17:00  심화 학습 / 실험
```

### 3. 학습 사이클

**주간 사이클**:
1. **월요일**: 이번 주 목표 설정, 논문 선정
2. **화-목**: 이론 학습 + 코드 구현
3. **금요일**: 주간 리뷰, 블로그 작성
4. **토-일**: 프로젝트 진행, 실험

**월간 사이클**:
1. **Week 1-2**: 새로운 주제 학습
2. **Week 3**: 구현 및 실험
3. **Week 4**: 복습 및 정리, 다음 달 계획

---

## 📚 Phase별 상세 가이드

## Phase 1: 수학 및 이론 기초 (3-4주)

### Week 1: 확률론 기초

#### Day 1-2: 확률의 기본 개념
**학습 내용**:
- 확률 공간 (Ω, F, P)
- 확률 변수와 확률 분포
- 기댓값과 분산

**실습**:
```python
import numpy as np
import matplotlib.pyplot as plt

# 동전 던지기 시뮬레이션
n_trials = 10000
results = np.random.choice([0, 1], size=n_trials, p=[0.5, 0.5])
heads_ratio = np.cumsum(results) / np.arange(1, n_trials + 1)

plt.plot(heads_ratio)
plt.axhline(y=0.5, color='r', linestyle='--', label='True probability')
plt.xlabel('Number of trials')
plt.ylabel('Ratio of heads')
plt.legend()
plt.show()
```

**체크포인트**:
- [ ] 확률의 공리 3가지 설명 가능
- [ ] 조건부 확률과 베이즈 정리 유도 가능
- [ ] 기댓값의 선형성 증명 가능

#### Day 3-4: 주요 확률 분포
**학습 내용**:
- Bernoulli, Binomial, Categorical
- Gaussian, Multivariate Gaussian
- 각 분포의 평균, 분산, 모멘트

**실습**:
```python
from scipy import stats

# 다양한 분포 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Bernoulli
x = [0, 1]
pmf = stats.bernoulli.pmf(x, 0.3)
axes[0,0].bar(x, pmf)
axes[0,0].set_title('Bernoulli(p=0.3)')

# Binomial
x = np.arange(0, 20)
pmf = stats.binom.pmf(x, n=20, p=0.3)
axes[0,1].bar(x, pmf)
axes[0,1].set_title('Binomial(n=20, p=0.3)')

# Normal
x = np.linspace(-4, 4, 100)
pdf = stats.norm.pdf(x, 0, 1)
axes[0,2].plot(x, pdf)
axes[0,2].set_title('Normal(μ=0, σ=1)')

# ... 더 많은 분포
plt.tight_layout()
plt.show()
```

**미니 프로젝트**:
- 다양한 분포에서 샘플링하여 히스토그램으로 이론적 분포 확인

#### Day 5-7: 정보 이론
**학습 내용**:
- Entropy (Shannon Entropy)
- Cross-Entropy
- KL Divergence
- Mutual Information

**핵심 수식**:
```
Entropy: H(X) = -Σ p(x) log p(x)
Cross-Entropy: H(p,q) = -Σ p(x) log q(x)
KL Divergence: D_KL(p||q) = Σ p(x) log(p(x)/q(x))
```

**실습**:
```python
def entropy(p):
    """Shannon Entropy"""
    return -np.sum(p * np.log2(p + 1e-10))

def kl_divergence(p, q):
    """KL Divergence"""
    return np.sum(p * np.log2((p + 1e-10) / (q + 1e-10)))

def cross_entropy(p, q):
    """Cross Entropy"""
    return -np.sum(p * np.log2(q + 1e-10))

# 예제
p = np.array([0.1, 0.2, 0.7])
q = np.array([0.3, 0.3, 0.4])

print(f"H(p) = {entropy(p):.4f}")
print(f"D_KL(p||q) = {kl_divergence(p, q):.4f}")
print(f"H(p,q) = {cross_entropy(p, q):.4f}")
print(f"H(p,q) = H(p) + D_KL(p||q) = {entropy(p) + kl_divergence(p, q):.4f}")
```

**체크포인트**:
- [ ] KL Divergence가 비대칭적임을 이해
- [ ] Cross-Entropy와 KL Divergence의 관계 설명 가능
- [ ] Entropy의 물리적 의미 이해

**Week 1 과제**:
1. 베이즈 정리를 이용한 스팸 필터 구현
2. 다양한 분포의 Entropy 계산 및 비교
3. 블로그 포스트: "정보 이론과 머신러닝"

---

### Week 2: 선형대수 및 최적화

#### Day 1-3: 선형대수
**학습 내용**:
- 벡터/행렬 연산
- 고유값 분해 (Eigenvalue Decomposition)
- 특이값 분해 (SVD)
- 행렬의 rank, null space

**실습**:
```python
import numpy as np
from numpy.linalg import eig, svd

# 고유값 분해
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# 재구성 확인
A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")

# SVD
U, S, Vt = svd(A)
print(f"Singular values: {S}")

# 이미지 압축 예제 (SVD 활용)
from PIL import Image
img = np.array(Image.open('image.jpg').convert('L'))

U, S, Vt = svd(img, full_matrices=False)

# 상위 k개 특이값만 사용
k = 50
img_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(img_compressed, cmap='gray')
axes[1].set_title(f'Compressed (k={k})')
plt.show()
```

#### Day 4-7: 최적화
**학습 내용**:
- Gradient Descent
- Newton's Method
- Constrained Optimization
- Lagrange Multipliers

**실습**:
```python
def gradient_descent(f, grad_f, x0, lr=0.01, n_iter=1000):
    """
    경사 하강법

    Args:
        f: 목적 함수
        grad_f: 목적 함수의 그래디언트
        x0: 초기값
        lr: 학습률
        n_iter: 반복 횟수
    """
    x = x0.copy()
    history = [x.copy()]

    for i in range(n_iter):
        grad = grad_f(x)
        x = x - lr * grad
        history.append(x.copy())

    return x, np.array(history)

# 예제: f(x,y) = x^2 + y^2 최소화
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([3.0, 3.0])
x_opt, history = gradient_descent(f, grad_f, x0, lr=0.1, n_iter=50)

# 시각화
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

plt.contour(X, Y, Z, levels=20)
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=4)
plt.plot(x0[0], x0[1], 'g*', markersize=15, label='Start')
plt.plot(x_opt[0], x_opt[1], 'r*', markersize=15, label='End')
plt.legend()
plt.title('Gradient Descent Path')
plt.show()
```

**체크포인트**:
- [ ] 학습률이 너무 크거나 작을 때의 문제점 이해
- [ ] Momentum의 동작 원리 설명 가능
- [ ] 볼록 함수와 비볼록 함수의 차이 이해

---

## Phase 2: 딥러닝 기초 (4-5주)

### Week 5: 신경망 기초

#### Day 1-2: PyTorch 마스터하기
**학습 내용**:
- Tensor 연산
- Autograd 시스템
- nn.Module 이해
- Dataset과 DataLoader

**실습 - 처음부터 MLP 구현**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 학습 루프
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(dataloader), correct / total

# 평가
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(dataloader), correct / total
```

#### Day 3-5: 역전파 직접 구현
**학습 목표**: Autograd 없이 역전파 구현하여 원리 완전 이해

```python
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient

class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

# 네트워크 구성 및 학습
# ... (구현)
```

**체크포인트**:
- [ ] 역전파의 체인 룰 완벽 이해
- [ ] Vanishing/Exploding Gradient 문제 설명 가능
- [ ] 다양한 활성화 함수의 장단점 설명 가능

#### Day 6-7: 고급 학습 기법
**학습 내용**:
- Learning Rate Scheduling
- Gradient Clipping
- Mixed Precision Training
- Distributed Training 기초

---

### Week 6-8: CNN 마스터하기

#### 프로젝트 기반 학습

**프로젝트 1: CIFAR-10 분류기 (Week 6)**
```python
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# 데이터 증강
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# ResNet-like 모델
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 전체 네트워크 구현...
```

**목표**:
- Test Accuracy 90% 이상 달성
- TensorBoard로 학습 과정 시각화
- 다양한 Data Augmentation 실험

**프로젝트 2: Transfer Learning (Week 7)**
- ImageNet pretrained 모델 fine-tuning
- 커스텀 데이터셋에 적용
- Feature Extraction vs Fine-tuning 비교

---

## Phase 3: Generative Models - 기본

### Week 12-14: VAE 완전 정복

#### Day 1-3: VAE 이론 깊이 이해

**핵심 개념**:

1. **잠재 변수 모델**
```
생성 과정: z ~ p(z), x ~ p(x|z)
우리가 모르는 것: p(z|x) (사후 분포)
근사: q(z|x) ≈ p(z|x)
```

2. **ELBO 유도**
```
log p(x) = ELBO + KL(q(z|x)||p(z|x))
ELBO = E_q[log p(x|z)] - KL(q(z|x)||p(z))
     = Reconstruction Loss - KL Divergence
```

3. **Reparameterization Trick**
```
z ~ N(μ, σ²)를 z = μ + σ * ε (ε ~ N(0,1))로 변환
→ 그래디언트가 μ, σ를 통과할 수 있음
```

**수식 유도 연습**:
```python
"""
ELBO 유도 단계별 증명:

1. log p(x) = log ∫ p(x,z) dz
2. = log ∫ [p(x,z) / q(z|x)] * q(z|x) dz
3. = log E_q[p(x,z) / q(z|x)]
4. ≥ E_q[log p(x,z) / q(z|x)]  (Jensen's inequality)
5. = E_q[log p(x|z) + log p(z) - log q(z|x)]
6. = E_q[log p(x|z)] - KL(q(z|x)||p(z))
"""
```

#### Day 4-7: VAE 구현 - 3가지 버전

**버전 1: Vanilla VAE**
```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
```

**버전 2: Convolutional VAE**
```python
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 4x4 -> 2x2
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    # ... (forward 등 구현)
```

**버전 3: β-VAE (Disentangled Representations)**
```python
def beta_vae_loss(recon_x, x, mu, logvar, beta=4.0):
    """
    β-VAE loss with controllable disentanglement

    β > 1: 더 많은 disentanglement, 낮은 reconstruction quality
    β < 1: 더 나은 reconstruction, 낮은 disentanglement
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD
```

#### Week 13-14: VAE 고급 실험

**실험 1: Latent Space Interpolation**
```python
def interpolate_latent_space(vae, img1, img2, n_steps=10):
    """두 이미지 사이의 latent space에서 보간"""
    vae.eval()

    with torch.no_grad():
        mu1, _ = vae.encode(img1)
        mu2, _ = vae.encode(img2)

        # Linear interpolation
        alphas = torch.linspace(0, 1, n_steps)
        interpolated_images = []

        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            img = vae.decode(z)
            interpolated_images.append(img)

    return torch.cat(interpolated_images)
```

**실험 2: Latent Space Arithmetic**
```python
# 예: "안경 쓴 남자" - "남자" + "여자" = "안경 쓴 여자"
def latent_arithmetic(vae, img_with_glasses_man, img_man, img_woman):
    with torch.no_grad():
        z_with_glasses, _ = vae.encode(img_with_glasses_man)
        z_man, _ = vae.encode(img_man)
        z_woman, _ = vae.encode(img_woman)

        z_result = z_with_glasses - z_man + z_woman
        return vae.decode(z_result)
```

**실험 3: Disentanglement 분석**
```python
def analyze_disentanglement(vae, dataset, latent_dim):
    """
    각 latent dimension을 순회하며 어떤 특징이 변화하는지 분석
    """
    vae.eval()

    # 평균 latent vector 계산
    mean_z = torch.zeros(latent_dim)

    for dim in range(latent_dim):
        # 해당 dimension만 변화
        z = mean_z.clone()
        values = torch.linspace(-3, 3, 10)

        images = []
        for val in values:
            z[dim] = val
            img = vae.decode(z.unsqueeze(0))
            images.append(img)

        # 시각화
        plot_images(images, title=f'Dimension {dim}')
```

---

## Phase 4: Diffusion Models 완전 정복

### Week 24-26: Diffusion Models

#### Week 24: DDPM 이론

**Day 1-2: Forward Process 이해**

```python
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear schedule for β_t"""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """Improved Cosine schedule"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DiffusionForwardProcess:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps

        # β schedule
        self.betas = cosine_beta_schedule(timesteps)

        # α = 1 - β
        self.alphas = 1.0 - self.betas

        # α̅_t = ∏(1 to t) α_i
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # α̅_{t-1}
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # sqrt(α̅_t)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # sqrt(1 - α̅_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0)
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
```

**Day 3-5: Reverse Process & Training**

```python
class UNet(nn.Module):
    """Simplified U-Net for DDPM"""
    def __init__(self, in_channels=3, model_channels=128, num_res_blocks=2):
        super().__init__()
        # ... U-Net 구현 (생략)

    def forward(self, x, t):
        """
        Args:
            x: noisy image [B, C, H, W]
            t: timestep [B]
        Returns:
            noise prediction [B, C, H, W]
        """
        # ... 구현

def train_ddpm(model, dataloader, diffusion, optimizer, device, epochs):
    """DDPM Training"""
    model.train()

    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            batch_size = x.size(0)

            # Random timestep
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)

            # Forward diffusion (add noise)
            noise = torch.randn_like(x)
            x_noisy = diffusion.q_sample(x, t, noise)

            # Predict noise
            noise_pred = model(x_noisy, t)

            # Loss (simple MSE)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

#### Week 25: DDIM & Sampling

**DDIM Sampling (Fast Sampling)**
```python
@torch.no_grad()
def ddim_sample(model, diffusion, shape, num_inference_steps=50, eta=0.0):
    """
    DDIM sampling - much faster than DDPM

    Args:
        eta: 0 = deterministic, 1 = stochastic (DDPM)
    """
    device = next(model.parameters()).device
    batch_size = shape[0]

    # Start from pure noise
    x = torch.randn(shape, device=device)

    # Create inference timesteps
    timesteps = torch.linspace(
        diffusion.timesteps - 1, 0, num_inference_steps, dtype=torch.long
    )

    for i, t in enumerate(timesteps):
        # Predict noise
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch)

        # Get parameters
        alpha_prod_t = diffusion.alphas_cumprod[t]

        if i < len(timesteps) - 1:
            alpha_prod_t_prev = diffusion.alphas_cumprod[timesteps[i + 1]]
        else:
            alpha_prod_t_prev = torch.tensor(1.0)

        # Predict x_0
        pred_original_sample = (x - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prod_t_prev - eta**2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)) * noise_pred

        # x_{t-1}
        x = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + dir_xt

        # Add noise
        if eta > 0:
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            sigma = eta * torch.sqrt(variance)
            x = x + sigma * torch.randn_like(x)

    return x
```

#### Week 26: Conditional Diffusion

**Classifier-Free Guidance**
```python
class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10, class_emb_dim=128):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        # ... U-Net 구현

    def forward(self, x, t, y=None):
        """
        Args:
            y: class labels (None for unconditional)
        """
        if y is not None:
            class_emb = self.class_emb(y)
            # Inject into U-Net
        # ... 구현

@torch.no_grad()
def classifier_free_guidance_sample(model, shape, y, guidance_scale=7.5):
    """
    Classifier-Free Guidance Sampling

    ε̃ = ε_uncond + s * (ε_cond - ε_uncond)
    """
    batch_size = shape[0]
    x = torch.randn(shape)

    for t in reversed(range(diffusion.timesteps)):
        t_batch = torch.full((batch_size,), t)

        # Conditional prediction
        noise_cond = model(x, t_batch, y)

        # Unconditional prediction
        noise_uncond = model(x, t_batch, None)

        # Guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Denoise step
        x = diffusion.p_sample(x, t, noise_pred)

    return x
```

---

## 학습 체크리스트

### Phase 1 완료 조건
- [ ] 베이즈 정리를 이용한 문제 풀이 (5문제 이상)
- [ ] KL Divergence 유도 및 구현
- [ ] SVD를 이용한 이미지 압축 구현
- [ ] Gradient Descent 3가지 변형 구현
- [ ] 블로그 포스트 2개 작성

### Phase 2 완료 조건
- [ ] 역전파 알고리즘 처음부터 구현
- [ ] MNIST 98% 이상 달성
- [ ] CIFAR-10 90% 이상 달성
- [ ] Transfer Learning 실습
- [ ] CNN 시각화 (CAM, Filter Visualization)

### Phase 3 완료 조건
- [ ] VAE ELBO 수식 유도
- [ ] 3가지 VAE 변형 구현
- [ ] Latent Space 분석 실험
- [ ] Flow 모델 1개 이상 구현
- [ ] 논문 5편 이상 읽기

### Phase 4 완료 조건
- [ ] GAN 학습 안정화 기법 3가지 실험
- [ ] WGAN-GP 구현
- [ ] FID Score 계산 구현
- [ ] DDPM 구현 및 학습
- [ ] Classifier-Free Guidance 구현
- [ ] 논문 10편 이상 읽기

---

**다음 페이지: [Interactive 웹페이지로 이동](../index.html)**
