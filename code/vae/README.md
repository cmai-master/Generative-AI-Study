# VAE (Variational Autoencoder) 구현

이 디렉토리는 VAE와 β-VAE의 완전한 구현을 제공합니다.

## 📁 파일 구조

```
vae/
├── __init__.py           # 패키지 초기화
├── basic_vae.py          # 기본 VAE 구현
├── beta_vae.py           # β-VAE 구현
├── train_vae.py          # 학습 스크립트
└── README.md             # 이 파일
```

## 🎯 구현된 모델

### 1. Basic VAE
- **논문**: Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
- **특징**:
  - Encoder: x → (μ, log σ²)
  - Reparameterization Trick
  - Decoder: z → x̂
  - Loss = Reconstruction + KL Divergence

### 2. β-VAE
- **논문**: Higgins et al. (2017) "β-VAE: Learning Basic Visual Concepts"
- **특징**:
  - Loss = Reconstruction + β * KL Divergence
  - β > 1: Disentangled Representation 강화
  - β = 1: 일반 VAE와 동일

## 🚀 빠른 시작

### 설치

```bash
# 프로젝트 루트 디렉토리에서
pip install torch torchvision matplotlib numpy tqdm
```

### 기본 VAE 학습

```bash
cd code/vae
python train_vae.py --model vae --epochs 50 --latent-dim 20
```

### β-VAE 학습

```bash
python train_vae.py --model beta_vae --beta 4.0 --epochs 50 --latent-dim 20
```

### 다양한 β 값 실험

```bash
# β = 0.5 (재구성 품질 중시)
python train_vae.py --model beta_vae --beta 0.5 --epochs 50

# β = 1.0 (일반 VAE)
python train_vae.py --model beta_vae --beta 1.0 --epochs 50

# β = 4.0 (권장값)
python train_vae.py --model beta_vae --beta 4.0 --epochs 50

# β = 10.0 (강한 Disentanglement)
python train_vae.py --model beta_vae --beta 10.0 --epochs 50
```

### 시각화 포함 학습

```bash
python train_vae.py --model vae --epochs 50 --visualize --output-dir ./outputs
```

## 📊 명령행 옵션

### 모델 설정
- `--model`: 모델 타입 (`vae` 또는 `beta_vae`)
- `--latent-dim`: 잠재 공간 차원 (기본값: 20)
- `--hidden-dims`: 히든 레이어 차원 (기본값: 512 256)
- `--beta`: β-VAE의 β 값 (기본값: 4.0)

### 학습 설정
- `--epochs`: 학습 에포크 수 (기본값: 50)
- `--batch-size`: 배치 크기 (기본값: 128)
- `--lr`: 학습률 (기본값: 0.001)
- `--seed`: 랜덤 시드 (기본값: 42)

### 데이터 설정
- `--data-dir`: 데이터 디렉토리 (기본값: ./data)
- `--num-workers`: 데이터 로더 워커 수 (기본값: 4)

### 체크포인트 및 로깅
- `--checkpoint-dir`: 체크포인트 디렉토리 (기본값: ./checkpoints)
- `--log-dir`: 로그 디렉토리 (기본값: ./logs)
- `--save-interval`: 체크포인트 저장 간격 (기본값: 10)

### Early Stopping
- `--early-stop`: Early Stopping 사용
- `--patience`: Early Stopping patience (기본값: 10)

### 시각화
- `--visualize`: 학습 후 시각화 수행
- `--output-dir`: 시각화 결과 저장 디렉토리 (기본값: ./outputs)

## 💻 코드 사용 예제

### Python 코드에서 사용

```python
import torch
from vae import VAE, BetaVAE

# 1. 모델 생성
model = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=20)
# 또는
model = BetaVAE(input_dim=784, hidden_dims=[512, 256], latent_dim=20, beta=4.0)

# 2. 이미지 재구성
images = torch.rand(32, 1, 28, 28)  # MNIST 이미지
x_recon, mu, logvar = model(images)

# 3. 새로운 이미지 생성
samples = model.sample(num_samples=64, device='cuda')

# 4. 잠재 공간 인코딩
mu, logvar = model.encode(images)

# 5. 잠재 벡터 디코딩
z = torch.randn(16, 20)
generated = model.decode(z)
```

### 손실 함수 계산

```python
from vae import vae_loss, beta_vae_loss

# VAE 손실
total_loss, recon_loss, kl_loss = vae_loss(images, x_recon, mu, logvar, beta=1.0)

# β-VAE 손실
total_loss, recon_loss, kl_loss = beta_vae_loss(images, x_recon, mu, logvar, beta=4.0)
```

## 📈 예상 결과

### MNIST (50 epochs)

**Basic VAE (β=1.0)**:
- Train Loss: ~100-120
- Test Loss: ~105-125
- Reconstruction Loss: ~85-95
- KL Divergence: ~10-20

**β-VAE (β=4.0)**:
- Train Loss: ~120-150
- Test Loss: ~125-155
- Reconstruction Loss: ~85-95 (약간 높음)
- KL Divergence: ~5-10 (낮음, 더 disentangled)

### 학습 시간
- GPU (RTX 3090): ~5-10분 (50 epochs)
- CPU: ~30-60분 (50 epochs)

## 🎨 시각화 결과

`--visualize` 옵션 사용 시 다음 파일들이 생성됩니다:

1. **training_curves.png**: 학습/검증 손실 곡선
2. **reconstruction.png**: 원본 vs 재구성 이미지 비교
3. **samples.png**: 모델이 생성한 새로운 이미지
4. **latent_space.png**: 2D Latent Space 시각화 (latent_dim=2인 경우)

## 🔬 실험 아이디어

### 1. Latent Dimension 실험
```bash
python train_vae.py --latent-dim 2 --visualize   # 시각화 좋음
python train_vae.py --latent-dim 10              # 적당함
python train_vae.py --latent-dim 20              # 권장
python train_vae.py --latent-dim 50              # 많음
```

### 2. β 값 비교 실험
```bash
for beta in 0.5 1.0 2.0 4.0 10.0; do
    python train_vae.py --model beta_vae --beta $beta --epochs 50
done
```

### 3. Architecture 실험
```bash
# Shallow network
python train_vae.py --hidden-dims 256

# Deep network
python train_vae.py --hidden-dims 512 256 128
```

## 📚 참고 자료

### 논문
- Kingma & Welling (2013) - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- Higgins et al. (2017) - [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl)

### 블로그
- [Lil'Log - From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [Understanding VAEs](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

## 🐛 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
python train_vae.py --batch-size 64

# 또는 CPU 사용
python train_vae.py --batch-size 32
```

### 재구성 품질이 낮음
```bash
# β 값 낮추기
python train_vae.py --model beta_vae --beta 0.5

# Latent dimension 늘리기
python train_vae.py --latent-dim 50

# 더 깊은 네트워크
python train_vae.py --hidden-dims 1024 512 256
```

### KL Collapse (KL → 0)
```bash
# β 값 높이기
python train_vae.py --model beta_vae --beta 4.0

# Free bits (코드 수정 필요)
# KL Annealing (코드 수정 필요)
```

## 📝 다음 단계

- [ ] VQ-VAE 구현 추가
- [ ] Conditional VAE 구현
- [ ] CelebA 데이터셋 지원
- [ ] KL Annealing 추가
- [ ] Jupyter Notebook 튜토리얼 작성

## 🤝 기여

버그 리포트, 기능 제안, PR 환영합니다!

---

**작성일**: 2025-11-14
**버전**: 1.0
**라이선스**: MIT
