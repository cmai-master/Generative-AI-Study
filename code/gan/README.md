## GAN (Generative Adversarial Networks) 구현

이 디렉토리는 DCGAN과 WGAN-GP의 완전한 구현을 제공합니다.

## 📁 파일 구조

```
gan/
├── __init__.py         # 패키지 초기화
├── dcgan.py            # DCGAN 구현
├── wgan_gp.py          # WGAN-GP 구현
├── train_gan.py        # 학습 스크립트
└── README.md           # 이 파일
```

## 🎯 구현된 모델

### 1. DCGAN
- **논문**: Radford et al. (2015) "Unsupervised Representation Learning with DCGAN"
- **특징**:
  - Convolutional 레이어 기반
  - Batch Normalization
  - ReLU/LeakyReLU 활성화 함수
  - Transposed Convolution for Generator

### 2. WGAN-GP
- **논문**: Gulrajani et al. (2017) "Improved Training of Wasserstein GANs"
- **특징**:
  - Wasserstein Distance 사용
  - Gradient Penalty로 Lipschitz Constraint
  - 더 안정적인 학습
  - Mode Collapse 완화

## 🚀 빠른 시작

### DCGAN 학습

```bash
cd code/gan
python train_gan.py --model dcgan --dataset mnist --epochs 100
```

### WGAN-GP 학습

```bash
python train_gan.py --model wgan_gp --dataset mnist --epochs 100 --n-critic 5
```

### CIFAR-10 학습

```bash
python train_gan.py --model dcgan --dataset cifar10 --epochs 200 --batch-size 64
```

## 📊 명령행 옵션

### 모델 설정
- `--model`: 모델 타입 (`dcgan` 또는 `wgan_gp`)
- `--latent-dim`: 잠재 벡터 차원 (기본값: 100)
- `--feature-map-size`: Feature map 크기 (기본값: 64)

### WGAN-GP 전용
- `--lambda-gp`: Gradient Penalty 가중치 (기본값: 10.0)
- `--n-critic`: Generator 1회당 Critic 학습 횟수 (기본값: 5)

### 학습 설정
- `--epochs`: 학습 에포크 수 (기본값: 100)
- `--batch-size`: 배치 크기 (기본값: 128)
- `--lr-g`: Generator 학습률 (기본값: 0.0002)
- `--lr-d`: Discriminator 학습률 (기본값: 0.0002)

## 💻 코드 사용 예제

```python
from gan import DCGAN, WGAN_GP

# DCGAN 생성
dcgan = DCGAN(latent_dim=100, num_channels=1, device='cuda')

# 이미지 생성
samples = dcgan.generate(num_samples=64)

# WGAN-GP 생성
wgan_gp = WGAN_GP(latent_dim=100, num_channels=3, lambda_gp=10.0, device='cuda')
samples = wgan_gp.generate(num_samples=64)
```

## 📈 예상 결과

### MNIST (100 epochs)

**DCGAN**:
- D Loss: ~0.5-1.5
- G Loss: ~1.0-3.0
- 학습 시간: ~30-60분 (GPU)

**WGAN-GP**:
- Critic Loss: -5.0 ~ -20.0 (음수는 정상)
- Generator Loss: -5.0 ~ -15.0
- 학습 시간: ~60-120분 (GPU, n_critic=5)

## 🔧 문제 해결

### Mode Collapse 발생 시
```bash
# WGAN-GP 사용
python train_gan.py --model wgan_gp --n-critic 5 --lambda-gp 10
```

### 학습 불안정
```bash
# 학습률 낮추기
python train_gan.py --lr-g 0.0001 --lr-d 0.0001

# 배치 크기 늘리기
python train_gan.py --batch-size 256
```

## 📚 참고 자료

- Radford et al. (2015) - [DCGAN](https://arxiv.org/abs/1511.06434)
- Gulrajani et al. (2017) - [WGAN-GP](https://arxiv.org/abs/1704.00028)

---

**작성일**: 2025-11-14
**버전**: 1.0
