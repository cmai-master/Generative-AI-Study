# 벤치마크 결과 및 실험 분석

> **목적**: 구현된 생성 모델들의 성능을 정량적으로 평가하고 분석합니다.
>
> **모델**: VAE, β-VAE, DCGAN, WGAN-GP, DDPM, DDIM

---

## 📊 실험 환경

### 하드웨어
- **CPU**: Intel Xeon / AMD Ryzen
- **GPU**: NVIDIA RTX 3090 / RTX 4090 / A100 (권장)
- **RAM**: 32GB+
- **Storage**: SSD 100GB+

### 소프트웨어
- **Python**: 3.9+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+

### 데이터셋
- **MNIST**: 28x28 grayscale, 60K train / 10K test
- **Fashion-MNIST**: 28x28 grayscale, 60K train / 10K test
- **CIFAR-10**: 32x32 RGB, 50K train / 10K test
- **CelebA**: 64x64 RGB (cropped & aligned), 200K images

---

## 🎯 평가 지표

### 1. Inception Score (IS)
- **목적**: 생성 이미지의 품질과 다양성 평가
- **계산**: IS = exp(E[KL(p(y|x) || p(y))])
- **범위**: 1.0 ~ ∞ (높을수록 좋음)
- **해석**:
  - 높은 IS = 선명하고 다양한 이미지
  - 낮은 IS = 흐릿하거나 다양성 부족

### 2. Fréchet Inception Distance (FID)
- **목적**: 생성 분포와 실제 분포의 거리 측정
- **계산**: FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real·Σ_fake))
- **범위**: 0 ~ ∞ (낮을수록 좋음)
- **해석**:
  - FID < 10: 매우 우수
  - FID < 30: 우수
  - FID < 50: 양호
  - FID > 100: 불량

### 3. Reconstruction Loss (VAE)
- **목적**: 재구성 품질 평가
- **계산**: BCE(x, x̂)
- **범위**: 0 ~ ∞ (낮을수록 좋음)

### 4. Training Time
- **목적**: 학습 효율성 평가
- **단위**: 시간 (hours)

### 5. Sampling Time
- **목적**: 생성 속도 평가
- **단위**: 초/이미지

---

## 📈 VAE 실험 결과

### MNIST 실험

#### 설정
- **모델**: Basic VAE
- **Latent Dim**: 20
- **Hidden Dims**: [512, 256]
- **Batch Size**: 128
- **Learning Rate**: 1e-3
- **Epochs**: 50
- **Optimizer**: Adam

#### 예상 결과

| Metric | Value | Notes |
|--------|-------|-------|
| **Train Loss** | 105-110 | ELBO |
| **Test Loss** | 110-115 | ELBO |
| **Reconstruction Loss** | 85-95 | BCE |
| **KL Divergence** | 15-20 | |
| **Training Time** | 15-20 min | RTX 3090 |
| **Params** | ~2.5M | |

#### β-VAE 비교 (MNIST)

| β | Recon Loss | KL Div | Disentanglement | Total Loss |
|---|------------|--------|-----------------|------------|
| **0.5** | 82-88 | 25-30 | ⭐⭐ | 95-105 |
| **1.0** | 85-95 | 15-20 | ⭐⭐⭐ | 105-110 |
| **4.0** | 95-105 | 5-10 | ⭐⭐⭐⭐⭐ | 115-125 |
| **10.0** | 105-115 | 2-5 | ⭐⭐⭐⭐⭐ | 125-140 |

**관찰**:
- β ↑ → KL Divergence ↓, Disentanglement ↑
- β ↑ → Reconstruction Quality ↓
- **최적 β**: 4.0 (품질과 disentanglement 균형)

### Fashion-MNIST 실험

#### 예상 결과

| Metric | Value |
|--------|-------|
| **Train Loss** | 120-130 |
| **Test Loss** | 125-135 |
| **Reconstruction Loss** | 95-110 |
| **KL Divergence** | 15-25 |
| **Training Time** | 20-25 min |

**관찰**:
- Fashion-MNIST가 MNIST보다 복잡 → 손실 약간 높음
- 옷의 텍스처가 숫자보다 재구성 어려움

---

## 📈 GAN 실험 결과

### DCGAN on MNIST (64x64)

#### 설정
- **Latent Dim**: 100
- **Feature Map Size**: 64
- **Batch Size**: 128
- **Learning Rate**: 2e-4 (G & D)
- **Beta1**: 0.5
- **Epochs**: 100
- **Optimizer**: Adam

#### 예상 결과

| Metric | Value | Notes |
|--------|-------|-------|
| **D Loss** | 0.5-1.5 | |
| **G Loss** | 1.0-3.0 | |
| **FID** | 15-30 | |
| **IS** | 2.5-3.5 | |
| **Training Time** | 40-60 min | RTX 3090 |
| **Params (G)** | ~3.5M | |
| **Params (D)** | ~2.8M | |

**관찰**:
- D Loss와 G Loss가 진동하며 균형 유지
- Mode collapse 거의 없음 (모든 숫자 생성)

### WGAN-GP vs DCGAN (MNIST)

| Metric | DCGAN | WGAN-GP | Winner |
|--------|-------|---------|--------|
| **Training Stability** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | WGAN-GP |
| **FID** | 20-30 | 15-25 | WGAN-GP |
| **IS** | 2.8-3.5 | 3.0-3.8 | WGAN-GP |
| **Mode Collapse** | Rare | Very Rare | WGAN-GP |
| **Training Time** | 1.0x | 2.5x | DCGAN |
| **Convergence** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | WGAN-GP |

**결론**:
- **WGAN-GP**: 더 안정적, 고품질, 느림
- **DCGAN**: 빠름, 좋은 품질, 가끔 불안정

### CIFAR-10 실험

#### DCGAN 결과

| Metric | Value |
|--------|-------|
| **FID** | 40-60 |
| **IS** | 5.0-6.5 |
| **Training Time** | 3-5 hours |
| **Epochs** | 200 |

#### WGAN-GP 결과

| Metric | Value |
|--------|-------|
| **FID** | 35-50 |
| **IS** | 5.5-7.0 |
| **Training Time** | 8-12 hours |
| **Epochs** | 200 |

**관찰**:
- CIFAR-10이 훨씬 어려움 (다양한 객체, 배경)
- FID 높음 → 개선 여지 많음

---

## 📈 Diffusion Models 실험 결과

### DDPM on MNIST (32x32)

#### 설정
- **Model**: U-Net (model_channels=128)
- **Timesteps**: 1000
- **Schedule**: Cosine
- **Batch Size**: 128
- **Learning Rate**: 1e-4
- **Epochs**: 50
- **Optimizer**: Adam

#### 예상 결과

| Metric | Value | Notes |
|--------|-------|-------|
| **Train Loss** | 0.015-0.025 | MSE |
| **Test Loss** | 0.018-0.028 | MSE |
| **FID** | 5-15 | Excellent |
| **IS** | 8.0-9.5 | Very Good |
| **Sampling Time (1000 steps)** | 30-60s / image | Slow |
| **Training Time** | 2-3 hours | RTX 3090 |
| **Params** | ~30M | |

### DDIM vs DDPM Sampling Speed

#### MNIST (32x32, batch_size=16)

| Method | Steps | Time | Quality (FID) | Speedup |
|--------|-------|------|---------------|---------|
| **DDPM** | 1000 | 60s | 5-10 | 1.0x |
| **DDIM (η=0)** | 100 | 6s | 5-12 | **10x** ⚡ |
| **DDIM (η=0)** | 50 | 3s | 8-15 | **20x** ⚡ |
| **DDIM (η=0)** | 20 | 1.2s | 15-25 | **50x** ⚡ |

**관찰**:
- DDIM 50 steps: 품질 거의 동일, 20배 빠름
- DDIM 20 steps: 약간 품질 저하, 50배 빠름
- **최적**: DDIM 50-100 steps

### Noise Schedule 비교

| Schedule | FID | IS | Convergence Speed |
|----------|-----|----|--------------------|
| **Linear** | 12-18 | 8.0-8.5 | ⭐⭐⭐ |
| **Cosine** | 8-15 | 8.5-9.5 | ⭐⭐⭐⭐⭐ |
| **Sigmoid** | 10-16 | 8.2-9.0 | ⭐⭐⭐⭐ |

**결론**: Cosine schedule이 가장 우수

### CIFAR-10 실험

#### DDPM 결과

| Metric | Value |
|--------|-------|
| **FID** | 15-25 |
| **IS** | 7.5-8.5 |
| **Training Time** | 12-18 hours |
| **Epochs** | 200 |

**관찰**:
- Diffusion이 GAN보다 FID 낮음 (더 좋음)
- 학습 시간 길지만 안정적

---

## 📊 모델 비교 (MNIST)

### 품질 비교

| Model | FID ↓ | IS ↑ | Recon Loss ↓ | Quality |
|-------|-------|------|--------------|---------|
| **VAE** | - | - | 85-95 | ⭐⭐⭐ |
| **β-VAE (β=4)** | - | - | 95-105 | ⭐⭐⭐ |
| **DCGAN** | 20-30 | 3.0-3.5 | - | ⭐⭐⭐⭐ |
| **WGAN-GP** | 15-25 | 3.2-3.8 | - | ⭐⭐⭐⭐ |
| **DDPM** | 5-15 | 8.5-9.5 | - | ⭐⭐⭐⭐⭐ |

**순위**: DDPM > WGAN-GP > DCGAN > VAE

### 학습 시간 비교 (50 epochs)

| Model | Training Time | Speed |
|-------|---------------|-------|
| **VAE** | 15-20 min | ⭐⭐⭐⭐⭐ |
| **β-VAE** | 15-20 min | ⭐⭐⭐⭐⭐ |
| **DCGAN** | 40-60 min | ⭐⭐⭐⭐ |
| **WGAN-GP** | 100-150 min | ⭐⭐⭐ |
| **DDPM** | 120-180 min | ⭐⭐ |

**순위**: VAE > DCGAN > WGAN-GP > DDPM

### 샘플링 시간 비교

| Model | Time per Image | Speed |
|-------|----------------|-------|
| **VAE** | 0.001s | ⚡⚡⚡⚡⚡ |
| **DCGAN** | 0.002s | ⚡⚡⚡⚡⚡ |
| **WGAN-GP** | 0.002s | ⚡⚡⚡⚡⚡ |
| **DDPM (1000 steps)** | 3-4s | ⚡ |
| **DDIM (50 steps)** | 0.15-0.2s | ⚡⚡⚡⚡ |

**순위**: VAE/GAN > DDIM > DDPM

### 종합 평가

| Model | Quality | Speed | Stability | Diversity | 추천 용도 |
|-------|---------|-------|-----------|-----------|-----------|
| **VAE** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 빠른 프로토타입, Latent space 분석 |
| **β-VAE** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Disentanglement, 해석 가능성 |
| **DCGAN** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 빠른 고품질 생성 |
| **WGAN-GP** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 안정적 학습, 프로덕션 |
| **DDPM** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 최고 품질, 연구 |
| **DDIM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 빠른 고품질 생성, 실용 |

---

## 🔬 Ablation Studies

### VAE: Latent Dimension 영향

| Latent Dim | Recon Loss | KL Div | Total Loss | Training Time |
|------------|------------|--------|------------|---------------|
| **2** | 95-105 | 1-3 | 96-108 | ⭐⭐⭐⭐⭐ |
| **10** | 88-98 | 8-12 | 100-108 | ⭐⭐⭐⭐ |
| **20** | 85-95 | 15-20 | 105-110 | ⭐⭐⭐⭐ |
| **50** | 82-92 | 25-35 | 110-120 | ⭐⭐⭐ |

**결론**: Latent Dim 20이 최적 (품질 vs 효율성)

### GAN: Learning Rate 영향

| LR (G & D) | D Loss | G Loss | FID | Stability |
|------------|--------|--------|-----|-----------|
| **1e-5** | 0.3-0.8 | 2.5-4.0 | 25-35 | ⭐⭐⭐⭐⭐ (느림) |
| **1e-4** | 0.4-1.2 | 1.5-3.5 | 18-28 | ⭐⭐⭐⭐ |
| **2e-4** | 0.5-1.5 | 1.0-3.0 | 15-25 | ⭐⭐⭐⭐ |
| **5e-4** | 0.6-2.0 | 0.8-2.5 | 20-35 | ⭐⭐⭐ (불안정) |

**결론**: 2e-4가 최적

### Diffusion: U-Net Size 영향

| Model Channels | Params | FID | IS | Training Time |
|----------------|--------|-----|----|--------------  |
| **64** | ~7M | 12-20 | 8.0-8.8 | 1.0x |
| **128** | ~30M | 8-15 | 8.5-9.5 | 2.5x |
| **256** | ~120M | 5-12 | 9.0-10.0 | 6.0x |

**결론**: 128이 최적 (품질 vs 속도)

---

## 💡 주요 발견 (Key Findings)

### 1. 모델 선택 가이드

**빠른 프로토타입**:
- VAE (1등) ⚡
- DCGAN (2등)

**최고 품질**:
- DDPM (1등) 🏆
- DDIM (2등, 훨씬 빠름)
- WGAN-GP (3등)

**안정적 학습**:
- DDPM (1등)
- WGAN-GP (2등)
- VAE (3등)

**Disentanglement**:
- β-VAE (1등) 🏆

### 2. 하이퍼파라미터 권장값

**VAE**:
- Latent Dim: 20
- β: 1.0 (일반), 4.0 (disentanglement)
- LR: 1e-3

**GAN**:
- LR: 2e-4
- Beta1: 0.5
- n_critic (WGAN-GP): 5

**Diffusion**:
- Schedule: Cosine
- Model Channels: 128
- LR: 1e-4
- Sampling: DDIM 50 steps

### 3. 데이터셋 난이도

**쉬움**: MNIST (FID < 20)
**중간**: Fashion-MNIST (FID 20-40)
**어려움**: CIFAR-10 (FID 40-80)
**매우 어려움**: CelebA (FID 80-150)

---

## 📝 실험 재현 가이드

### VAE 실험
```bash
python code/vae/train_vae.py --model vae --latent-dim 20 --epochs 50
python code/vae/train_vae.py --model beta_vae --beta 4.0 --epochs 50
```

### GAN 실험
```bash
python code/gan/train_gan.py --model dcgan --dataset mnist --epochs 100
python code/gan/train_gan.py --model wgan_gp --dataset mnist --epochs 100
```

### Diffusion 실험
```bash
# DDPM 학습 (구현 필요)
# DDIM 샘플링 테스트
```

---

## 🎯 향후 개선 방향

### 단기 (1-2주)
- [ ] 실제 학습 실행 및 결과 수집
- [ ] FID/IS 자동 계산 스크립트
- [ ] 시각화 자동 생성

### 중기 (1-2개월)
- [ ] Classifier-Free Guidance (CFG)
- [ ] Latent Diffusion Models
- [ ] StyleGAN 구현

### 장기 (3-6개월)
- [ ] Text-to-Image 모델
- [ ] Video Generation
- [ ] 3D Generation

---

**작성일**: 2025-11-14
**버전**: 1.0 (예상 결과 기반)
**상태**: 실험 대기 중

**Note**: 이 문서의 수치는 문헌 조사 및 이론적 분석을 기반으로 한 **예상 결과**입니다. 실제 학습 후 업데이트 예정입니다.
