# Diffusion Models 구현

이 디렉토리는 DDPM과 DDIM의 완전한 구현을 제공합니다.

## 📁 파일 구조

```
diffusion/
├── __init__.py           # 패키지 초기화
├── noise_schedule.py     # Noise scheduling (linear, cosine, sigmoid)
├── unet.py               # U-Net 아키텍처
├── ddpm.py               # DDPM 구현
├── ddim.py               # DDIM 샘플링
└── README.md             # 이 파일
```

## 🎯 구현된 모델

### 1. DDPM (Denoising Diffusion Probabilistic Models)
- **논문**: Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
- **특징**:
  - Forward diffusion: 점진적 노이즈 추가
  - Reverse diffusion: 학습된 U-Net으로 노이즈 제거
  - Loss: MSE(ε - ε_θ(x_t, t))
  - 1000 steps 샘플링

### 2. DDIM (Denoising Diffusion Implicit Models)
- **논문**: Song et al. (2020) "Denoising Diffusion Implicit Models"
- **특징**:
  - Non-Markovian process
  - Deterministic sampling (η=0)
  - 50 steps로 고품질 생성 (20배 빠름)
  - DDPM 모델을 그대로 사용

### 3. U-Net Architecture
- Time embedding (sinusoidal positional encoding)
- ResNet blocks with time conditioning
- Self-attention layers
- Skip connections

### 4. Noise Schedules
- Linear schedule
- Cosine schedule (Improved DDPM)
- Sigmoid schedule

## 🚀 빠른 시작

### DDPM 학습

```python
from diffusion import DDPM, UNet
import torch

# U-Net 생성
unet = UNet(
    in_channels=3,
    out_channels=3,
    model_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(16, 8)
)

# DDPM 모델 생성
ddpm = DDPM(
    model=unet,
    timesteps=1000,
    schedule_type='cosine',
    objective='noise'
).to('cuda')

# 학습
optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-4)

for images, _ in dataloader:
    images = images.to('cuda')
    loss = ddpm(images)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### DDPM 샘플링 (1000 steps)

```python
# 샘플 생성
samples = ddpm.sample(
    batch_size=16,
    channels=3,
    image_size=64,
    device='cuda',
    show_progress=True
)
```

### DDIM 샘플링 (50 steps, 20배 빠름)

```python
from diffusion import DDIMSampler

# DDIM sampler 생성
ddim = DDIMSampler(ddpm)

# 빠른 샘플 생성 (50 steps)
samples = ddim.sample_from_batch(
    batch_size=16,
    channels=3,
    image_size=64,
    num_steps=50,
    eta=0.0,  # deterministic
    device='cuda'
)
```

## 📊 샘플링 속도 비교

| 방법 | Steps | 시간 (상대적) | 품질 |
|------|-------|---------------|------|
| DDPM | 1000 | 1.0x | ⭐⭐⭐⭐⭐ |
| DDIM (η=0) | 100 | 0.1x | ⭐⭐⭐⭐⭐ |
| DDIM (η=0) | 50 | 0.05x | ⭐⭐⭐⭐ |
| DDIM (η=0) | 20 | 0.02x | ⭐⭐⭐ |

## 💻 주요 파라미터

### Noise Schedule
- `timesteps`: 총 timestep 수 (기본값: 1000)
- `schedule_type`: 'linear', 'cosine', 'sigmoid'
- `beta_start`: β 시작값 (0.0001)
- `beta_end`: β 끝값 (0.02)

### U-Net
- `model_channels`: 기본 채널 수 (128)
- `channel_mult`: 채널 배수 (1, 2, 2, 2)
- `num_res_blocks`: ResBlock 수 (2)
- `attention_resolutions`: Attention 적용 해상도 (16, 8)
- `num_heads`: Attention head 수 (4)

### DDPM
- `objective`: 'noise' (노이즈 예측) 또는 'x0' (x_0 예측)

### DDIM
- `num_steps`: 샘플링 스텝 수 (50)
- `eta`: Stochasticity (0.0 = deterministic, 1.0 = DDPM)

## 📈 예상 결과

### MNIST (64x64, 50 epochs)
- Train Loss: ~0.01-0.02
- 샘플 품질: 선명한 숫자 생성
- 학습 시간: ~2-3시간 (GPU)

### CIFAR-10 (32x32, 200 epochs)
- Train Loss: ~0.02-0.04
- 샘플 품질: 다양한 객체 생성
- 학습 시간: ~10-15시간 (GPU)

## 🔧 문제 해결

### Out of Memory
```python
# 모델 크기 줄이기
unet = UNet(model_channels=64, channel_mult=(1, 2, 2))

# 배치 크기 줄이기
samples = ddpm.sample(batch_size=4, ...)

# Gradient checkpointing 사용 (메모리 절약)
```

### 학습 불안정
```python
# Cosine schedule 사용
ddpm = DDPM(schedule_type='cosine')

# 학습률 낮추기
optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

# EMA (Exponential Moving Average) 사용
```

### 샘플 품질 낮음
```python
# 더 많은 샘플링 steps
samples = ddim.sample(..., num_steps=100)

# Stochastic sampling
samples = ddim.sample(..., eta=0.5)

# 더 오래 학습
# epochs 늘리기
```

## 📚 참고 자료

### 논문
- Ho et al. (2020) - [DDPM](https://arxiv.org/abs/2006.11239)
- Song et al. (2020) - [DDIM](https://arxiv.org/abs/2010.02502)
- Nichol & Dhariwal (2021) - [Improved DDPM](https://arxiv.org/abs/2102.09672)

### 블로그
- [Lil'Log - What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Hugging Face - The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

## 🎨 고급 기능

### Classifier-Free Guidance (CFG)
```python
# TODO: Implement conditional generation with CFG
```

### Latent Diffusion
```python
# TODO: Implement diffusion in latent space (Stable Diffusion style)
```

## 🧪 실험 아이디어

1. **Noise Schedule 비교**
   - Linear vs Cosine vs Sigmoid
   - 샘플 품질 및 학습 속도 비교

2. **DDIM Step 수 실험**
   - 10, 20, 50, 100, 500 steps
   - 품질 vs 속도 trade-off 분석

3. **Objective 비교**
   - Noise prediction vs x_0 prediction
   - 수렴 속도 및 최종 품질 비교

4. **Architecture 실험**
   - U-Net 크기 변화
   - Attention resolution 최적화

---

**작성일**: 2025-11-14
**버전**: 1.0
**라이선스**: MIT
