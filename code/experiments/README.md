# 실험 자동화 스크립트

이 디렉토리는 VAE, GAN, Diffusion 모델의 체계적인 실험을 위한 자동화 스크립트를 제공합니다.

## 📁 파일 구조

```
experiments/
├── run_vae_experiments.py      # VAE/β-VAE 실험
├── run_gan_experiments.py      # GAN (DCGAN, WGAN-GP) 실험
├── run_diffusion_experiments.py # Diffusion (DDPM, DDIM) 실험
├── compare_all_models.py       # 모든 모델 비교
└── README.md                   # 이 파일
```

## 🚀 빠른 시작

### 1. VAE 실험

```bash
# 모든 VAE 실험 실행 (β 비교, latent dim ablation 등)
python run_vae_experiments.py --dataset mnist --epochs 50

# 특정 실험만 실행
python run_vae_experiments.py --experiments 1 2 --dataset mnist

# CIFAR-10으로 실험
python run_vae_experiments.py --dataset cifar10 --epochs 100
```

**실험 목록:**
1. β-VAE β 값 비교 (0.5, 1.0, 4.0, 10.0)
2. Latent dimension ablation (10, 20, 50, 100)
3. Learning rate ablation (1e-5, 1e-4, 1e-3)
4. Hidden dimensions ablation

### 2. GAN 실험

```bash
# 모든 GAN 실험 실행
python run_gan_experiments.py --dataset mnist --epochs 100

# DCGAN vs WGAN-GP 비교만 실행
python run_gan_experiments.py --experiments 1 --dataset mnist

# CelebA로 실험
python run_gan_experiments.py --dataset celeba --epochs 200
```

**실험 목록:**
1. DCGAN vs WGAN-GP 비교
2. Learning rate ablation (DCGAN)
3. Architecture ablation (feature map size)
4. WGAN-GP gradient penalty weight ablation
5. Latent dimension ablation

### 3. Diffusion 실험

```bash
# DDPM 학습 실험
python run_diffusion_experiments.py --experiments 1 2 --dataset mnist --epochs 50

# DDIM 샘플링 실험 (학습된 모델 필요)
python run_diffusion_experiments.py \
  --experiments 3 4 \
  --ddpm-checkpoint ./results/diffusion_experiments/ddpm_baseline/checkpoint_epoch_50.pt
```

**실험 목록:**
1. DDPM baseline 학습
2. Noise schedule 비교 (linear, cosine, sigmoid)
3. DDIM 샘플링 스텝 수 비교 (10, 20, 50, 100, 500)
4. DDIM eta 비교 (deterministic vs stochastic)

### 4. 모든 모델 비교

```bash
# 모든 모델 비교 (FID, IS, 생성 속도)
python compare_all_models.py \
  --vae-checkpoint ./results/vae_experiments/vae_baseline/checkpoint_final.pt \
  --dcgan-checkpoint ./results/gan_experiments/dcgan_baseline/checkpoint_final.pt \
  --wgan-checkpoint ./results/gan_experiments/wgan_gp_baseline/checkpoint_final.pt \
  --ddpm-checkpoint ./results/diffusion_experiments/ddpm_baseline/checkpoint_epoch_50.pt \
  --dataset mnist
```

## 📊 실험 결과

각 실험은 다음을 생성합니다:

### 디렉토리 구조
```
results/
├── vae_experiments/
│   ├── beta_vae_beta_0.5/
│   │   ├── checkpoint_epoch_10.pt
│   │   ├── checkpoint_epoch_20.pt
│   │   ├── samples_epoch_10.png
│   │   ├── training_log.json
│   │   └── loss_curves.png
│   ├── beta_vae_beta_1.0/
│   └── ...
├── gan_experiments/
│   ├── dcgan_baseline/
│   ├── wgan_gp_baseline/
│   └── ...
├── diffusion_experiments/
│   ├── ddpm_baseline/
│   ├── ddim_steps_comparison/
│   │   ├── ddim_10_steps.png
│   │   ├── ddim_50_steps.png
│   │   └── timing_results.json
│   └── ...
└── model_comparison/
    ├── vae_samples.png
    ├── dcgan_samples.png
    ├── wgan_gp_samples.png
    ├── ddpm_samples.png
    ├── ddim_50_samples.png
    └── comparison_results.json
```

### 저장되는 파일

1. **Checkpoints** (`*.pt`): 모델 가중치, 옵티마이저 상태
2. **Samples** (`samples_*.png`): 생성된 샘플 이미지
3. **Training logs** (`training_log.json`): Epoch별 손실 및 메트릭
4. **Metrics** (`*_results.json`): FID, IS, 생성 시간 등

## 🔧 고급 옵션

### 커스텀 하이퍼파라미터

각 스크립트는 기본 하이퍼파라미터를 사용하지만, 스크립트를 수정하여 커스터마이징할 수 있습니다:

```python
# run_vae_experiments.py 예시
args_dict = {
    'batch_size': 256,      # 배치 크기 변경
    'lr': 5e-5,             # 학습률 변경
    'latent_dim': 32,       # Latent dimension 변경
    # ...
}
```

### GPU 메모리 최적화

```bash
# 배치 크기 줄이기
python run_vae_experiments.py --batch-size 64

# CPU에서 실행 (느리지만 GPU 없이 가능)
python run_vae_experiments.py --device cpu
```

### 체크포인트 로딩

모든 스크립트는 중단된 학습을 재개할 수 있습니다:

```python
# 학습 스크립트 내에서
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

## 📈 벤치마크 재현

[BENCHMARK-RESULTS.md](../../docs/BENCHMARK-RESULTS.md)의 결과를 재현하려면:

### 1. VAE on MNIST
```bash
python run_vae_experiments.py \
  --experiments 1 \
  --dataset mnist \
  --epochs 50 \
  --device cuda
```

**예상 결과:**
- β=1.0: Train Loss ~110, Recon ~90, KL ~20
- β=4.0: Train Loss ~120, Recon ~85, KL ~35

### 2. DCGAN vs WGAN-GP on MNIST
```bash
python run_gan_experiments.py \
  --experiments 1 \
  --dataset mnist \
  --epochs 100 \
  --device cuda
```

**예상 결과:**
- DCGAN: FID 20-30, IS 3.0-3.5
- WGAN-GP: FID 15-25, IS 3.2-3.8

### 3. DDPM on MNIST
```bash
# 1. DDPM 학습
python run_diffusion_experiments.py \
  --experiments 1 \
  --dataset mnist \
  --epochs 50 \
  --device cuda

# 2. DDIM 샘플링 속도 테스트
python run_diffusion_experiments.py \
  --experiments 3 \
  --ddpm-checkpoint ./results/diffusion_experiments/ddpm_baseline/checkpoint_epoch_50.pt \
  --device cuda
```

**예상 결과:**
- DDPM: FID 10-20 (1000 steps)
- DDIM-50: FID 12-25 (20배 빠름)

## 🧪 Ablation Studies

### β-VAE β 값 영향
```bash
python run_vae_experiments.py --experiments 1
```

**분석:** β가 클수록:
- Reconstruction quality ↓ (높은 recon loss)
- Disentanglement ↑
- KL divergence ↑

### GAN Architecture 영향
```bash
python run_gan_experiments.py --experiments 3
```

**분석:** Feature map size가 클수록:
- 모델 크기 ↑
- 샘플 품질 ↑
- 학습 시간 ↑

### Diffusion Noise Schedule 영향
```bash
python run_diffusion_experiments.py --experiments 2
```

**분석:**
- **Cosine**: 가장 안정적, 추천
- **Linear**: 빠르지만 품질 낮음
- **Sigmoid**: Cosine과 유사

## 🔍 결과 분석

### Training Logs 분석

```python
import json
import matplotlib.pyplot as plt

# Load training log
with open('results/vae_experiments/beta_vae_beta_4.0/training_log.json', 'r') as f:
    log = json.load(f)

# Plot loss curves
epochs = [entry['epoch'] for entry in log]
losses = [entry['loss'] for entry in log]

plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Training Loss')
plt.savefig('loss_curve.png')
```

### FID/IS 비교

```python
import json

# Load comparison results
with open('results/model_comparison/comparison_results.json', 'r') as f:
    results = json.load(f)

# Print sorted by FID
sorted_models = sorted(results.items(), key=lambda x: x[1]['fid'])
for model, metrics in sorted_models:
    print(f"{model}: FID={metrics['fid']:.2f}, IS={metrics['is_mean']:.2f}")
```

## 💡 팁

1. **GPU 메모리 부족 시:**
   - 배치 크기 줄이기
   - 모델 크기 줄이기 (feature_map_size, model_channels)
   - Gradient checkpointing 사용

2. **학습이 느릴 때:**
   - Epoch 수 줄이기 (빠른 테스트용)
   - 더 작은 데이터셋 사용 (MNIST → Fashion-MNIST)
   - Mixed precision training 사용

3. **재현성을 위해:**
   - Random seed 고정
   - 동일한 하이퍼파라미터 사용
   - 동일한 데이터 전처리 사용

## 📚 참고 자료

- [BENCHMARK-RESULTS.md](../../docs/BENCHMARK-RESULTS.md): 상세한 벤치마크 결과
- [ADVANCEMENT-ROADMAP.md](../../docs/ADVANCEMENT-ROADMAP.md): 전체 로드맵
- 각 모델 디렉토리의 README: VAE, GAN, Diffusion

---

**작성일**: 2025-11-15
**버전**: 1.0
