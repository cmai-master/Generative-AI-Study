# Style Transfer 프로젝트

> CycleGAN을 활용한 이미지 스타일 변환
>
> **목표**: 페어 데이터 없이 두 도메인 간 스타일을 상호 변환

---

## 📖 프로젝트 개요

이 프로젝트는 CycleGAN을 사용하여 이미지의 스타일을 변환하는 시스템입니다. 페어 데이터 없이 학습 가능합니다.

**주요 기능:**
- ✅ Unpaired Image-to-Image Translation
- ✅ 양방향 변환 (A→B, B→A)
- ✅ Cycle Consistency Loss
- ✅ 실시간 스타일 변환 (추론)

**응용 분야:**
- 계절 변환 (여름 ↔ 겨울)
- 화풍 변환 (사진 ↔ 모네 화풍)
- 객체 변환 (말 ↔ 얼룩말)
- 낮/밤 변환

---

## 🏗️ CycleGAN 아키텍처

```
Domain A (e.g., 사진)  ←→  Domain B (e.g., 그림)

A → G_AB → B' (fake B)
B' → G_BA → A'' (reconstructed A)
Cycle Loss: ||A - A''||

B → G_BA → A' (fake A)
A' → G_AB → B'' (reconstructed B)
Cycle Loss: ||B - B''||

Discriminators:
- D_A: 진짜 A vs 가짜 A (G_BA의 출력)
- D_B: 진짜 B vs 가짜 B (G_AB의 출력)
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터셋 구조
data/
├── trainA/       # 도메인 A 이미지들
│   ├── img1.jpg
│   └── img2.jpg
├── trainB/       # 도메인 B 이미지들
│   ├── img1.jpg
│   └── img2.jpg
├── testA/
└── testB/

# 예시: 계절 변환
data/
├── trainA/  # 여름 이미지
└── trainB/  # 겨울 이미지
```

### 3. 학습

```bash
# 기본 학습
python train.py --dataroot ./data \
                --name summer2winter \
                --model cycle_gan

# 계속 학습
python train.py --dataroot ./data \
                --name summer2winter \
                --continue_train \
                --epoch_count 101
```

### 4. 테스트/변환

```bash
# A→B 변환
python test.py --dataroot ./data \
               --name summer2winter \
               --model cycle_gan \
               --direction AtoB

# B→A 변환
python test.py --dataroot ./data \
               --name summer2winter \
               --model cycle_gan \
               --direction BtoA

# 단일 이미지 변환
python convert.py --image input.jpg \
                  --checkpoint checkpoints/summer2winter/latest.pth \
                  --output output.jpg
```

### 5. Web UI

```bash
# Gradio 앱 실행
python app.py --checkpoint checkpoints/summer2winter/latest.pth

# 브라우저에서 http://localhost:7860 접속
```

---

## 📁 프로젝트 구조

```
style_transfer/
├── data/
│   ├── trainA/
│   ├── trainB/
│   ├── testA/
│   └── testB/
├── models/
│   ├── cycle_gan_model.py  # CycleGAN 모델
│   ├── networks.py         # Generator, Discriminator
│   └── losses.py           # GAN loss, Cycle loss, Identity loss
├── utils/
│   ├── image_pool.py       # History buffer for discriminator
│   └── visualizer.py       # 결과 시각화
├── checkpoints/            # 저장된 모델
├── results/                # 생성된 이미지
├── train.py               # 학습 스크립트
├── test.py                # 테스트 스크립트
├── convert.py             # 단일 이미지 변환
├── app.py                 # Gradio UI
└── README.md
```

---

## ⚙️ 모델 설정

### 하이퍼파라미터

```python
# Generator
- Architecture: ResNet (9 blocks)
- Input/Output: 3 channels (RGB)
- Feature maps: 64 → 128 → 256

# Discriminator
- Architecture: PatchGAN (70×70)
- Layers: 4 conv layers

# Training
- Epochs: 200
- Batch size: 1
- Learning rate: 0.0002
- Beta1: 0.5 (Adam)
- Lambda cycle: 10.0
- Lambda identity: 5.0

# Data augmentation
- Random crop: 256×256
- Random flip: horizontal
- Normalize: [-1, 1]
```

---

## 🎨 손실 함수

### 1. Adversarial Loss (GAN Loss)

$$
\mathcal{L}_{GAN}(G, D_B, A, B) = \mathbb{E}_{b \sim p(b)}[\log D_B(b)] + \mathbb{E}_{a \sim p(a)}[\log(1 - D_B(G(a)))]
$$

### 2. Cycle Consistency Loss

$$
\mathcal{L}_{cyc}(G_{AB}, G_{BA}) = \mathbb{E}_{a \sim p(a)}[\|G_{BA}(G_{AB}(a)) - a\|_1] + \mathbb{E}_{b \sim p(b)}[\|G_{AB}(G_{BA}(b)) - b\|_1]
$$

### 3. Identity Loss (선택적)

$$
\mathcal{L}_{identity}(G_{AB}, G_{BA}) = \mathbb{E}_{b \sim p(b)}[\|G_{AB}(b) - b\|_1] + \mathbb{E}_{a \sim p(a)}[\|G_{BA}(a) - a\|_1]
$$

### Total Loss

$$
\mathcal{L} = \mathcal{L}_{GAN}(G_{AB}, D_B) + \mathcal{L}_{GAN}(G_{BA}, D_A) + \lambda_{cyc}\mathcal{L}_{cyc} + \lambda_{id}\mathcal{L}_{identity}
$$

---

## 📊 실험 결과

### 학습 곡선

| Epoch | G Loss | D Loss | Cycle Loss |
|-------|--------|--------|------------|
| 10    | 2.34   | 1.12   | 15.6       |
| 50    | 1.56   | 0.98   | 8.3        |
| 100   | 1.12   | 0.87   | 4.2        |
| 200   | 0.89   | 0.76   | 2.1        |

### 변환 예시

```
Input (Summer) → Output (Winter)
🌞 ────────────→ ❄️

Before: 푸른 나무, 밝은 하늘
After:  눈 덮인 나무, 회색 하늘
```

---

## 🔧 고급 기법

### 1. Attention-based CycleGAN

```python
# Attention module 추가
class AttentionGenerator(nn.Module):
    def __init__(self):
        # Self-attention layer
        self.attention = SelfAttention(256)

    def forward(self, x):
        # ... ResNet blocks ...
        x = self.attention(x)
        # ... more blocks ...
```

### 2. Multi-domain Translation (StarGAN)

```python
# 여러 도메인 동시 학습
domains = ['summer', 'winter', 'spring', 'fall']

# Label-conditional generation
output = generator(input, target_domain='winter')
```

### 3. High-resolution Images

```python
# Progressive growing
# 64×64 → 128×128 → 256×256 → 512×512

# 또는 Patch-based training
train_on_patches(image, patch_size=256)
```

---

## 🎨 Gradio UI 기능

### 기본 사용

1. **이미지 업로드**: 변환할 이미지 선택
2. **방향 선택**: A→B 또는 B→A
3. **변환 버튼 클릭**
4. **결과 다운로드**

### 고급 옵션

- **Strength**: 변환 강도 조절 (0.0~1.0)
- **Preserve Details**: 세부사항 보존 정도
- **Color Transfer**: 색상만 변환 (구조 유지)

---

## 🐛 문제 해결

### 1. Mode Collapse

```yaml
# 해결 방법
- History buffer 사용 (image pool)
- Spectral normalization
- Learning rate 감소
- Lambda cycle 증가
```

### 2. 변환 품질 낮음

```yaml
# 개선 방법
- 더 많은 epoch 학습 (200+)
- ResNet blocks 증가 (9 → 12)
- Identity loss 추가
- 데이터 증강 강화
```

### 3. 학습 불안정

```yaml
# 안정화
- Learning rate: 0.0001
- Batch size: 4
- Gradient clip: 1.0
- TTUR (Two Time-scale Update Rule)
```

---

## 📚 참고 자료

### 논문
- [CycleGAN](https://arxiv.org/abs/1703.10593) - Unpaired Image-to-Image Translation
- [StarGAN](https://arxiv.org/abs/1711.09020) - Multi-Domain Image-to-Image Translation
- [AttentionGAN](https://arxiv.org/abs/1903.12296) - Attention-Guided Generative Adversarial Networks

### 코드베이스
- [Official CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [StarGAN v2](https://github.com/clovaai/stargan-v2)

---

## 🤝 기여

프로젝트 개선 환영!

---

## 📄 라이선스

MIT License

---

**Happy Transferring!** 🎨↔️🖼️
