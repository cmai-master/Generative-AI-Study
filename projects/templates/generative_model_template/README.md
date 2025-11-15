# Generative Model Project Template

> 생성 모델 프로젝트를 위한 표준 템플릿
>
> 이 템플릿을 사용하여 빠르게 프로젝트를 시작하세요!

---

## 📁 프로젝트 구조

```
generative_model_template/
├── configs/                 # 설정 파일
│   ├── base_config.yaml    # 기본 설정
│   └── experiment_*.yaml   # 실험별 설정
├── data/                   # 데이터셋 및 전처리
│   ├── __init__.py
│   ├── dataset.py         # Dataset 클래스
│   └── transforms.py      # 데이터 변환
├── models/                # 모델 정의
│   ├── __init__.py
│   ├── generator.py       # Generator/생성 모델
│   ├── discriminator.py   # Discriminator (GAN용)
│   └── losses.py          # 손실 함수
├── utils/                 # 유틸리티
│   ├── __init__.py
│   ├── logger.py          # 로깅
│   ├── checkpoint.py      # 체크포인트 관리
│   ├── metrics.py         # 평가 지표
│   └── visualization.py   # 시각화
├── scripts/               # 실행 스크립트
│   ├── train.sh          # 학습 스크립트
│   └── evaluate.sh       # 평가 스크립트
├── train.py              # 학습 메인 스크립트
├── evaluate.py           # 평가 메인 스크립트
├── generate.py           # 샘플 생성 스크립트
├── requirements.txt      # 의존성
└── README.md            # 이 파일
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
conda create -n myproject python=3.10
conda activate myproject

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 다운로드 (예시)
python -m data.download --dataset mnist

# 또는 직접 data/ 디렉토리에 배치
```

### 3. 학습

```bash
# 기본 설정으로 학습
python train.py --config configs/base_config.yaml

# 실험 설정으로 학습
python train.py --config configs/experiment_1.yaml

# 커스텀 설정
python train.py --config configs/base_config.yaml \
                --batch-size 128 \
                --epochs 100 \
                --lr 0.0001
```

### 4. 평가

```bash
# 학습된 모델 평가
python evaluate.py --checkpoint checkpoints/best_model.pth

# 샘플 생성
python generate.py --checkpoint checkpoints/best_model.pth \
                   --num-samples 100 \
                   --output-dir samples/
```

---

## ⚙️ 설정 파일

### YAML 설정 예시 (configs/base_config.yaml)

```yaml
# 모델 설정
model:
  name: "MyGenerativeModel"
  type: "vae"  # vae, gan, diffusion, flow
  latent_dim: 128
  hidden_dims: [512, 256, 128]

# 데이터 설정
data:
  dataset: "mnist"
  data_dir: "./data"
  batch_size: 128
  num_workers: 4
  image_size: 64

# 학습 설정
training:
  epochs: 100
  lr: 0.0001
  optimizer: "adam"
  scheduler: "cosine"
  gradient_clip: 1.0

# 로깅 설정
logging:
  log_dir: "./logs"
  tensorboard: true
  wandb: false
  save_interval: 10
  eval_interval: 5

# 체크포인트 설정
checkpoint:
  save_dir: "./checkpoints"
  save_best: true
  save_last: true
```

### 파이썬에서 설정 로드

```python
import yaml
from types import SimpleNamespace

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v
                               for k, v in config.items()})

# 사용
config = load_config('configs/base_config.yaml')
print(config.model.latent_dim)  # 128
```

---

## 📊 로깅 및 모니터링

### TensorBoard

```bash
# TensorBoard 실행
tensorboard --logdir logs/

# 브라우저에서 http://localhost:6006 접속
```

### Weights & Biases (선택)

```python
import wandb

# 초기화
wandb.init(project="my-generative-model", config=config)

# 로깅
wandb.log({"train_loss": loss.item(), "epoch": epoch})

# 이미지 로깅
wandb.log({"samples": [wandb.Image(img) for img in samples]})
```

---

## 🔧 커스터마이징 가이드

### 1. 새로운 모델 추가

**models/my_model.py**:
```python
import torch.nn as nn

class MyGenerativeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 모델 정의

    def forward(self, x):
        # Forward pass
        return output

    def loss(self, x):
        # 손실 함수 계산
        return loss
```

**models/__init__.py**에 등록:
```python
from .my_model import MyGenerativeModel

MODEL_REGISTRY = {
    'vae': VAE,
    'gan': GAN,
    'my_model': MyGenerativeModel,  # 추가
}
```

### 2. 새로운 데이터셋 추가

**data/dataset.py**:
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 데이터 로드

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
```

### 3. 새로운 평가 지표 추가

**utils/metrics.py**:
```python
def my_metric(generated, real):
    """
    커스텀 평가 지표

    Args:
        generated: 생성된 샘플
        real: 실제 데이터

    Returns:
        metric_value: 지표 값
    """
    # 계산
    return metric_value
```

---

## 📦 체크포인트 관리

### 저장

```python
from utils.checkpoint import CheckpointManager

checkpoint_manager = CheckpointManager(save_dir='./checkpoints')

# 체크포인트 저장
checkpoint_manager.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    loss=loss,
    metrics={'fid': fid_score},
    filename='best_model.pth'
)
```

### 로드

```python
# 체크포인트 로드
checkpoint = checkpoint_manager.load('best_model.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 🎨 샘플 생성

### 기본 생성

```python
import torch
from models import load_model

# 모델 로드
model = load_model('checkpoints/best_model.pth')
model.eval()

# 샘플 생성
with torch.no_grad():
    samples = model.sample(num_samples=64)

# 저장
from torchvision.utils import save_image
save_image(samples, 'generated_samples.png', nrow=8, normalize=True)
```

### Interpolation

```python
# Latent space interpolation
z1 = torch.randn(1, latent_dim)
z2 = torch.randn(1, latent_dim)

for alpha in np.linspace(0, 1, 10):
    z = (1 - alpha) * z1 + alpha * z2
    sample = model.decode(z)
    save_image(sample, f'interp_{alpha:.1f}.png')
```

---

## 🐛 디버깅 팁

### 1. Overfitting 테스트

```bash
# 작은 데이터셋으로 빠르게 overfitting 되는지 확인
python train.py --config configs/debug.yaml \
                --dataset-size 100 \
                --epochs 10
```

### 2. Gradient 확인

```python
# Gradient 모니터링
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### 3. 메모리 사용량 확인

```python
import torch

print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## 📚 추가 리소스

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [TensorBoard 가이드](https://www.tensorflow.org/tensorboard)
- [Weights & Biases 튜토리얼](https://docs.wandb.ai/)
- [Hydra Config](https://hydra.cc/) - 더 강력한 설정 관리

---

## 🤝 기여

이 템플릿을 개선하고 싶으시다면:
1. Fork
2. 개선 사항 구현
3. Pull Request

---

## 📄 라이선스

MIT License

---

**Happy Coding!** 🚀
