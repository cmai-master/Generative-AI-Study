# Data Directory

이 디렉토리는 데이터셋을 저장하는 공간입니다.

## 구조

```
data/
├── raw/              # 원본 데이터
├── processed/        # 전처리된 데이터
└── external/         # 외부 데이터 (다운로드된 데이터셋 등)
```

## 주의사항

- 대용량 데이터는 `.gitignore`에 포함되어 Git에 커밋되지 않습니다.
- 데이터셋은 직접 다운로드하거나 스크립트를 통해 가져와야 합니다.
- 저작권이 있는 데이터는 공개 저장소에 업로드하지 마세요.

## 데이터셋 다운로드 예시

### MNIST
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data/raw',
    train=True,
    download=True,
    transform=transform
)
```

### CelebA
```python
from torchvision import datasets

dataset = datasets.CelebA(
    root='./data/raw',
    split='train',
    download=True
)
```

### Hugging Face Datasets
```python
from datasets import load_dataset

dataset = load_dataset("cifar10")
dataset.save_to_disk("./data/raw/cifar10")
```
