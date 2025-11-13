# 개발 환경 설정 가이드

Generative AI 학습을 위한 완벽한 개발 환경 설정 가이드입니다.

---

## 📋 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [Python 환경 설정](#python-환경-설정)
3. [필수 라이브러리 설치](#필수-라이브러리-설치)
4. [GPU 설정](#gpu-설정)
5. [클라우드 환경](#클라우드-환경)
6. [개발 도구](#개발-도구)
7. [문제 해결](#문제-해결)

---

## 🖥 시스템 요구사항

### 최소 사양 (학습 가능)
- **CPU**: Intel i5 / AMD Ryzen 5 이상
- **RAM**: 16GB
- **GPU**: NVIDIA GPU 8GB VRAM (RTX 3060 Ti, RTX 3070)
- **Storage**: SSD 250GB
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS

### 권장 사양 (편안한 학습)
- **CPU**: Intel i7 / AMD Ryzen 7 이상
- **RAM**: 32GB
- **GPU**: NVIDIA GPU 12GB+ VRAM (RTX 3090, RTX 4070 Ti, RTX 4080)
- **Storage**: SSD 500GB+
- **OS**: Ubuntu 22.04 (리눅스 권장)

### 이상적 사양 (프로젝트 수행)
- **CPU**: Intel i9 / AMD Ryzen 9
- **RAM**: 64GB+
- **GPU**: NVIDIA GPU 24GB+ VRAM (RTX 3090 x2, RTX 4090, A5000, A6000)
- **Storage**: NVMe SSD 1TB+
- **OS**: Ubuntu 22.04

### GPU 없는 환경
- Google Colab (무료 GPU)
- Kaggle Notebooks (무료 GPU)
- Paperspace Gradient (무료/유료)
- 클라우드 서비스 (AWS, GCP, Azure)

---

## 🐍 Python 환경 설정

### 1. Anaconda/Miniconda 설치

#### Windows
```bash
# Miniconda 다운로드 및 설치
# https://docs.conda.io/en/latest/miniconda.html

# 설치 후 Anaconda Prompt 실행
```

#### Linux/macOS
```bash
# Miniconda 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 실행 권한 부여
chmod +x Miniconda3-latest-Linux-x86_64.sh

# 설치
./Miniconda3-latest-Linux-x86_64.sh

# 쉘 재시작
source ~/.bashrc  # or ~/.zshrc for macOS with zsh
```

### 2. 가상환경 생성

```bash
# Python 3.10 환경 생성 (권장)
conda create -n genai python=3.10

# 환경 활성화
conda activate genai

# 확인
python --version  # Python 3.10.x
```

### 3. 추가 환경 (선택사항)

```bash
# 특정 프로젝트용 환경
conda create -n stable-diffusion python=3.10
conda create -n llm python=3.10
conda create -n research python=3.10
```

---

## 📦 필수 라이브러리 설치

### 1. PyTorch 설치

#### NVIDIA GPU 있는 경우 (CUDA 11.8)
```bash
# CUDA 11.8 버전
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### NVIDIA GPU 있는 경우 (CUDA 12.1)
```bash
# CUDA 12.1 버전
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU만 있는 경우
```bash
pip install torch torchvision torchaudio
```

#### Apple Silicon (M1/M2/M3)
```bash
# MPS 가속 지원
pip install torch torchvision torchaudio
```

#### 설치 확인
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 2. 기본 라이브러리

```bash
# 수치 계산
pip install numpy scipy pandas

# 시각화
pip install matplotlib seaborn plotly

# 이미지 처리
pip install Pillow opencv-python imageio

# 진행 표시
pip install tqdm

# Jupyter
pip install jupyter notebook ipywidgets jupyterlab

# 과학 계산
pip install scikit-learn scikit-image
```

### 3. Generative AI 라이브러리

```bash
# Hugging Face 생태계
pip install transformers datasets accelerate
pip install diffusers

# 토크나이저
pip install tokenizers sentencepiece

# PEFT (Parameter-Efficient Fine-Tuning)
pip install peft

# bitsandbytes (양자화)
pip install bitsandbytes

# 실험 추적
pip install wandb tensorboard

# 유틸리티
pip install einops timm torchinfo torchsummary

# 이미지 생성 UI
pip install gradio streamlit
```

### 4. LLM 관련

```bash
# LangChain 생태계
pip install langchain langchain-community langchain-openai

# Vector Database
pip install chromadb faiss-cpu  # CPU 버전
# pip install faiss-gpu  # GPU 버전 (CUDA 필요)

# LlamaIndex
pip install llama-index

# OpenAI API
pip install openai

# Text Generation WebUI 관련
pip install bitsandbytes accelerate
```

### 5. 오디오/비디오

```bash
# 오디오
pip install librosa soundfile

# 비디오
pip install moviepy
```

### 6. 3D

```bash
# PyTorch3D (선택사항)
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
```

### 7. 추가 유틸리티

```bash
# HTTP 요청
pip install requests

# YAML 설정
pip install pyyaml

# 환경 변수
pip install python-dotenv

# 타입 체킹
pip install typing-extensions
```

---

## 🎮 GPU 설정

### NVIDIA GPU 드라이버 확인

#### Windows
```bash
# CMD 또는 PowerShell에서
nvidia-smi
```

#### Linux
```bash
nvidia-smi
```

출력 예시:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
```

### CUDA 설치 (Linux)

#### Ubuntu 22.04
```bash
# CUDA Toolkit 11.8 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 확인
nvcc --version
```

### cuDNN 설치 (선택사항)
```bash
# TensorFlow 사용 시 필요
# https://developer.nvidia.com/cudnn
```

### Mixed Precision Training 설정
```python
# PyTorch에서 자동 혼합 정밀도 사용
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## ☁️ 클라우드 환경

### Google Colab

#### 기본 설정
```python
# GPU 확인
import torch
print(torch.cuda.is_available())

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# GitHub 클론
!git clone https://github.com/your-repo.git
%cd your-repo
!pip install -r requirements.txt
```

#### Pro/Pro+ 활용
- Colab Pro: 더 강력한 GPU, 더 긴 실행 시간
- Background execution 활용

### Kaggle Notebooks

#### GPU 활성화
1. Settings → Accelerator → GPU T4 x2 선택
2. Internet On 설정

```python
# Kaggle 데이터셋 활용
!cp -r /kaggle/input/your-dataset ./data
```

### Paperspace Gradient

```bash
# CLI 설치
pip install gradient

# 로그인
gradient login

# 노트북 생성
gradient notebooks create \
  --machineType P5000 \
  --container pytorch/pytorch:latest
```

---

## 🛠 개발 도구

### 1. VS Code 설정

#### 확장 프로그램
- Python
- Pylance
- Jupyter
- GitLens
- autoDocstring
- Better Comments

#### settings.json
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.linting.pylintArgs": [
        "--max-line-length=120"
    ]
}
```

### 2. Git 설정

```bash
# Git 설치 확인
git --version

# 사용자 정보 설정
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 에디터 설정
git config --global core.editor "code --wait"

# 기본 브랜치 이름
git config --global init.defaultBranch main
```

### 3. 프로젝트 구조

```
project/
├── data/                 # 데이터셋
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/            # Jupyter 노트북
├── src/                  # 소스 코드
│   ├── models/
│   ├── data/
│   ├── training/
│   └── utils/
├── configs/              # 설정 파일
├── experiments/          # 실험 결과
├── checkpoints/          # 모델 체크포인트
├── outputs/              # 생성 결과물
├── tests/                # 테스트
├── requirements.txt      # 의존성
├── README.md
└── .gitignore
```

### 4. requirements.txt 생성

```bash
# 현재 환경의 패키지 저장
pip freeze > requirements.txt

# 또는 수동으로 작성
```

예시 requirements.txt:
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
diffusers>=0.18.0
datasets>=2.12.0
accelerate>=0.20.0
wandb>=0.15.0
jupyter>=1.0.0
matplotlib>=3.7.0
pillow>=9.5.0
```

### 5. .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Jupyter
.ipynb_checkpoints

# 데이터
data/raw/*
data/processed/*
*.csv
*.h5
*.pkl

# 모델
checkpoints/*
*.pth
*.pt
*.ckpt
*.safetensors

# 실험
wandb/
outputs/
experiments/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# 환경 변수
.env
```

---

## 🐛 문제 해결

### CUDA Out of Memory

```python
# 배치 크기 줄이기
batch_size = 8  # 16에서 8로

# Gradient accumulation 사용
accumulation_steps = 4

# Mixed precision training
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)

# 메모리 정리
import torch
torch.cuda.empty_cache()

# 메모리 사용량 확인
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")
```

### ImportError: No module named 'xxx'

```bash
# 가상환경 확인
conda activate genai

# 패키지 재설치
pip install xxx --upgrade

# 캐시 삭제 후 재설치
pip cache purge
pip install xxx --no-cache-dir
```

### RuntimeError: CUDA error

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 재설치 (CUDA 버전에 맞게)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Slow Training

```python
# DataLoader workers 증가
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # CPU 코어 수에 맞게
    pin_memory=True
)

# Compile (PyTorch 2.0+)
model = torch.compile(model)

# 프로파일링
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)
print(prof.key_averages().table())
```

### 디스크 공간 부족

```bash
# Conda 캐시 정리
conda clean --all

# Pip 캐시 정리
pip cache purge

# Hugging Face 캐시 정리
rm -rf ~/.cache/huggingface/

# Docker 정리 (사용 시)
docker system prune -a
```

---

## 📊 성능 모니터링

### GPU 모니터링

```bash
# nvidia-smi 실시간 모니터링
watch -n 1 nvidia-smi

# gpustat 설치 및 사용 (더 예쁨)
pip install gpustat
gpustat -i 1
```

### Python 코드에서 모니터링

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")

# 학습 중에 호출
print_gpu_memory()
```

### W&B로 실험 추적

```python
import wandb

wandb.login()

# 실험 시작
wandb.init(project="my-generative-model", name="experiment-1")

# 설정 로깅
wandb.config.update({
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
})

# 학습 중 로깅
for epoch in range(epochs):
    loss = train_one_epoch()
    wandb.log({"loss": loss, "epoch": epoch})

# 종료
wandb.finish()
```

---

## ✅ 설치 체크리스트

완료 후 체크:

- [ ] Python 3.10 설치 확인
- [ ] 가상환경 생성 및 활성화
- [ ] PyTorch GPU 버전 설치 (또는 CPU)
- [ ] CUDA 사용 가능 확인 (GPU 사용 시)
- [ ] 기본 라이브러리 설치
- [ ] Hugging Face 라이브러리 설치
- [ ] Jupyter 노트북 실행 가능
- [ ] Git 설정 완료
- [ ] VS Code (또는 선호하는 IDE) 설정
- [ ] 첫 번째 예제 코드 실행 성공

### 간단한 테스트 코드

```python
# test_setup.py
import torch
import torchvision
import transformers
import diffusers
import numpy as np
import matplotlib.pyplot as plt

print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Transformers version: {transformers.__version__}")
print(f"Diffusers version: {diffusers.__version__}")

# 간단한 텐서 연산
x = torch.randn(3, 3)
if torch.cuda.is_available():
    x = x.cuda()
    print(f"✅ Tensor on GPU: {x.device}")
print("✅ Setup complete!")
```

---

**설정 완료 후 README.md의 Phase 1부터 학습을 시작하세요!**

**마지막 업데이트**: 2025-01
