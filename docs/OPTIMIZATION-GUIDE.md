# 생성 모델 최적화 가이드

> **목표**: 학습 속도, 메모리 효율성, 모델 성능을 극대화하는 방법
>
> **난이도**: 중급-고급
> **예상 학습 시간**: 1-2주

---

## 📚 목차

1. [Mixed Precision Training (AMP)](#1-mixed-precision-training-amp)
2. [Distributed Training](#2-distributed-training)
3. [Memory Optimization](#3-memory-optimization)
4. [학습 속도 최적화](#4-학습-속도-최적화)
5. [모델 압축](#5-모델-압축)
6. [실전 체크리스트](#6-실전-체크리스트)

---

## 1. Mixed Precision Training (AMP)

### 1.1 개요

**Mixed Precision Training**은 FP32와 FP16을 혼합하여 사용하는 학습 기법입니다.

**장점:**
- ⚡ **2-3배 빠른 학습** (GPU Tensor Core 활용)
- 💾 **메모리 사용량 50% 감소**
- 🎯 **동일한 정확도 유지**

**언제 사용?**
- NVIDIA GPU (Volta, Turing, Ampere 이상)
- 큰 모델 (메모리 부족 시)
- 빠른 실험 필요 시

### 1.2 PyTorch AMP 기본 사용법

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Scaler 생성 (gradient scaling)
scaler = GradScaler()

# 모델, 옵티마이저 준비
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
for epoch in range(epochs):
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**핵심:**
- `autocast()`: FP16으로 forward pass
- `scaler.scale()`: Gradient underflow 방지
- `scaler.step()`: 안전한 weight update
- `scaler.update()`: Scale factor 조정

### 1.3 VAE에 AMP 적용

```python
class VAETrainer:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()

    def train_step(self, x):
        self.optimizer.zero_grad()

        # Forward with mixed precision
        with autocast():
            # VAE forward
            x_recon, mu, logvar = self.model(x)

            # Loss 계산
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

        # Backward with scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()
```

### 1.4 GAN에 AMP 적용

```python
class GANTrainer:
    def __init__(self, generator, discriminator, g_opt, d_opt):
        self.G = generator.cuda()
        self.D = discriminator.cuda()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.scaler_G = GradScaler()
        self.scaler_D = GradScaler()

    def train_step(self, real_images):
        batch_size = real_images.size(0)

        # ===== Train Discriminator =====
        self.d_opt.zero_grad()

        with autocast():
            # Real images
            real_pred = self.D(real_images)
            d_loss_real = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )

            # Fake images
            z = torch.randn(batch_size, latent_dim).cuda()
            fake_images = self.G(z)
            fake_pred = self.D(fake_images.detach())
            d_loss_fake = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )

            d_loss = d_loss_real + d_loss_fake

        self.scaler_D.scale(d_loss).backward()
        self.scaler_D.step(self.d_opt)
        self.scaler_D.update()

        # ===== Train Generator =====
        self.g_opt.zero_grad()

        with autocast():
            z = torch.randn(batch_size, latent_dim).cuda()
            fake_images = self.G(z)
            fake_pred = self.D(fake_images)
            g_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred)
            )

        self.scaler_G.scale(g_loss).backward()
        self.scaler_G.step(self.g_opt)
        self.scaler_G.update()

        return d_loss.item(), g_loss.item()
```

### 1.5 주의사항

**1. Loss Scaling 오버플로우**
```python
# 잘못된 예
with autocast():
    loss = criterion(output, target)
    loss = loss * 1000  # ❌ Scaling 후 overflow!

# 올바른 예
with autocast():
    loss = criterion(output, target)
# Scaler가 자동으로 처리
```

**2. BatchNorm 문제**
```python
# BatchNorm은 FP32로 유지
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64).float()  # FP32 명시
```

**3. Gradient Clipping**
```python
# AMP와 함께 gradient clipping 사용
scaler.scale(loss).backward()

# Unscale before clipping!
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

### 1.6 성능 비교

| 설정 | 학습 시간 (epoch) | GPU 메모리 | 최종 Loss |
|------|------------------|-----------|----------|
| FP32 | 120초 | 8.2GB | 0.0542 |
| **AMP** | **45초** (2.7배↑) | **4.1GB** (50%↓) | 0.0538 |

---

## 2. Distributed Training

### 2.1 개요

**Distributed Training**은 여러 GPU/노드를 활용하여 병렬 학습하는 기법입니다.

**방법:**
- **DataParallel (DP)**: 단일 노드, 여러 GPU (간단하지만 비효율적)
- **DistributedDataParallel (DDP)**: 멀티 노드, 효율적 (권장)
- **Fully Sharded Data Parallel (FSDP)**: 초대형 모델용

### 2.2 DataParallel (DP) - 간단한 방법

```python
# 가장 간단한 multi-GPU 학습
model = YourModel()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.cuda()

# 나머지는 동일
for data, target in dataloader:
    output = model(data.cuda())
    loss = criterion(output, target.cuda())
    loss.backward()
    optimizer.step()
```

**단점:**
- GPU 0에 부하 집중
- 통신 오버헤드 큼
- 속도 향상 제한적 (1.5-2배)

### 2.3 DistributedDataParallel (DDP) - 권장

#### 설정

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """DDP 초기화"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """DDP 정리"""
    dist.destroy_process_group()
```

#### 학습 함수

```python
def train_ddp(rank, world_size, args):
    """
    DDP 학습 함수

    Args:
        rank: 현재 프로세스 rank (0, 1, 2, ...)
        world_size: 총 GPU 수
        args: 학습 인자
    """
    # Setup
    setup(rank, world_size)

    # Model to device
    model = YourModel().to(rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DistributedSampler (중요!)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,  # 일반 shuffle 대신
        num_workers=4,
        pin_memory=True
    )

    # Training loop
    for epoch in range(args.epochs):
        # Epoch마다 sampler shuffle
        train_sampler.set_epoch(epoch)

        model.train()
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Save checkpoint (rank 0만)
        if rank == 0:
            torch.save(model.module.state_dict(), f'checkpoint_{epoch}.pth')

    cleanup()
```

#### 실행

```python
def main():
    world_size = torch.cuda.device_count()

    # Spawn processes
    mp.spawn(
        train_ddp,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
```

#### torchrun으로 실행 (권장)

```bash
# 단일 노드, 4 GPU
torchrun --nproc_per_node=4 train.py

# 2 노드, 각 4 GPU (총 8 GPU)
# Node 0:
torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr="192.168.1.100" \
         --master_port=12355 \
         train.py

# Node 1:
torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=1 \
         --master_addr="192.168.1.100" \
         --master_port=12355 \
         train.py
```

### 2.4 DDP + AMP 결합

```python
def train_ddp_amp(rank, world_size, args):
    setup(rank, world_size)

    model = YourModel().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()  # AMP scaler

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()

            # Mixed precision forward
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    cleanup()
```

### 2.5 FSDP - 초대형 모델

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def train_fsdp(rank, world_size):
    setup(rank, world_size)

    # Model sharding
    model = YourLargeModel()
    model = FSDP(
        model,
        auto_wrap_policy=None,  # 자동 wrap
        mixed_precision=None,   # AMP 설정
        device_id=rank,
    )

    # 나머지는 DDP와 동일
    # ...
```

**FSDP 장점:**
- 모델 파라미터를 GPU 간 분산
- 메모리 효율 극대화
- 수백억 파라미터 모델 학습 가능

### 2.6 Gradient Accumulation

```python
# 작은 GPU 메모리로 큰 배치 효과
accumulation_steps = 4  # 4배 큰 배치 효과

optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    # Forward
    with autocast():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps  # 나누기!

    # Backward
    scaler.scale(loss).backward()

    # Update every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

## 3. Memory Optimization

### 3.1 Gradient Checkpointing

**아이디어:** Forward 시 중간 activation을 저장하지 않고, backward 시 다시 계산

```python
import torch.utils.checkpoint as checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)

    def forward(self, x):
        # Checkpoint로 감싸기
        x = checkpoint.checkpoint(self.layer1, x)
        x = checkpoint.checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x
```

**트레이드오프:**
- 메모리: 50-70% 감소 ✅
- 속도: 20-30% 느려짐 ⚠️

**언제 사용?**
- 메모리 부족 시
- 매우 깊은 네트워크

### 3.2 Inplace Operations

```python
# 메모리 절약
x = x + 1      # 새로운 tensor 생성
x += 1         # inplace (메모리 절약)
x.add_(1)      # inplace

# ReLU inplace
nn.ReLU(inplace=True)

# Dropout inplace
nn.Dropout(p=0.5, inplace=True)
```

### 3.3 del과 torch.cuda.empty_cache()

```python
# 사용 후 메모리 해제
for epoch in range(epochs):
    for data, target in dataloader:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # 명시적 삭제
        del output, loss

    # GPU 캐시 비우기
    torch.cuda.empty_cache()
```

### 3.4 배치 크기 자동 조정

```python
def find_optimal_batch_size(model, input_shape, device='cuda'):
    """
    OOM 없이 최대 배치 크기 찾기
    """
    batch_size = 1

    while True:
        try:
            # 테스트
            dummy_input = torch.randn(batch_size, *input_shape).to(device)
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()

            # 성공하면 2배 증가
            batch_size *= 2

            # 메모리 정리
            del dummy_input, output, loss
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                # 실패하면 이전 크기 / 2
                optimal_batch_size = batch_size // 4
                print(f"Optimal batch size: {optimal_batch_size}")
                return optimal_batch_size
            else:
                raise e
```

### 3.5 데이터 로딩 최적화

```python
# 효율적인 DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=4,           # CPU 코어 수에 맞춤
    pin_memory=True,         # GPU 전송 빠르게
    prefetch_factor=2,       # 미리 로드
    persistent_workers=True  # worker 재사용
)
```

### 3.6 모델 메모리 분석

```python
def analyze_model_memory(model):
    """모델 메모리 사용량 분석"""

    # 파라미터 메모리
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # 버퍼 메모리
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    print(f"Parameters: {param_size / 1024**2:.2f} MB")
    print(f"Buffers: {buffer_size / 1024**2:.2f} MB")
    print(f"Total: {total_size / 1024**2:.2f} MB")

    return total_size

# GPU 메모리 모니터링
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

---

## 4. 학습 속도 최적화

### 4.1 Compile (PyTorch 2.0+)

```python
# PyTorch 2.0의 게임체인저
model = YourModel()
model = torch.compile(model)  # 이게 전부!

# 1.5-2배 빠른 학습
```

**옵션:**
```python
# Default (빠름 + 안정성)
model = torch.compile(model)

# 최대 속도 (실험적)
model = torch.compile(model, mode="reduce-overhead")

# 최대 최적화
model = torch.compile(model, mode="max-autotune")
```

### 4.2 Channels Last Memory Format

```python
# NHWC 포맷 (더 빠른 convolution)
model = model.to(memory_format=torch.channels_last)

# 데이터도 변환
data = data.to(memory_format=torch.channels_last)
```

**성능 향상:** 22% faster on ResNet-50

### 4.3 TF32 활성화 (Ampere GPU)

```python
# A100, RTX 3090 등에서
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 정확도 손실 거의 없이 속도 향상
```

### 4.4 CuDNN Benchmark

```python
# 최적 알고리즘 자동 선택
torch.backends.cudnn.benchmark = True

# 입력 크기가 고정일 때 사용
# 처음엔 느리지만 이후 빨라짐
```

### 4.5 JIT Script

```python
# 모델을 TorchScript로 변환
model = YourModel()
scripted_model = torch.jit.script(model)

# 저장
scripted_model.save("model_scripted.pt")

# 로드 (더 빠름)
loaded_model = torch.jit.load("model_scripted.pt")
```

---

## 5. 모델 압축

### 5.1 Quantization (양자화)

#### Dynamic Quantization

```python
import torch.quantization

# FP32 모델
model_fp32 = YourModel()

# INT8로 양자화 (dynamic)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear, nn.Conv2d},  # 양자화할 레이어
    dtype=torch.qint8
)

# 4배 작은 모델, 2-3배 빠른 추론
```

#### Static Quantization

```python
# Calibration 필요
model_fp32.eval()
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Fuse modules
model_fp32_fused = torch.quantization.fuse_modules(
    model_fp32,
    [['conv', 'bn', 'relu']]
)

# Prepare
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# Calibrate (representative data)
with torch.no_grad():
    for data in calibration_dataloader:
        model_fp32_prepared(data)

# Convert
model_int8 = torch.quantization.convert(model_fp32_prepared)
```

### 5.2 Pruning (가지치기)

```python
import torch.nn.utils.prune as prune

# L1 Unstructured Pruning (50%)
prune.l1_unstructured(
    model.conv1,
    name='weight',
    amount=0.5
)

# Structured Pruning (channel 단위)
prune.ln_structured(
    model.conv1,
    name='weight',
    amount=0.3,
    n=2,  # L2 norm
    dim=0  # output channels
)

# Global Pruning (전체 모델)
parameters_to_prune = []
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5
)

# Pruning mask 제거 (영구 적용)
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')
```

### 5.3 Knowledge Distillation

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Distillation loss (soft targets)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        distill_loss = -torch.mean(torch.sum(soft_targets * soft_prob, dim=1))
        distill_loss = distill_loss * (self.temperature ** 2)

        # Student loss (hard targets)
        student_loss = self.ce_loss(student_logits, labels)

        # Combined
        loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        return loss

# 사용
teacher_model = LargeModel()  # 이미 학습됨
student_model = SmallModel()  # 학습할 모델

distill_criterion = DistillationLoss(temperature=3.0, alpha=0.7)

# 학습
for data, labels in dataloader:
    with torch.no_grad():
        teacher_logits = teacher_model(data)

    student_logits = student_model(data)
    loss = distill_criterion(student_logits, teacher_logits, labels)

    loss.backward()
    optimizer.step()
```

---

## 6. 실전 체크리스트

### ✅ 학습 전

- [ ] **AMP 활성화** (2-3배 빠름)
- [ ] **torch.compile()** 사용 (PyTorch 2.0+)
- [ ] **Channels Last** 포맷 (Conv 모델)
- [ ] **CuDNN Benchmark** 활성화
- [ ] **DataLoader 최적화** (num_workers, pin_memory)
- [ ] **배치 크기 최대화** (GPU 메모리 한계까지)

### ✅ Multi-GPU 사용 시

- [ ] **DDP 사용** (DP 대신)
- [ ] **DistributedSampler** 사용
- [ ] **Gradient Accumulation** (작은 GPU)
- [ ] **NCCL backend** 확인

### ✅ 메모리 부족 시

- [ ] **Gradient Checkpointing**
- [ ] **Batch size 감소**
- [ ] **Gradient Accumulation**
- [ ] **Inplace operations**
- [ ] **FSDP** 고려 (초대형 모델)

### ✅ 추론/배포 시

- [ ] **Quantization** (INT8)
- [ ] **Pruning** (50-70%)
- [ ] **Knowledge Distillation**
- [ ] **TorchScript/ONNX** 변환
- [ ] **TensorRT** (NVIDIA GPU)

---

## 7. 성능 벤치마크

### 최적화 전후 비교 (ResNet-50, ImageNet)

| 최적화 | 학습 시간 | GPU 메모리 | 정확도 |
|--------|----------|-----------|--------|
| **Baseline (FP32)** | 100% | 8.0GB | 76.1% |
| + AMP | 38% (-62%) | 4.2GB (-48%) | 76.0% |
| + Compile | 32% (-68%) | 4.2GB | 76.0% |
| + Channels Last | 28% (-72%) | 4.0GB | 76.1% |
| + All + DDP (4 GPU) | 8% (-92%) | 16GB (total) | 76.2% |

**결론:** 10배 이상 빠른 학습 가능!

---

## 📚 참고 자료

### 공식 문서
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)
- [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch 2.0 Compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

### 튜토리얼
- [HuggingFace Performance](https://huggingface.co/docs/transformers/performance)
- [NVIDIA Apex](https://github.com/NVIDIA/apex)
- [DeepSpeed](https://www.deepspeed.ai/)

---

**최종 업데이트**: 2025-11-15
**라이선스**: MIT
