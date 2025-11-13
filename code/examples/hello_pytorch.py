"""
Hello PyTorch - 환경 설정 확인용 예제

이 파일은 PyTorch 환경이 올바르게 설정되었는지 확인하기 위한 간단한 예제입니다.
"""

import torch
import torch.nn as nn
import numpy as np


def check_installation():
    """설치된 라이브러리 버전 확인"""
    print("=" * 50)
    print("PyTorch 환경 확인")
    print("=" * 50)

    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")

    # CUDA 확인
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

    # MPS (Apple Silicon) 확인
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")

    print("=" * 50)


def simple_tensor_operations():
    """간단한 텐서 연산 테스트"""
    print("\n텐서 연산 테스트")
    print("-" * 50)

    # 텐서 생성
    x = torch.randn(3, 3)
    print(f"Random tensor:\n{x}\n")

    # GPU로 이동 (가능한 경우)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    print(f"Tensor device: {x.device}")

    # 연산
    y = x + 2
    z = torch.matmul(x, y.T)
    print(f"Matrix multiplication result:\n{z}\n")

    print("-" * 50)


def simple_neural_network():
    """간단한 신경망 테스트"""
    print("\n신경망 테스트")
    print("-" * 50)

    # 간단한 모델 정의
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 모델 생성
    model = SimpleNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Model:\n{model}\n")

    # Forward pass
    x = torch.randn(32, 10).to(device)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (first 3 samples):\n{output[:3]}\n")

    # Parameter 확인
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    print("-" * 50)


def autograd_example():
    """Autograd 테스트"""
    print("\nAutograd 테스트")
    print("-" * 50)

    # requires_grad=True로 그래디언트 추적
    x = torch.tensor([2.0], requires_grad=True)

    # y = x^2 + 2x + 1
    y = x**2 + 2*x + 1

    # 역전파
    y.backward()

    print(f"x = {x.item()}")
    print(f"y = x^2 + 2x + 1 = {y.item()}")
    print(f"dy/dx = 2x + 2 = {x.grad.item()}")
    print(f"Expected: {2*x.item() + 2}")

    print("-" * 50)


if __name__ == "__main__":
    # 환경 확인
    check_installation()

    # 텐서 연산
    simple_tensor_operations()

    # 신경망
    simple_neural_network()

    # Autograd
    autograd_example()

    print("\n✅ 모든 테스트 완료! PyTorch 환경이 정상적으로 설정되었습니다.")
