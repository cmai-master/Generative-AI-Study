"""
데이터셋 로더 유틸리티

주요 데이터셋을 쉽게 로드하고 전처리하는 함수들을 제공합니다.
- MNIST
- FashionMNIST
- CIFAR-10
- CelebA
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_mnist(data_dir='./data', batch_size=128, num_workers=4, download=True):
    """
    MNIST 데이터셋 로더

    Args:
        data_dir: 데이터 저장 경로
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        download: 데이터 다운로드 여부

    Returns:
        train_loader, test_loader: 학습/테스트 DataLoader
    """
    # 전처리: [0, 1] 범위로 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 학습 데이터
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    # 테스트 데이터
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_fashion_mnist(data_dir='./data', batch_size=128, num_workers=4, download=True):
    """
    Fashion-MNIST 데이터셋 로더

    Args:
        data_dir: 데이터 저장 경로
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        download: 데이터 다운로드 여부

    Returns:
        train_loader, test_loader: 학습/테스트 DataLoader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_cifar10(data_dir='./data', batch_size=128, num_workers=4, download=True):
    """
    CIFAR-10 데이터셋 로더

    Args:
        data_dir: 데이터 저장 경로
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        download: 데이터 다운로드 여부

    Returns:
        train_loader, test_loader: 학습/테스트 DataLoader
    """
    # CIFAR-10은 [-1, 1] 범위로 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_celeba(data_dir='./data/celeba', batch_size=128, image_size=64,
               num_workers=4, download=False):
    """
    CelebA 데이터셋 로더

    Args:
        data_dir: 데이터 저장 경로
        batch_size: 배치 크기
        image_size: 이미지 크기 (정사각형)
        num_workers: 데이터 로딩 워커 수
        download: 데이터 다운로드 여부 (매우 오래 걸림)

    Returns:
        train_loader, test_loader: 학습/테스트 DataLoader

    Note:
        CelebA는 약 200,000개의 얼굴 이미지를 포함하며,
        다운로드에 시간이 오래 걸릴 수 있습니다.
    """
    # CelebA용 전처리: Center Crop + Resize + Normalize
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # 얼굴 중심으로 크롭
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # 학습 데이터 (전체의 80%를 학습에 사용)
        train_dataset = datasets.CelebA(
            root=data_dir,
            split='train',
            download=download,
            transform=transform
        )

        # 검증 데이터
        test_dataset = datasets.CelebA(
            root=data_dir,
            split='valid',
            download=download,
            transform=transform
        )

    except RuntimeError as e:
        print(f"CelebA 데이터셋을 로드할 수 없습니다: {e}")
        print("CelebA는 수동으로 다운로드해야 할 수 있습니다.")
        print("https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        raise

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_dataloader(dataset_name, **kwargs):
    """
    데이터셋 이름으로 DataLoader 반환

    Args:
        dataset_name: 'mnist', 'fashion_mnist', 'cifar10', 'celeba' 중 하나
        **kwargs: 각 데이터셋 함수에 전달할 인자들

    Returns:
        train_loader, test_loader

    Example:
        >>> train_loader, test_loader = get_dataloader('mnist', batch_size=64)
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'mnist':
        return get_mnist(**kwargs)
    elif dataset_name == 'fashion_mnist' or dataset_name == 'fashionmnist':
        return get_fashion_mnist(**kwargs)
    elif dataset_name == 'cifar10':
        return get_cifar10(**kwargs)
    elif dataset_name == 'celeba':
        return get_celeba(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # 간단한 테스트
    print("Testing dataset loaders...")

    # MNIST 테스트
    print("\n1. Loading MNIST...")
    train_loader, test_loader = get_mnist(batch_size=32)
    images, labels = next(iter(train_loader))
    print(f"   MNIST batch shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Fashion-MNIST 테스트
    print("\n2. Loading Fashion-MNIST...")
    train_loader, test_loader = get_fashion_mnist(batch_size=32)
    images, labels = next(iter(train_loader))
    print(f"   Fashion-MNIST batch shape: {images.shape}")
    print(f"   Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # CIFAR-10 테스트
    print("\n3. Loading CIFAR-10...")
    train_loader, test_loader = get_cifar10(batch_size=32)
    images, labels = next(iter(train_loader))
    print(f"   CIFAR-10 batch shape: {images.shape}")
    print(f"   Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    print("\n✅ All dataset loaders work correctly!")
