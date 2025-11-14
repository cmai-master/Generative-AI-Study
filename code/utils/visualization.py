"""
시각화 유틸리티 함수

생성 모델의 결과를 시각화하는 함수들을 제공합니다.
- 이미지 그리드 시각화
- 학습 곡선 플롯
- Latent Space 시각화
- 이미지 보간(Interpolation)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


def imshow(img, title=None, figsize=(10, 10), cmap=None):
    """
    PyTorch 텐서 이미지를 matplotlib으로 표시

    Args:
        img: 이미지 텐서 (C, H, W) 또는 (H, W)
        title: 제목
        figsize: 그림 크기
        cmap: 컬러맵 (grayscale의 경우 'gray')
    """
    # 텐서를 numpy로 변환
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    # 단일 채널 이미지인 경우
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)
        if cmap is None:
            cmap = 'gray'

    # 값 범위 조정 [0, 1]
    img = np.clip(img, 0, 1)

    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def show_image_grid(images, nrow=8, title=None, figsize=(12, 12), save_path=None):
    """
    여러 이미지를 그리드로 표시

    Args:
        images: 이미지 텐서 (B, C, H, W)
        nrow: 한 행에 표시할 이미지 수
        title: 제목
        figsize: 그림 크기
        save_path: 저장 경로 (None이면 표시만)
    """
    # 그리드 생성
    grid = make_grid(images, nrow=nrow, normalize=True, padding=2)

    # 표시
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to: {save_path}")
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses=None, title='Training Curves',
                          save_path=None, figsize=(10, 6)):
    """
    학습 곡선 플롯

    Args:
        train_losses: 학습 손실 리스트
        val_losses: 검증 손실 리스트
        title: 제목
        save_path: 저장 경로
        figsize: 그림 크기
    """
    plt.figure(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to: {save_path}")
    else:
        plt.show()


def plot_multiple_metrics(metrics_dict, title='Training Metrics',
                          save_path=None, figsize=(12, 8)):
    """
    여러 메트릭을 한 번에 플롯

    Args:
        metrics_dict: {'metric_name': [values]} 형태의 딕셔너리
        title: 제목
        save_path: 저장 경로
        figsize: 그림 크기

    Example:
        >>> metrics = {
        ...     'train_loss': [1.0, 0.8, 0.6],
        ...     'val_loss': [1.1, 0.85, 0.65],
        ...     'reconstruction': [0.5, 0.4, 0.3]
        ... }
        >>> plot_multiple_metrics(metrics)
    """
    n_metrics = len(metrics_dict)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)

    # 빈 subplot 제거
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to: {save_path}")
    else:
        plt.show()


def visualize_latent_space_2d(latent_codes, labels=None, title='Latent Space',
                                save_path=None, figsize=(10, 10)):
    """
    2D Latent Space 시각화

    Args:
        latent_codes: Latent 벡터 (N, 2)
        labels: 레이블 (N,) - 색상 구분용
        title: 제목
        save_path: 저장 경로
        figsize: 그림 크기
    """
    if isinstance(latent_codes, torch.Tensor):
        latent_codes = latent_codes.detach().cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    plt.figure(figsize=figsize)

    if labels is not None:
        scatter = plt.scatter(latent_codes[:, 0], latent_codes[:, 1],
                             c=labels, cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter)
    else:
        plt.scatter(latent_codes[:, 0], latent_codes[:, 1],
                   alpha=0.6, s=20)

    plt.xlabel('Latent Dimension 1', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to: {save_path}")
    else:
        plt.show()


def visualize_interpolation(model, z1, z2, n_steps=10, device='cpu',
                            title='Latent Space Interpolation',
                            save_path=None):
    """
    두 latent 벡터 사이의 보간 시각화

    Args:
        model: 디코더 모델
        z1: 시작 latent 벡터 (latent_dim,)
        z2: 끝 latent 벡터 (latent_dim,)
        n_steps: 보간 스텝 수
        device: 디바이스
        title: 제목
        save_path: 저장 경로
    """
    model.eval()

    # 보간 벡터 생성
    alphas = np.linspace(0, 1, n_steps)
    z_interp = torch.stack([
        (1 - alpha) * z1 + alpha * z2
        for alpha in alphas
    ]).to(device)

    # 디코딩
    with torch.no_grad():
        if hasattr(model, 'decode'):
            images = model.decode(z_interp)
        else:
            images = model.decoder(z_interp)

    # 시각화
    show_image_grid(images, nrow=n_steps, title=title, save_path=save_path)


def visualize_reconstruction(model, images, device='cpu', n_samples=8,
                             title='Reconstruction', save_path=None):
    """
    원본 이미지와 재구성 이미지 비교

    Args:
        model: VAE 모델
        images: 원본 이미지 (B, C, H, W)
        device: 디바이스
        n_samples: 표시할 샘플 수
        title: 제목
        save_path: 저장 경로
    """
    model.eval()

    images = images[:n_samples].to(device)

    # 재구성
    with torch.no_grad():
        if hasattr(model, 'reconstruct'):
            recon_images = model.reconstruct(images)
        else:
            recon_images, _, _ = model(images)

    # 원본과 재구성 이미지를 번갈아 배치
    comparison = torch.stack([images, recon_images], dim=1).flatten(0, 1)

    # 시각화
    show_image_grid(comparison, nrow=n_samples, title=title, save_path=save_path)


def visualize_samples(model, n_samples=64, latent_dim=20, device='cpu',
                     title='Generated Samples', save_path=None):
    """
    모델에서 랜덤 샘플 생성 및 시각화

    Args:
        model: 생성 모델
        n_samples: 생성할 샘플 수
        latent_dim: Latent 차원
        device: 디바이스
        title: 제목
        save_path: 저장 경로
    """
    model.eval()

    # 랜덤 latent 벡터
    z = torch.randn(n_samples, latent_dim).to(device)

    # 생성
    with torch.no_grad():
        if hasattr(model, 'decode'):
            samples = model.decode(z)
        elif hasattr(model, 'sample'):
            samples = model.sample(n_samples, device=device)
        else:
            samples = model.decoder(z)

    # 시각화
    show_image_grid(samples, nrow=8, title=title, save_path=save_path)


def compare_models(images_dict, title='Model Comparison', save_path=None,
                  figsize=(15, 5)):
    """
    여러 모델의 결과 비교

    Args:
        images_dict: {'model_name': images} 형태의 딕셔너리
        title: 제목
        save_path: 저장 경로
        figsize: 그림 크기

    Example:
        >>> images = {
        ...     'Original': original_images,
        ...     'VAE': vae_recon,
        ...     'β-VAE': beta_vae_recon
        ... }
        >>> compare_models(images)
    """
    n_models = len(images_dict)

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, images) in enumerate(images_dict.items()):
        grid = make_grid(images, nrow=8, normalize=True, padding=2)
        axes[idx].imshow(grid.permute(1, 2, 0).cpu().numpy())
        axes[idx].set_title(model_name, fontsize=12)
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # 간단한 테스트
    print("Testing visualization utilities...")

    # 더미 이미지 생성
    dummy_images = torch.rand(16, 1, 28, 28)

    print("\n1. Testing show_image_grid...")
    show_image_grid(dummy_images, nrow=4, title='Test Grid')

    print("\n2. Testing plot_training_curves...")
    train_losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.42]
    val_losses = [1.1, 0.85, 0.65, 0.55, 0.52, 0.50]
    plot_training_curves(train_losses, val_losses)

    print("\n3. Testing visualize_latent_space_2d...")
    latent_codes = torch.randn(1000, 2)
    labels = torch.randint(0, 10, (1000,))
    visualize_latent_space_2d(latent_codes, labels)

    print("\n✅ All visualization utilities work correctly!")
