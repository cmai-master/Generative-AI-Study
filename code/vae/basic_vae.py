"""
Basic VAE (Variational Autoencoder) 구현

Kingma & Welling (2013) "Auto-Encoding Variational Bayes"를 기반으로 한
기본 VAE 구현입니다.

주요 특징:
- Encoder: x → (μ, log_σ²)
- Reparameterization Trick: z = μ + σ * ε
- Decoder: z → x̂
- Loss: Reconstruction + KL Divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    VAE Encoder Network

    입력 이미지를 잠재 공간의 평균(μ)과 로그 분산(log σ²)으로 인코딩합니다.
    """

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        """
        Args:
            input_dim: 입력 차원 (예: 28*28=784 for MNIST)
            hidden_dims: 히든 레이어 차원 리스트 (예: [512, 256])
            latent_dim: 잠재 공간 차원
        """
        super().__init__()

        # 입력 레이어부터 히든 레이어까지
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # μ와 log σ² 출력 레이어
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: 입력 이미지 (batch_size, input_dim)

        Returns:
            mu: 평균 (batch_size, latent_dim)
            logvar: 로그 분산 (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    VAE Decoder Network

    잠재 벡터 z를 원본 이미지 공간으로 디코딩합니다.
    """

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        """
        Args:
            latent_dim: 잠재 공간 차원
            hidden_dims: 히든 레이어 차원 리스트 (예: [256, 512])
            output_dim: 출력 차원 (예: 28*28=784 for MNIST)
        """
        super().__init__()

        # 잠재 벡터부터 히든 레이어까지
        layers = []
        prev_dim = latent_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # [0, 1] 범위로 출력

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            z: 잠재 벡터 (batch_size, latent_dim)

        Returns:
            x_recon: 재구성된 이미지 (batch_size, output_dim)
        """
        return self.decoder(z)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE)

    완전한 VAE 모델로, Encoder, Decoder, Reparameterization Trick을 포함합니다.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = [512, 256],
        latent_dim: int = 20
    ):
        """
        Args:
            input_dim: 입력 차원 (기본값: 784 for MNIST)
            hidden_dims: 히든 레이어 차원 (기본값: [512, 256])
            latent_dim: 잠재 공간 차원 (기본값: 20)
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder와 Decoder 생성
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick

        z = μ + σ * ε, where ε ~ N(0, I)

        Args:
            mu: 평균 (batch_size, latent_dim)
            logvar: 로그 분산 (batch_size, latent_dim)

        Returns:
            z: 샘플링된 잠재 벡터 (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log σ²)
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        z = mu + eps * std              # z = μ + σ * ε
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: 입력 이미지 (batch_size, C, H, W) 또는 (batch_size, input_dim)

        Returns:
            x_recon: 재구성된 이미지 (batch_size, input_dim)
            mu: 평균 (batch_size, latent_dim)
            logvar: 로그 분산 (batch_size, latent_dim)
        """
        # Flatten
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Encode
        mu, logvar = self.encoder(x_flat)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        잠재 공간에서 랜덤 샘플링하여 이미지 생성

        Args:
            num_samples: 생성할 샘플 수
            device: 디바이스

        Returns:
            samples: 생성된 이미지 (num_samples, input_dim)
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 이미지 재구성 (결정적, 평균만 사용)

        Args:
            x: 입력 이미지

        Returns:
            재구성된 이미지
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        mu, _ = self.encoder(x_flat)
        x_recon = self.decoder(mu)

        return x_recon


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE 손실 함수

    Loss = Reconstruction Loss + β * KL Divergence

    Args:
        x: 원본 이미지 (batch_size, input_dim)
        x_recon: 재구성된 이미지 (batch_size, input_dim)
        mu: 평균 (batch_size, latent_dim)
        logvar: 로그 분산 (batch_size, latent_dim)
        beta: KL Divergence 가중치 (β-VAE)

    Returns:
        total_loss: 전체 손실
        recon_loss: 재구성 손실
        kl_loss: KL Divergence 손실
    """
    # Flatten
    batch_size = x.size(0)
    x_flat = x.view(batch_size, -1)

    # Reconstruction Loss (Binary Cross-Entropy)
    # BCE = -Σ [x * log(x_recon) + (1-x) * log(1-x_recon)]
    recon_loss = F.binary_cross_entropy(
        x_recon, x_flat, reduction='sum'
    ) / batch_size

    # KL Divergence Loss
    # KL(q(z|x) || p(z)) = -0.5 * Σ [1 + log(σ²) - μ² - σ²]
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / batch_size

    # Total Loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_epoch(
    model: VAE,
    train_loader,
    optimizer,
    device: str,
    beta: float = 1.0
) -> Tuple[float, float, float]:
    """
    한 에포크 학습

    Args:
        model: VAE 모델
        train_loader: 학습 데이터 로더
        optimizer: 옵티마이저
        device: 디바이스
        beta: β-VAE 가중치

    Returns:
        avg_loss: 평균 전체 손실
        avg_recon: 평균 재구성 손실
        avg_kl: 평균 KL Divergence
    """
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        x_recon, mu, logvar = model(data)

        # Loss 계산
        loss, recon, kl = vae_loss(data, x_recon, mu, logvar, beta)

        # Backward pass
        loss.backward()
        optimizer.step()

        # 누적
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    # 평균 계산
    n_batches = len(train_loader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


@torch.no_grad()
def evaluate(
    model: VAE,
    test_loader,
    device: str,
    beta: float = 1.0
) -> Tuple[float, float, float]:
    """
    모델 평가

    Args:
        model: VAE 모델
        test_loader: 테스트 데이터 로더
        device: 디바이스
        beta: β-VAE 가중치

    Returns:
        avg_loss: 평균 전체 손실
        avg_recon: 평균 재구성 손실
        avg_kl: 평균 KL Divergence
    """
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for data, _ in test_loader:
        data = data.to(device)

        # Forward pass
        x_recon, mu, logvar = model(data)

        # Loss 계산
        loss, recon, kl = vae_loss(data, x_recon, mu, logvar, beta)

        # 누적
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    # 평균 계산
    n_batches = len(test_loader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


if __name__ == "__main__":
    # 간단한 테스트
    print("Testing Basic VAE...")

    # 모델 생성
    model = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=20)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 더미 데이터
    batch_size = 32
    dummy_data = torch.rand(batch_size, 1, 28, 28)

    # Forward pass
    x_recon, mu, logvar = model(dummy_data)
    print(f"\nInput shape: {dummy_data.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Latent μ shape: {mu.shape}")
    print(f"Latent log σ² shape: {logvar.shape}")

    # Loss 계산
    loss, recon, kl = vae_loss(dummy_data, x_recon, mu, logvar)
    print(f"\nTotal Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {recon.item():.4f}")
    print(f"KL Divergence: {kl.item():.4f}")

    # 샘플링
    samples = model.sample(num_samples=16)
    print(f"\nGenerated samples shape: {samples.shape}")

    print("\n✅ Basic VAE works correctly!")
