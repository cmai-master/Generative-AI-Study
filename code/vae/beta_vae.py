"""
β-VAE (Beta Variational Autoencoder) 구현

Higgins et al. (2017) "β-VAE: Learning Basic Visual Concepts with a
Constrained Variational Framework"를 기반으로 한 β-VAE 구현입니다.

주요 특징:
- Loss = Reconstruction + β * KL Divergence
- β > 1: Disentangled Representation 학습 강화
- β = 1: 일반 VAE와 동일
"""

import torch
import torch.nn as nn
from typing import Tuple
from basic_vae import Encoder, Decoder, vae_loss as base_vae_loss


class BetaVAE(nn.Module):
    """
    β-VAE (Beta Variational Autoencoder)

    Basic VAE와 동일한 구조이지만, KL Divergence에 β 가중치를 적용하여
    disentangled representation 학습을 강화합니다.

    β 값에 따른 효과:
    - β = 1.0: 일반 VAE
    - β < 1.0: 재구성 품질 중시 (더 선명한 이미지)
    - β > 1.0: Disentanglement 중시 (더 독립적인 latent factors)
    - 일반적인 β 범위: [0.5, 10.0]
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = [512, 256],
        latent_dim: int = 20,
        beta: float = 4.0
    ):
        """
        Args:
            input_dim: 입력 차원 (기본값: 784 for MNIST)
            hidden_dims: 히든 레이어 차원 (기본값: [512, 256])
            latent_dim: 잠재 공간 차원 (기본값: 20)
            beta: KL Divergence 가중치 (기본값: 4.0)
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder와 Decoder (Basic VAE와 동일)
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        입력을 잠재 공간으로 인코딩

        Args:
            x: 입력 이미지

        Returns:
            mu: 평균
            logvar: 로그 분산
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.encoder(x_flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        잠재 벡터를 이미지로 디코딩

        Args:
            z: 잠재 벡터

        Returns:
            재구성된 이미지
        """
        return self.decoder(z)


def beta_vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-VAE 손실 함수

    Loss = Reconstruction Loss + β * KL Divergence

    Args:
        x: 원본 이미지
        x_recon: 재구성된 이미지
        mu: 평균
        logvar: 로그 분산
        beta: KL Divergence 가중치

    Returns:
        total_loss: 전체 손실
        recon_loss: 재구성 손실
        kl_loss: KL Divergence 손실
    """
    return base_vae_loss(x, x_recon, mu, logvar, beta)


def train_epoch_beta_vae(
    model: BetaVAE,
    train_loader,
    optimizer,
    device: str
) -> Tuple[float, float, float]:
    """
    한 에포크 학습 (β-VAE)

    Args:
        model: β-VAE 모델
        train_loader: 학습 데이터 로더
        optimizer: 옵티마이저
        device: 디바이스

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

        # Loss 계산 (model의 beta 사용)
        loss, recon, kl = beta_vae_loss(data, x_recon, mu, logvar, model.beta)

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
def evaluate_beta_vae(
    model: BetaVAE,
    test_loader,
    device: str
) -> Tuple[float, float, float]:
    """
    모델 평가 (β-VAE)

    Args:
        model: β-VAE 모델
        test_loader: 테스트 데이터 로더
        device: 디바이스

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
        loss, recon, kl = beta_vae_loss(data, x_recon, mu, logvar, model.beta)

        # 누적
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    # 평균 계산
    n_batches = len(test_loader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def compare_beta_values(
    model_class,
    beta_values: list,
    data,
    device: str
):
    """
    여러 β 값에 대한 재구성 품질 비교

    Args:
        model_class: BetaVAE 클래스
        beta_values: 테스트할 β 값 리스트
        data: 입력 데이터
        device: 디바이스

    Returns:
        results: {beta: (recon_loss, kl_loss)} 딕셔너리
    """
    results = {}

    for beta in beta_values:
        model = model_class(beta=beta).to(device)
        model.eval()

        with torch.no_grad():
            x_recon, mu, logvar = model(data)
            _, recon, kl = beta_vae_loss(data, x_recon, mu, logvar, beta)

        results[beta] = (recon.item(), kl.item())
        print(f"β={beta:.1f}: Recon={recon.item():.4f}, KL={kl.item():.4f}")

    return results


if __name__ == "__main__":
    # 간단한 테스트
    print("Testing β-VAE...")

    # 다양한 β 값으로 모델 생성
    beta_values = [0.5, 1.0, 4.0, 10.0]

    for beta in beta_values:
        model = BetaVAE(
            input_dim=784,
            hidden_dims=[512, 256],
            latent_dim=20,
            beta=beta
        )
        print(f"\nβ-VAE (β={beta})")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Beta: {model.beta}")

    # 더미 데이터로 테스트
    print("\n\nTesting forward pass...")
    dummy_data = torch.rand(32, 1, 28, 28)
    model = BetaVAE(beta=4.0)

    x_recon, mu, logvar = model(dummy_data)
    loss, recon, kl = beta_vae_loss(dummy_data, x_recon, mu, logvar, beta=4.0)

    print(f"Input shape: {dummy_data.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {recon.item():.4f}")
    print(f"KL Divergence: {kl.item():.4f}")

    # 샘플링
    samples = model.sample(num_samples=16)
    print(f"\nGenerated samples shape: {samples.shape}")

    print("\n✅ β-VAE works correctly!")
