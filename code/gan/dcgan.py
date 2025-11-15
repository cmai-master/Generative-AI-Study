"""
DCGAN (Deep Convolutional GAN) 구현

Radford et al. (2015) "Unsupervised Representation Learning with
Deep Convolutional Generative Adversarial Networks"

주요 특징:
- Convolutional 레이어 사용 (No fully connected hidden layers)
- Batch Normalization 사용
- ReLU (Generator), LeakyReLU (Discriminator)
- Transposed Convolution (Decoder/Generator)
- Adam optimizer with β1=0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Generator(nn.Module):
    """
    DCGAN Generator

    잠재 벡터 z를 이미지로 생성하는 네트워크

    Architecture:
        z (latent_dim)
        → Linear → Reshape
        → ConvTranspose2d + BatchNorm + ReLU
        → ConvTranspose2d + BatchNorm + ReLU
        → ConvTranspose2d + BatchNorm + ReLU
        → ConvTranspose2d + Tanh
        → Image (C, H, W)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        feature_map_size: int = 64,
        num_channels: int = 1
    ):
        """
        Args:
            latent_dim: 잠재 벡터 차원 (기본값: 100)
            feature_map_size: Feature map 크기 (기본값: 64)
            num_channels: 출력 이미지 채널 (1: grayscale, 3: RGB)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.feature_map_size = feature_map_size

        # Initial projection: z → feature map
        # (latent_dim) → (feature_map_size * 8) * 4 * 4
        self.project = nn.Sequential(
            nn.Linear(latent_dim, feature_map_size * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_size * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Convolutional layers
        # 4x4 → 8x8 → 16x16 → 32x32 (for 32x32 output)
        # or 4x4 → 7x7 → 14x14 → 28x28 (for 28x28 output)
        self.convs = nn.Sequential(
            # Layer 1: (feature_map_size * 8, 4, 4) → (feature_map_size * 4, 8, 8)
            nn.ConvTranspose2d(
                feature_map_size * 8, feature_map_size * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),

            # Layer 2: (feature_map_size * 4, 8, 8) → (feature_map_size * 2, 16, 16)
            nn.ConvTranspose2d(
                feature_map_size * 4, feature_map_size * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),

            # Layer 3: (feature_map_size * 2, 16, 16) → (feature_map_size, 32, 32)
            nn.ConvTranspose2d(
                feature_map_size * 2, feature_map_size,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),

            # Layer 4: (feature_map_size, 32, 32) → (num_channels, 64, 64)
            nn.ConvTranspose2d(
                feature_map_size, num_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            z: 잠재 벡터 (batch_size, latent_dim)

        Returns:
            Generated images (batch_size, num_channels, 64, 64)
        """
        # Project and reshape
        x = self.project(z)
        x = x.view(-1, self.feature_map_size * 8, 4, 4)

        # Generate image
        x = self.convs(x)

        return x


class Discriminator(nn.Module):
    """
    DCGAN Discriminator

    이미지가 진짜인지 가짜인지 판별하는 네트워크

    Architecture:
        Image (C, H, W)
        → Conv2d + LeakyReLU
        → Conv2d + BatchNorm + LeakyReLU
        → Conv2d + BatchNorm + LeakyReLU
        → Conv2d + BatchNorm + LeakyReLU
        → Flatten → Linear
        → Sigmoid (probability)
    """

    def __init__(
        self,
        num_channels: int = 1,
        feature_map_size: int = 64
    ):
        """
        Args:
            num_channels: 입력 이미지 채널 (1: grayscale, 3: RGB)
            feature_map_size: Feature map 크기 (기본값: 64)
        """
        super().__init__()

        self.feature_map_size = feature_map_size

        # Convolutional layers
        # 64x64 → 32x32 → 16x16 → 8x8 → 4x4
        self.convs = nn.Sequential(
            # Layer 1: (num_channels, 64, 64) → (feature_map_size, 32, 32)
            nn.Conv2d(
                num_channels, feature_map_size,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: (feature_map_size, 32, 32) → (feature_map_size * 2, 16, 16)
            nn.Conv2d(
                feature_map_size, feature_map_size * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: (feature_map_size * 2, 16, 16) → (feature_map_size * 4, 8, 8)
            nn.Conv2d(
                feature_map_size * 2, feature_map_size * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: (feature_map_size * 4, 8, 8) → (feature_map_size * 8, 4, 4)
            nn.Conv2d(
                feature_map_size * 4, feature_map_size * 8,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final layer: (feature_map_size * 8, 4, 4) → (1, 1, 1)
            nn.Conv2d(
                feature_map_size * 8, 1,
                kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 이미지 (batch_size, num_channels, 64, 64)

        Returns:
            판별 확률 (batch_size, 1, 1, 1)
        """
        return self.convs(x)


class DCGAN:
    """
    DCGAN (Deep Convolutional GAN) 모델

    Generator와 Discriminator를 함께 관리하는 클래스
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_channels: int = 1,
        feature_map_size: int = 64,
        device: str = 'cpu'
    ):
        """
        Args:
            latent_dim: 잠재 벡터 차원
            num_channels: 이미지 채널 수
            feature_map_size: Feature map 크기
            device: 디바이스
        """
        self.latent_dim = latent_dim
        self.device = device

        # Generator와 Discriminator 생성
        self.generator = Generator(
            latent_dim=latent_dim,
            feature_map_size=feature_map_size,
            num_channels=num_channels
        ).to(device)

        self.discriminator = Discriminator(
            num_channels=num_channels,
            feature_map_size=feature_map_size
        ).to(device)

        # 가중치 초기화
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        """
        DCGAN 논문의 가중치 초기화

        - Conv/ConvTranspose: mean=0, std=0.02
        - BatchNorm: weight=1, bias=0
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        """
        랜덤 잠재 벡터 샘플링

        Args:
            batch_size: 배치 크기

        Returns:
            z ~ N(0, 1) (batch_size, latent_dim)
        """
        return torch.randn(batch_size, self.latent_dim).to(self.device)

    def generate(self, z: torch.Tensor = None, num_samples: int = 64) -> torch.Tensor:
        """
        이미지 생성

        Args:
            z: 잠재 벡터 (None이면 랜덤 샘플링)
            num_samples: 생성할 이미지 수 (z가 None일 때)

        Returns:
            Generated images
        """
        if z is None:
            z = self.sample_noise(num_samples)

        self.generator.eval()
        with torch.no_grad():
            images = self.generator(z)

        return images


def train_step_dcgan(
    generator: nn.Module,
    discriminator: nn.Module,
    real_images: torch.Tensor,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    latent_dim: int,
    device: str
) -> Tuple[float, float]:
    """
    DCGAN 한 스텝 학습

    Args:
        generator: Generator 모델
        discriminator: Discriminator 모델
        real_images: 진짜 이미지
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        latent_dim: 잠재 벡터 차원
        device: 디바이스

    Returns:
        d_loss: Discriminator 손실
        g_loss: Generator 손실
    """
    batch_size = real_images.size(0)
    real_images = real_images.to(device)

    # Labels
    real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)

    # ========== Train Discriminator ==========
    optimizer_d.zero_grad()

    # Real images
    real_outputs = discriminator(real_images)
    d_loss_real = F.binary_cross_entropy(real_outputs, real_labels)

    # Fake images
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_images = generator(z)
    fake_outputs = discriminator(fake_images.detach())
    d_loss_fake = F.binary_cross_entropy(fake_outputs, fake_labels)

    # Total discriminator loss
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_d.step()

    # ========== Train Generator ==========
    optimizer_g.zero_grad()

    # Generator wants discriminator to think fake images are real
    fake_outputs = discriminator(fake_images)
    g_loss = F.binary_cross_entropy(fake_outputs, real_labels)

    g_loss.backward()
    optimizer_g.step()

    return d_loss.item(), g_loss.item()


if __name__ == "__main__":
    # 간단한 테스트
    print("Testing DCGAN...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 모델 생성
    dcgan = DCGAN(
        latent_dim=100,
        num_channels=1,
        feature_map_size=64,
        device=device
    )

    print(f"\nGenerator Parameters: {sum(p.numel() for p in dcgan.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in dcgan.discriminator.parameters()):,}")

    # 더미 데이터
    batch_size = 16
    dummy_images = torch.randn(batch_size, 1, 64, 64).to(device)

    # Generator 테스트
    z = dcgan.sample_noise(batch_size)
    generated = dcgan.generator(z)
    print(f"\nGenerated images shape: {generated.shape}")
    print(f"Generated images range: [{generated.min().item():.2f}, {generated.max().item():.2f}]")

    # Discriminator 테스트
    real_pred = dcgan.discriminator(dummy_images)
    fake_pred = dcgan.discriminator(generated)
    print(f"\nReal prediction shape: {real_pred.shape}")
    print(f"Real predictions mean: {real_pred.mean().item():.4f}")
    print(f"Fake predictions mean: {fake_pred.mean().item():.4f}")

    # 학습 스텝 테스트
    optimizer_g = torch.optim.Adam(dcgan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(dcgan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    d_loss, g_loss = train_step_dcgan(
        dcgan.generator, dcgan.discriminator,
        dummy_images, optimizer_g, optimizer_d,
        dcgan.latent_dim, device
    )

    print(f"\nTrain step test:")
    print(f"  D Loss: {d_loss:.4f}")
    print(f"  G Loss: {g_loss:.4f}")

    print("\n✅ DCGAN works correctly!")
