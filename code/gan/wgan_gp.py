"""
WGAN-GP (Wasserstein GAN with Gradient Penalty) 구현

Gulrajani et al. (2017) "Improved Training of Wasserstein GANs"

주요 특징:
- Wasserstein Distance 사용
- Gradient Penalty로 Lipschitz Constraint 강제
- Critic (Discriminator)의 출력에 Sigmoid 없음
- 더 안정적인 학습
- Mode Collapse 완화
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Tuple


class WGANGenerator(nn.Module):
    """
    WGAN-GP Generator

    DCGAN과 유사한 구조이지만, WGAN에 최적화됨
    """

    def __init__(
        self,
        latent_dim: int = 100,
        feature_map_size: int = 64,
        num_channels: int = 1
    ):
        """
        Args:
            latent_dim: 잠재 벡터 차원
            feature_map_size: Feature map 크기
            num_channels: 출력 이미지 채널
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.feature_map_size = feature_map_size

        # Initial projection
        self.project = nn.Sequential(
            nn.Linear(latent_dim, feature_map_size * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_size * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Convolutional layers
        self.convs = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(
                feature_map_size * 8, feature_map_size * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),

            # 8x8 → 16x16
            nn.ConvTranspose2d(
                feature_map_size * 4, feature_map_size * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),

            # 16x16 → 32x32
            nn.ConvTranspose2d(
                feature_map_size * 2, feature_map_size,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),

            # 32x32 → 64x64
            nn.ConvTranspose2d(
                feature_map_size, num_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.project(z)
        x = x.view(-1, self.feature_map_size * 8, 4, 4)
        x = self.convs(x)
        return x


class WGANCritic(nn.Module):
    """
    WGAN-GP Critic (Discriminator)

    주요 차이점:
    - 출력에 Sigmoid 없음 (실수값 출력)
    - BatchNorm 대신 LayerNorm 사용 권장 (GP와의 호환성)
    - 1-Lipschitz constraint는 Gradient Penalty로 강제
    """

    def __init__(
        self,
        num_channels: int = 1,
        feature_map_size: int = 64,
        use_layer_norm: bool = True
    ):
        """
        Args:
            num_channels: 입력 이미지 채널
            feature_map_size: Feature map 크기
            use_layer_norm: LayerNorm 사용 여부 (GP와 호환성 좋음)
        """
        super().__init__()

        self.feature_map_size = feature_map_size
        self.use_layer_norm = use_layer_norm

        # Convolutional layers
        layers = []

        # Layer 1: 64x64 → 32x32
        layers.append(nn.Conv2d(
            num_channels, feature_map_size,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer 2: 32x32 → 16x16
        layers.append(nn.Conv2d(
            feature_map_size, feature_map_size * 2,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        if use_layer_norm:
            layers.append(nn.LayerNorm([feature_map_size * 2, 16, 16]))
        else:
            layers.append(nn.BatchNorm2d(feature_map_size * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer 3: 16x16 → 8x8
        layers.append(nn.Conv2d(
            feature_map_size * 2, feature_map_size * 4,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        if use_layer_norm:
            layers.append(nn.LayerNorm([feature_map_size * 4, 8, 8]))
        else:
            layers.append(nn.BatchNorm2d(feature_map_size * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer 4: 8x8 → 4x4
        layers.append(nn.Conv2d(
            feature_map_size * 4, feature_map_size * 8,
            kernel_size=4, stride=2, padding=1, bias=False
        ))
        if use_layer_norm:
            layers.append(nn.LayerNorm([feature_map_size * 8, 4, 4]))
        else:
            layers.append(nn.BatchNorm2d(feature_map_size * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layer: 4x4 → 1
        layers.append(nn.Conv2d(
            feature_map_size * 8, 1,
            kernel_size=4, stride=1, padding=0, bias=False
        ))
        # No Sigmoid! (WGAN의 핵심)

        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 이미지 (batch_size, num_channels, H, W)

        Returns:
            Critic score (batch_size, 1, 1, 1) - 실수값
        """
        return self.convs(x)


def compute_gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Gradient Penalty 계산

    1-Lipschitz constraint를 강제하기 위한 페널티:
    GP = λ * E[(||∇_x̂ D(x̂)||_2 - 1)²]

    where x̂ = α * x_real + (1-α) * x_fake

    Args:
        critic: Critic 네트워크
        real_images: 진짜 이미지
        fake_images: 가짜 이미지
        device: 디바이스

    Returns:
        gradient_penalty: Gradient Penalty 값
    """
    batch_size = real_images.size(0)

    # Random interpolation factor α ~ U(0, 1)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

    # Interpolated images: x̂ = α * x_real + (1-α) * x_fake
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)

    # Critic score for interpolated images
    critic_interpolated = critic(interpolated)

    # Compute gradients
    gradients = autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten gradients
    gradients = gradients.view(batch_size, -1)

    # Compute gradient norm
    gradient_norm = gradients.norm(2, dim=1)

    # Gradient Penalty: (||∇||_2 - 1)²
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


class WGAN_GP:
    """
    WGAN-GP (Wasserstein GAN with Gradient Penalty)

    Generator와 Critic을 함께 관리하는 클래스
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_channels: int = 1,
        feature_map_size: int = 64,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        device: str = 'cpu'
    ):
        """
        Args:
            latent_dim: 잠재 벡터 차원
            num_channels: 이미지 채널 수
            feature_map_size: Feature map 크기
            lambda_gp: Gradient Penalty 가중치 (기본값: 10.0)
            n_critic: Generator 1회당 Critic 학습 횟수 (기본값: 5)
            device: 디바이스
        """
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.device = device

        # Generator와 Critic 생성
        self.generator = WGANGenerator(
            latent_dim=latent_dim,
            feature_map_size=feature_map_size,
            num_channels=num_channels
        ).to(device)

        self.critic = WGANCritic(
            num_channels=num_channels,
            feature_map_size=feature_map_size,
            use_layer_norm=True
        ).to(device)

        # 가중치 초기화
        self.generator.apply(self._weights_init)
        self.critic.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        """가중치 초기화"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        """랜덤 잠재 벡터 샘플링"""
        return torch.randn(batch_size, self.latent_dim).to(self.device)

    def generate(self, z: torch.Tensor = None, num_samples: int = 64) -> torch.Tensor:
        """이미지 생성"""
        if z is None:
            z = self.sample_noise(num_samples)

        self.generator.eval()
        with torch.no_grad():
            images = self.generator(z)

        return images


def train_step_wgan_gp(
    generator: nn.Module,
    critic: nn.Module,
    real_images: torch.Tensor,
    optimizer_g: torch.optim.Optimizer,
    optimizer_c: torch.optim.Optimizer,
    latent_dim: int,
    lambda_gp: float,
    device: str,
    update_generator: bool = True
) -> Tuple[float, float, float]:
    """
    WGAN-GP 한 스텝 학습

    Args:
        generator: Generator 모델
        critic: Critic 모델
        real_images: 진짜 이미지
        optimizer_g: Generator optimizer
        optimizer_c: Critic optimizer
        latent_dim: 잠재 벡터 차원
        lambda_gp: Gradient Penalty 가중치
        device: 디바이스
        update_generator: Generator 업데이트 여부 (n_critic 주기)

    Returns:
        c_loss: Critic 손실
        g_loss: Generator 손실 (update_generator=True일 때만 의미)
        gp: Gradient Penalty 값
    """
    batch_size = real_images.size(0)
    real_images = real_images.to(device)

    # ========== Train Critic ==========
    optimizer_c.zero_grad()

    # Real images
    real_validity = critic(real_images)

    # Fake images
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_images = generator(z)
    fake_validity = critic(fake_images.detach())

    # Gradient Penalty
    gp = compute_gradient_penalty(critic, real_images, fake_images, device)

    # Wasserstein loss with Gradient Penalty
    # Critic wants to maximize: E[D(real)] - E[D(fake)]
    # → Minimize: -E[D(real)] + E[D(fake)] + λ * GP
    c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

    c_loss.backward()
    optimizer_c.step()

    # ========== Train Generator ==========
    g_loss = 0.0
    if update_generator:
        optimizer_g.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        fake_validity = critic(fake_images)

        # Generator wants to maximize E[D(fake)]
        # → Minimize: -E[D(fake)]
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        optimizer_g.step()

        g_loss = g_loss.item()

    return c_loss.item(), g_loss, gp.item()


if __name__ == "__main__":
    # 간단한 테스트
    print("Testing WGAN-GP...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 모델 생성
    wgan_gp = WGAN_GP(
        latent_dim=100,
        num_channels=1,
        feature_map_size=64,
        lambda_gp=10.0,
        n_critic=5,
        device=device
    )

    print(f"\nGenerator Parameters: {sum(p.numel() for p in wgan_gp.generator.parameters()):,}")
    print(f"Critic Parameters: {sum(p.numel() for p in wgan_gp.critic.parameters()):,}")

    # 더미 데이터
    batch_size = 16
    dummy_images = torch.randn(batch_size, 1, 64, 64).to(device)

    # Generator 테스트
    z = wgan_gp.sample_noise(batch_size)
    generated = wgan_gp.generator(z)
    print(f"\nGenerated images shape: {generated.shape}")
    print(f"Generated images range: [{generated.min().item():.2f}, {generated.max().item():.2f}]")

    # Critic 테스트
    real_score = wgan_gp.critic(dummy_images)
    fake_score = wgan_gp.critic(generated)
    print(f"\nCritic output shape: {real_score.shape}")
    print(f"Real score mean: {real_score.mean().item():.4f}")
    print(f"Fake score mean: {fake_score.mean().item():.4f}")

    # Gradient Penalty 테스트
    gp = compute_gradient_penalty(wgan_gp.critic, dummy_images, generated, device)
    print(f"\nGradient Penalty: {gp.item():.4f}")

    # 학습 스텝 테스트
    optimizer_g = torch.optim.Adam(wgan_gp.generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizer_c = torch.optim.Adam(wgan_gp.critic.parameters(), lr=0.0001, betas=(0.0, 0.9))

    c_loss, g_loss, gp_value = train_step_wgan_gp(
        wgan_gp.generator, wgan_gp.critic,
        dummy_images, optimizer_g, optimizer_c,
        wgan_gp.latent_dim, wgan_gp.lambda_gp,
        device, update_generator=True
    )

    print(f"\nTrain step test:")
    print(f"  Critic Loss: {c_loss:.4f}")
    print(f"  Generator Loss: {g_loss:.4f}")
    print(f"  Gradient Penalty: {gp_value:.4f}")

    print("\n✅ WGAN-GP works correctly!")
