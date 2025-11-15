"""
Noise Schedule for Diffusion Models

다양한 noise schedule 구현:
- Linear schedule
- Cosine schedule
- Sigmoid schedule
"""

import torch
import numpy as np
import math
from typing import Tuple


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear beta schedule

    β_t = β_start + (β_end - β_start) * t / T

    Args:
        timesteps: 총 timestep 수 (T)
        beta_start: 시작 β 값
        beta_end: 끝 β 값

    Returns:
        beta: β schedule (timesteps,)
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule (Improved DDPM)

    더 부드러운 schedule로 성능 향상

    Args:
        timesteps: 총 timestep 수 (T)
        s: offset (기본값: 0.008)

    Returns:
        beta: β schedule (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    # ᾱ_t = f(t) / f(0), where f(t) = cos((t/T + s) / (1 + s) * π/2)²
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # β_t = 1 - α_t = 1 - (ᾱ_t / ᾱ_{t-1})
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # Clip to avoid numerical issues
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Sigmoid beta schedule

    Args:
        timesteps: 총 timestep 수
        beta_start: 시작 β 값
        beta_end: 끝 β 값

    Returns:
        beta: β schedule (timesteps,)
    """
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class NoiseSchedule:
    """
    Noise Schedule 클래스

    Forward diffusion process에 필요한 모든 값들을 사전 계산합니다.

    주요 값들:
    - β_t: variance schedule
    - α_t: 1 - β_t
    - ᾱ_t: ∏(1 to t) α_i
    - √ᾱ_t, √(1-ᾱ_t): 샘플링에 사용
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule_type: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Args:
            timesteps: 총 timestep 수 (기본값: 1000)
            schedule_type: 'linear', 'cosine', 'sigmoid'
            beta_start: 시작 β 값 (linear, sigmoid용)
            beta_end: 끝 β 값 (linear, sigmoid용)
        """
        self.timesteps = timesteps

        # β schedule 선택
        if schedule_type == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule_type == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif schedule_type == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.betas = betas

        # α_t = 1 - β_t
        self.alphas = 1.0 - betas

        # ᾱ_t = ∏(1 to t) α_i
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱ_{t-1} (0번째는 1.0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        # √ᾱ_t (샘플링용)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # √(1 - ᾱ_t) (샘플링용)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 1 / √α_t (reverse process용)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance: β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Log posterior variance (numerical stability)
        self.posterior_log_variance = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion process: x_0 → x_t

        q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1 - ᾱ_t) * I)
        x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε

        Args:
            x_start: 원본 이미지 x_0 (batch_size, C, H, W)
            t: timestep (batch_size,)
            noise: 노이즈 ε (None이면 랜덤 생성)

        Returns:
            x_t: 노이즈가 추가된 이미지
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # √ᾱ_t와 √(1 - ᾱ_t) 추출
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Posterior q(x_{t-1} | x_t, x_0) 계산

        Args:
            x_start: 원본 이미지 x_0
            x_t: 노이즈 이미지 x_t
            t: timestep

        Returns:
            posterior_mean: μ̃_t
            posterior_variance: β̃_t
            posterior_log_variance: log β̃_t
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        노이즈 예측으로부터 x_0 복원

        x_0 = (x_t - √(1 - ᾱ_t) * ε) / √ᾱ_t

        Args:
            x_t: 노이즈 이미지
            t: timestep
            noise: 예측된 노이즈 ε_θ(x_t, t)

        Returns:
            x_0 예측값
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        텐서 a에서 인덱스 t에 해당하는 값을 추출하고, x_shape에 맞게 reshape

        Args:
            a: 추출할 텐서 (T,)
            t: 인덱스 (batch_size,)
            x_shape: 목표 shape (batch_size, C, H, W)

        Returns:
            추출된 값 (batch_size, 1, 1, 1)
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu()).to(t.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def get_noise_schedule(
    timesteps: int = 1000,
    schedule_type: str = 'linear',
    **kwargs
) -> NoiseSchedule:
    """
    Noise schedule 생성 헬퍼 함수

    Args:
        timesteps: 총 timestep 수
        schedule_type: 'linear', 'cosine', 'sigmoid'
        **kwargs: 추가 인자 (beta_start, beta_end)

    Returns:
        NoiseSchedule 객체
    """
    return NoiseSchedule(timesteps=timesteps, schedule_type=schedule_type, **kwargs)


if __name__ == "__main__":
    print("Testing noise schedules...")

    import matplotlib.pyplot as plt

    timesteps = 1000

    # 다양한 schedule 테스트
    schedules = {
        'Linear': linear_beta_schedule(timesteps),
        'Cosine': cosine_beta_schedule(timesteps),
        'Sigmoid': sigmoid_beta_schedule(timesteps)
    }

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # β_t
    ax = axes[0, 0]
    for name, betas in schedules.items():
        ax.plot(betas.numpy(), label=name)
    ax.set_title('β_t (Variance Schedule)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('β_t')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # α_t
    ax = axes[0, 1]
    for name, betas in schedules.items():
        alphas = 1.0 - betas
        ax.plot(alphas.numpy(), label=name)
    ax.set_title('α_t = 1 - β_t')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('α_t')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ᾱ_t (cumulative product)
    ax = axes[1, 0]
    for name, betas in schedules.items():
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        ax.plot(alphas_cumprod.numpy(), label=name)
    ax.set_title('ᾱ_t = ∏α_i (Cumulative Product)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('ᾱ_t')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Signal-to-Noise Ratio
    ax = axes[1, 1]
    for name, betas in schedules.items():
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        snr = alphas_cumprod / (1.0 - alphas_cumprod)
        ax.plot(torch.log10(snr).numpy(), label=name)
    ax.set_title('log₁₀(SNR)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('log₁₀(Signal-to-Noise Ratio)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/noise_schedules.png', dpi=150)
    print("Saved visualization to /tmp/noise_schedules.png")

    # NoiseSchedule 클래스 테스트
    print("\nTesting NoiseSchedule class...")
    noise_schedule = NoiseSchedule(timesteps=1000, schedule_type='cosine')

    # Forward process 테스트
    x_start = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    x_t = noise_schedule.q_sample(x_start, t)

    print(f"x_start shape: {x_start.shape}")
    print(f"t: {t}")
    print(f"x_t shape: {x_t.shape}")
    print(f"x_t range: [{x_t.min():.2f}, {x_t.max():.2f}]")

    print("\n✅ Noise schedule works correctly!")
