"""
DDPM (Denoising Diffusion Probabilistic Models) 구현

Ho et al. (2020) "Denoising Diffusion Probabilistic Models"

주요 특징:
- Forward process: q(x_t | x_{t-1}) = N(√α_t x_{t-1}, (1-α_t)I)
- Reverse process: p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), Σ_θ(x_t, t))
- Loss: L = E[||ε - ε_θ(x_t, t)||²]
- Sampling: iterative denoising from x_T ~ N(0, I)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Tuple

from .noise_schedule import NoiseSchedule
from .unet import UNet


class DDPM(nn.Module):
    """
    DDPM (Denoising Diffusion Probabilistic Models)

    Forward process와 reverse process를 모두 포함하는 완전한 diffusion 모델
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule_type: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        objective: str = 'noise'  # 'noise' or 'x0'
    ):
        """
        Args:
            model: 노이즈 예측 모델 (U-Net)
            timesteps: 총 timestep 수 (기본값: 1000)
            schedule_type: Noise schedule 타입
            beta_start: β 시작값
            beta_end: β 끝값
            objective: 학습 목표 ('noise': ε 예측, 'x0': x_0 예측)
        """
        super().__init__()

        self.model = model
        self.timesteps = timesteps
        self.objective = objective

        # Noise schedule 생성
        self.noise_schedule = NoiseSchedule(
            timesteps=timesteps,
            schedule_type=schedule_type,
            beta_start=beta_start,
            beta_end=beta_end
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: x_0 → x_t

        q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)

        Args:
            x_start: 원본 이미지 x_0
            t: timestep
            noise: 노이즈 (None이면 랜덤 생성)

        Returns:
            x_t: 노이즈가 추가된 이미지
        """
        return self.noise_schedule.q_sample(x_start, t, noise)

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        학습 손실 계산

        L = E[||ε - ε_θ(x_t, t)||²]  (noise prediction)
        or
        L = E[||x_0 - x_θ(x_t, t)||²]  (x0 prediction)

        Args:
            x_start: 원본 이미지 x_0
            t: timestep
            noise: 노이즈 (None이면 랜덤 생성)

        Returns:
            loss: MSE loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Forward process: x_0 → x_t
        x_t = self.q_sample(x_start, t, noise)

        # Predict
        model_output = self.model(x_t, t)

        # Calculate loss based on objective
        if self.objective == 'noise':
            # Predict noise ε
            target = noise
        elif self.objective == 'x0':
            # Predict x_0
            target = x_start
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # MSE loss
        loss = F.mse_loss(model_output, target, reduction='mean')

        return loss

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Reverse process 한 스텝: x_t → x_{t-1}

        p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), σ_t²I)

        Args:
            x_t: 현재 노이즈 이미지
            t: 현재 timestep
            clip_denoised: 결과를 [-1, 1]로 클립할지 여부

        Returns:
            x_{t-1}: 한 스텝 디노이즈된 이미지
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Timestep tensor
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Model prediction
        model_output = self.model(x_t, t_tensor)

        # Extract coefficients
        sqrt_recip_alphas_t = self.noise_schedule._extract(
            self.noise_schedule.sqrt_recip_alphas, t_tensor, x_t.shape
        )
        betas_t = self.noise_schedule._extract(
            self.noise_schedule.betas, t_tensor, x_t.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule._extract(
            self.noise_schedule.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape
        )

        # Predict x_0 from noise
        if self.objective == 'noise':
            # x_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
            pred_noise = model_output
            pred_x0 = self.noise_schedule.predict_start_from_noise(x_t, t_tensor, pred_noise)
        elif self.objective == 'x0':
            pred_x0 = model_output
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Clip x_0
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Compute μ_θ(x_t, t)
        # μ_θ = (x_t - β_t / √(1-ᾱ_t) * ε_θ) / √α_t
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        # Add noise (except for t=0)
        if t > 0:
            posterior_variance_t = self.noise_schedule._extract(
                self.noise_schedule.posterior_variance, t_tensor, x_t.shape
            )
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: tuple,
        device: str = 'cpu',
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        완전한 샘플링: x_T ~ N(0, I) → x_0

        Args:
            shape: 생성할 이미지 shape (batch_size, C, H, W)
            device: 디바이스
            show_progress: Progress bar 표시 여부

        Returns:
            x_0: 생성된 이미지
        """
        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Iterative denoising
        timesteps = reversed(range(self.timesteps))
        if show_progress:
            timesteps = tqdm(timesteps, desc='Sampling')

        for t in timesteps:
            x_t = self.p_sample(x_t, t)

        return x_t

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        channels: int = 3,
        image_size: int = 64,
        device: str = 'cpu',
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        샘플 생성

        Args:
            batch_size: 배치 크기
            channels: 이미지 채널 수
            image_size: 이미지 크기
            device: 디바이스
            show_progress: Progress bar 표시 여부

        Returns:
            samples: 생성된 이미지 (batch_size, channels, image_size, image_size)
        """
        shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(shape, device, show_progress)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        학습용 forward pass

        Args:
            x: 입력 이미지 (batch_size, C, H, W)

        Returns:
            loss: 손실 값
        """
        batch_size = x.shape[0]
        device = x.device

        # Random timestep
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

        # Calculate loss
        return self.p_losses(x, t)


def train_step_ddpm(
    model: DDPM,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer
) -> float:
    """
    DDPM 한 스텝 학습

    Args:
        model: DDPM 모델
        images: 학습 이미지
        optimizer: 옵티마이저

    Returns:
        loss: 손실 값
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    loss = model(images)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_ddpm(
    model: DDPM,
    dataloader,
    device: str
) -> float:
    """
    DDPM 평가

    Args:
        model: DDPM 모델
        dataloader: 데이터 로더
        device: 디바이스

    Returns:
        avg_loss: 평균 손실
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, _ in dataloader:
        images = images.to(device)

        # Calculate loss
        loss = model(images)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


if __name__ == "__main__":
    print("Testing DDPM...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # U-Net 생성
    unet = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16,)
    )

    # DDPM 모델 생성
    ddpm = DDPM(
        model=unet,
        timesteps=1000,
        schedule_type='cosine',
        objective='noise'
    ).to(device)

    print(f"\nDDPM Parameters: {sum(p.numel() for p in ddpm.parameters()):,}")

    # Forward process 테스트
    print("\n1. Testing forward process...")
    x_start = torch.randn(4, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)
    x_t = ddpm.q_sample(x_start, t)
    print(f"   x_start shape: {x_start.shape}")
    print(f"   x_t shape: {x_t.shape}")
    print(f"   x_t range: [{x_t.min():.2f}, {x_t.max():.2f}]")

    # Loss 계산 테스트
    print("\n2. Testing loss calculation...")
    loss = ddpm(x_start)
    print(f"   Loss: {loss.item():.4f}")

    # 샘플링 테스트 (빠른 테스트를 위해 적은 timesteps)
    print("\n3. Testing sampling (quick test with 10 steps)...")
    ddpm.timesteps = 10
    ddpm.noise_schedule = NoiseSchedule(timesteps=10, schedule_type='cosine')

    samples = ddpm.sample(
        batch_size=2,
        channels=3,
        image_size=64,
        device=device,
        show_progress=False
    )
    print(f"   Samples shape: {samples.shape}")
    print(f"   Samples range: [{samples.min():.2f}, {samples.max():.2f}]")

    print("\n✅ DDPM works correctly!")
