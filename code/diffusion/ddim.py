"""
DDIM (Denoising Diffusion Implicit Models) Sampling

Song et al. (2020) "Denoising Diffusion Implicit Models"

주요 특징:
- Non-Markovian process
- Deterministic sampling (η=0) 또는 Stochastic (η>0)
- 적은 step으로도 고품질 샘플 생성 (예: 50 steps vs 1000 steps)
- DDPM보다 10-50배 빠른 샘플링
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional

from .ddpm import DDPM


class DDIMSampler:
    """
    DDIM Sampler

    DDPM 모델을 사용하지만, 더 빠른 샘플링을 위해 DDIM 방식 사용
    """

    def __init__(self, ddpm_model: DDPM):
        """
        Args:
            ddpm_model: 학습된 DDPM 모델
        """
        self.model = ddpm_model.model
        self.noise_schedule = ddpm_model.noise_schedule
        self.timesteps = ddpm_model.timesteps
        self.objective = ddpm_model.objective

    @torch.no_grad()
    def ddim_sample_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        DDIM 한 스텝 샘플링: x_t → x_{t_prev}

        x_{t-1} = √ᾱ_{t-1} * x̂_0 + √(1-ᾱ_{t-1}-σ_t²) * ε̂_θ + σ_t * ε

        Args:
            x_t: 현재 노이즈 이미지
            t: 현재 timestep
            t_prev: 이전 timestep (t_prev < t)
            eta: Stochasticity (0.0 = deterministic, 1.0 = DDPM)
            clip_denoised: x̂_0를 [-1, 1]로 클립할지 여부

        Returns:
            x_{t_prev}: 디노이즈된 이미지
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Timestep tensors
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)

        # Model prediction
        model_output = self.model(x_t, t_tensor)

        # Predict x_0 from noise
        if self.objective == 'noise':
            pred_noise = model_output
            pred_x0 = self.noise_schedule.predict_start_from_noise(x_t, t_tensor, pred_noise)
        elif self.objective == 'x0':
            pred_x0 = model_output
            pred_noise = (
                self.noise_schedule._extract(
                    self.noise_schedule.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape
                ) * x_t -
                self.noise_schedule._extract(
                    self.noise_schedule.sqrt_alphas_cumprod, t_tensor, x_t.shape
                ) * pred_x0
            ) / self.noise_schedule._extract(
                self.noise_schedule.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape
            )
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Clip x_0
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Get ᾱ_t and ᾱ_{t-1}
        alphas_cumprod_t = self.noise_schedule._extract(
            self.noise_schedule.alphas_cumprod, t_tensor, x_t.shape
        )

        if t_prev >= 0:
            alphas_cumprod_t_prev = self.noise_schedule._extract(
                self.noise_schedule.alphas_cumprod, t_prev_tensor, x_t.shape
            )
        else:
            # t_prev = -1일 때 (마지막 스텝)
            alphas_cumprod_t_prev = torch.ones_like(alphas_cumprod_t)

        # Calculate σ_t
        # σ_t = η * √((1-ᾱ_{t-1}) / (1-ᾱ_t)) * √(1 - ᾱ_t / ᾱ_{t-1})
        sigma_t = eta * torch.sqrt(
            (1 - alphas_cumprod_t_prev) / (1 - alphas_cumprod_t) *
            (1 - alphas_cumprod_t / alphas_cumprod_t_prev)
        )

        # Calculate direction pointing to x_t
        # √(1-ᾱ_{t-1}-σ_t²) * ε̂_θ
        dir_xt = torch.sqrt(1 - alphas_cumprod_t_prev - sigma_t ** 2) * pred_noise

        # DDIM update
        # x_{t-1} = √ᾱ_{t-1} * x̂_0 + √(1-ᾱ_{t-1}-σ_t²) * ε̂_θ + σ_t * ε
        x_prev = torch.sqrt(alphas_cumprod_t_prev) * pred_x0 + dir_xt

        # Add noise (stochastic sampling)
        if eta > 0 and t_prev >= 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: int = 50,
        eta: float = 0.0,
        device: str = 'cpu',
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        DDIM 샘플링

        Args:
            shape: 생성할 이미지 shape (batch_size, C, H, W)
            num_steps: 샘플링 스텝 수 (기본값: 50)
            eta: Stochasticity (0.0 = deterministic, 1.0 = DDPM)
            device: 디바이스
            show_progress: Progress bar 표시 여부

        Returns:
            samples: 생성된 이미지
        """
        # Timestep subsequence
        # 예: [0, 20, 40, ..., 980] (num_steps=50일 때)
        c = self.timesteps // num_steps
        timesteps = list(range(0, self.timesteps, c))

        # Reverse
        timesteps = list(reversed(timesteps))

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Iterative denoising
        if show_progress:
            timesteps_iter = tqdm(timesteps, desc=f'DDIM Sampling (eta={eta})')
        else:
            timesteps_iter = timesteps

        for i, t in enumerate(timesteps_iter):
            # Get previous timestep
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            # DDIM step
            x_t = self.ddim_sample_step(x_t, t, t_prev, eta=eta)

        return x_t

    @torch.no_grad()
    def sample_from_batch(
        self,
        batch_size: int = 16,
        channels: int = 3,
        image_size: int = 64,
        num_steps: int = 50,
        eta: float = 0.0,
        device: str = 'cpu',
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        배치 샘플 생성 (편의 함수)

        Args:
            batch_size: 배치 크기
            channels: 이미지 채널 수
            image_size: 이미지 크기
            num_steps: 샘플링 스텝 수
            eta: Stochasticity
            device: 디바이스
            show_progress: Progress bar 표시 여부

        Returns:
            samples: 생성된 이미지
        """
        shape = (batch_size, channels, image_size, image_size)
        return self.sample(shape, num_steps, eta, device, show_progress)


def compare_sampling_speed(
    ddpm_model: DDPM,
    batch_size: int = 4,
    image_size: int = 32,
    device: str = 'cpu'
):
    """
    DDPM vs DDIM 샘플링 속도 비교

    Args:
        ddpm_model: DDPM 모델
        batch_size: 배치 크기
        image_size: 이미지 크기
        device: 디바이스
    """
    import time

    print("Comparing DDPM vs DDIM sampling speed...")
    print(f"Batch size: {batch_size}, Image size: {image_size}x{image_size}")

    channels = 3
    shape = (batch_size, channels, image_size, image_size)

    # DDIM sampler
    ddim = DDIMSampler(ddpm_model)

    # Test configurations
    configs = [
        ('DDIM 50 steps (deterministic)', 50, 0.0),
        ('DDIM 100 steps (deterministic)', 100, 0.0),
        ('DDIM 50 steps (stochastic)', 50, 1.0),
    ]

    for name, num_steps, eta in configs:
        start_time = time.time()
        samples = ddim.sample(shape, num_steps=num_steps, eta=eta, device=device, show_progress=False)
        elapsed = time.time() - start_time

        print(f"\n{name}:")
        print(f"  Time: {elapsed:.2f}s ({elapsed/batch_size:.2f}s per image)")
        print(f"  Samples shape: {samples.shape}")

    # DDPM (full 1000 steps) - comment out for quick test
    # print(f"\nDDPM 1000 steps:")
    # start_time = time.time()
    # samples = ddpm_model.p_sample_loop(shape, device=device, show_progress=False)
    # elapsed = time.time() - start_time
    # print(f"  Time: {elapsed:.2f}s ({elapsed/batch_size:.2f}s per image)")


if __name__ == "__main__":
    print("Testing DDIM...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Import U-Net
    from .unet import UNet

    # Create U-Net
    unet = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=64,  # Smaller for testing
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=()
    )

    # Create DDPM
    ddpm = DDPM(
        model=unet,
        timesteps=1000,
        schedule_type='cosine',
        objective='noise'
    ).to(device)

    print(f"\nDDPM Parameters: {sum(p.numel() for p in ddpm.parameters()):,}")

    # Create DDIM sampler
    ddim = DDIMSampler(ddpm)

    # Test DDIM sampling
    print("\nTesting DDIM sampling (50 steps, deterministic)...")
    samples = ddim.sample_from_batch(
        batch_size=4,
        channels=3,
        image_size=32,
        num_steps=50,
        eta=0.0,
        device=device,
        show_progress=True
    )

    print(f"\nSamples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.2f}, {samples.max():.2f}]")

    # Compare sampling speeds
    print("\n" + "="*70)
    compare_sampling_speed(ddpm, batch_size=2, image_size=32, device=device)

    print("\n✅ DDIM works correctly!")
