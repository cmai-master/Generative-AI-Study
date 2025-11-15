"""
U-Net for Diffusion Models

Diffusion 모델용 U-Net 아키텍처 구현

주요 특징:
- Time embedding (sinusoidal positional encoding)
- ResNet blocks with time conditioning
- Self-attention layers
- Down/Up sampling with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal Position Embedding for timesteps

    Transformer의 positional encoding과 동일한 방식
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding 차원
        """
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,)

        Returns:
            embeddings: (batch_size, dim)
        """
        device = timesteps.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class ResBlock(nn.Module):
    """
    Residual Block with Time Embedding

    x → [GroupNorm → SiLU → Conv] → + time_emb → [GroupNorm → SiLU → Conv] → + x
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 32,
        dropout: float = 0.0
    ):
        """
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
            time_emb_dim: Time embedding 차원
            num_groups: GroupNorm 그룹 수
            dropout: Dropout 비율
        """
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, H, W)
            time_emb: (batch_size, time_emb_dim)

        Returns:
            output: (batch_size, out_channels, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = F.silu(time_emb)
        time_emb = self.time_emb_proj(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Self-Attention Block

    Q, K, V를 사용한 multi-head self-attention
    """

    def __init__(self, channels: int, num_groups: int = 32, num_heads: int = 4):
        """
        Args:
            channels: 채널 수
            num_groups: GroupNorm 그룹 수
            num_heads: Attention head 수
        """
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, H, W)

        Returns:
            output: (batch_size, channels, H, W)
        """
        batch_size, channels, height, width = x.shape

        # Normalize
        h = self.norm(x)

        # Q, K, V
        qkv = self.qkv(h)  # (B, C*3, H, W)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2)  # (B, num_heads, head_dim, H*W)
        out = out.reshape(batch_size, channels, height, width)

        # Project
        out = self.proj_out(out)

        return x + out  # Residual connection


class Downsample(nn.Module):
    """Downsampling layer (2x)"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer (2x)"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for Diffusion Models

    Architecture:
        Input → Down1 → Down2 → Down3 → BottleNeck → Up3 → Up2 → Up1 → Output
                  ↓        ↓        ↓                    ↑      ↑      ↑
                  └────────┴────────┴────────────────────┴──────┴──────┘
                                  (skip connections)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.0,
        num_heads: int = 4,
        time_emb_dim: Optional[int] = None
    ):
        """
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
            model_channels: 기본 채널 수
            channel_mult: 각 레벨의 채널 배수
            num_res_blocks: 각 레벨의 ResBlock 수
            attention_resolutions: Attention을 적용할 해상도
            dropout: Dropout 비율
            num_heads: Attention head 수
            time_emb_dim: Time embedding 차원 (None이면 model_channels * 4)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels

        if time_emb_dim is None:
            time_emb_dim = model_channels * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = [model_channels]
        now_channels = model_channels

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(now_channels, out_ch, time_emb_dim, dropout=dropout)
                ]
                now_channels = out_ch

                # Add attention if needed
                if i in attention_resolutions or (32 // (2**i)) in attention_resolutions:
                    layers.append(AttentionBlock(now_channels, num_heads=num_heads))

                self.down_blocks.append(nn.ModuleList(layers))
                channels.append(now_channels)

            # Downsample (except last level)
            if i != len(channel_mult) - 1:
                self.down_samples.append(Downsample(now_channels))
                channels.append(now_channels)

        # Bottleneck
        self.mid_block1 = ResBlock(now_channels, now_channels, time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(now_channels, num_heads=num_heads)
        self.mid_block2 = ResBlock(now_channels, now_channels, time_emb_dim, dropout=dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult

            for j in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        now_channels + channels.pop(),
                        out_ch,
                        time_emb_dim,
                        dropout=dropout
                    )
                ]
                now_channels = out_ch

                # Add attention if needed
                level = len(channel_mult) - 1 - i
                if level in attention_resolutions or (32 // (2**level)) in attention_resolutions:
                    layers.append(AttentionBlock(now_channels, num_heads=num_heads))

                self.up_blocks.append(nn.ModuleList(layers))

            # Upsample (except last level)
            if i != len(channel_mult) - 1:
                self.up_samples.append(Upsample(now_channels))

        # Output convolution
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, H, W)
            timesteps: (batch_size,)

        Returns:
            output: (batch_size, out_channels, H, W)
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Input
        h = self.input_conv(x)

        # Downsampling with skip connections
        hs = [h]

        for i, layers in enumerate(self.down_blocks):
            for layer in layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, time_emb)
                else:  # AttentionBlock
                    h = layer(h)
            hs.append(h)

            # Downsample
            if i < len(self.down_samples):
                h = self.down_samples[i](h)
                hs.append(h)

        # Bottleneck
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # Upsampling with skip connections
        for i, layers in enumerate(self.up_blocks):
            # Concatenate skip connection
            h = torch.cat([h, hs.pop()], dim=1)

            for layer in layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, time_emb)
                else:  # AttentionBlock
                    h = layer(h)

            # Upsample
            if i < len(self.up_samples):
                h = self.up_samples[i](h)

        # Output
        return self.output_conv(h)


if __name__ == "__main__":
    print("Testing U-Net...")

    # 모델 생성
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        num_heads=4
    )

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass 테스트
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))

    output = model(x, timesteps)

    print(f"\nInput shape: {x.shape}")
    print(f"Timesteps: {timesteps}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")

    # Time embedding 테스트
    time_embed = model.time_embed(timesteps)
    print(f"\nTime embedding shape: {time_embed.shape}")

    print("\n✅ U-Net works correctly!")
