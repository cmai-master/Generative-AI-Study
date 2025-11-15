"""
Generate samples from trained diffusion model

Usage:
    # DDPM (1000 steps)
    python generate.py --checkpoint checkpoints/best_model.pth --num_samples 64 --output_dir samples/ddpm/

    # DDIM (50 steps, fast!)
    python generate.py --checkpoint checkpoints/best_model.pth --num_samples 64 --sampler ddim --num_steps 50 --output_dir samples/ddim/
"""

import argparse
import torch
from pathlib import Path
import sys
from torchvision.utils import save_image

# Add code directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "code"))

from diffusion.ddpm import DDPM
from diffusion.ddim import DDIMSampler
from diffusion.unet import UNet


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"🚀 Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Build U-Net
    unet = UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        model_channels=config.model.model_channels,
        channel_mult=tuple(config.model.channel_mult),
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=tuple(config.model.attention_resolutions),
        dropout=config.model.dropout,
        num_heads=config.model.num_heads
    )

    # Build DDPM
    ddpm = DDPM(
        model=unet,
        timesteps=config.diffusion.timesteps,
        schedule_type=config.diffusion.schedule_type,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        objective=config.diffusion.objective
    )

    # Load weights
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.to(device)
    ddpm.eval()

    print(f"✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint['epoch'] + 1}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")

    return ddpm, config


def generate_samples(
    model,
    config,
    num_samples,
    sampler_type,
    num_steps,
    device,
    seed
):
    """Generate samples"""
    # Set seed
    torch.manual_seed(seed)

    print(f"\n🎨 Generating {num_samples} samples...")
    print(f"   Sampler: {sampler_type}")
    print(f"   Steps: {num_steps}")
    print(f"   Seed: {seed}")
    print(f"   Device: {device}")

    if sampler_type == 'ddim':
        ddim = DDIMSampler(model)
        samples = ddim.sample_from_batch(
            batch_size=num_samples,
            channels=config.model.in_channels,
            image_size=config.model.image_size,
            num_steps=num_steps,
            eta=0.0,  # Deterministic
            device=device,
            show_progress=True
        )
    else:  # ddpm
        samples = model.sample(
            batch_size=num_samples,
            channels=config.model.in_channels,
            image_size=config.model.image_size,
            device=device,
            show_progress=True
        )

    # Denormalize to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)

    print(f"✅ Generated {num_samples} samples!")

    return samples


def main(args):
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Generate samples
    samples = generate_samples(
        model,
        config,
        args.num_samples,
        args.sampler,
        args.num_steps,
        device,
        args.seed
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as grid
    grid_path = output_dir / f"samples_grid_{args.sampler}_{args.num_steps}steps.png"
    save_image(samples, grid_path, nrow=8, padding=2)
    print(f"\n💾 Saved grid: {grid_path}")

    # Save individual images
    if args.save_individual:
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)

        for i, img in enumerate(samples):
            save_image(img, individual_dir / f"sample_{i:04d}.png")

        print(f"💾 Saved {len(samples)} individual images to {individual_dir}")

    print("\n✅ Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples from trained diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--sampler', type=str, default='ddim',
                        choices=['ddpm', 'ddim'],
                        help='Sampling method')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of sampling steps (for DDIM)')
    parser.add_argument('--output_dir', type=str, default='samples/',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_individual', action='store_true',
                        help='Save individual images')

    args = parser.parse_args()
    main(args)
