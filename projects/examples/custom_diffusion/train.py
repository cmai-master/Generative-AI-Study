"""
Training script for Custom Diffusion

Usage:
    python train.py --config configs/ddpm_mnist.yaml
"""

import argparse
import yaml
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import sys
from tqdm import tqdm
import time

# Add code directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "code"))

# Import diffusion models
from diffusion.ddpm import DDPM
from diffusion.ddim import DDIMSampler
from diffusion.unet import UNet

# Import utilities (if available)
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "templates/generative_model_template"))
    from utils.logger import Logger
    from utils.checkpoint import CheckpointManager
    HAS_UTILS = True
except:
    HAS_UTILS = False
    print("⚠️  Utils not available, using basic logging")


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert nested dicts to namespace for easier access
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    return dict_to_namespace(config_dict)


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    """Get computing device"""
    if config.training.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"🖥️  Using CPU")

    return device


def build_model(config):
    """Build DDPM model"""
    print(f"🏗️  Building model: {config.model.type}")

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

    return ddpm


def build_dataloader(config):
    """Build data loaders"""
    print(f"📊 Loading dataset: {config.data.dataset}")

    # MNIST transforms
    transform_list = [
        transforms.Resize(config.data.image_size),
        transforms.ToTensor(),
    ]

    if config.data.normalize:
        # Normalize to [-1, 1]
        transform_list.append(transforms.Normalize([0.5], [0.5]))

    transform = transforms.Compose(transform_list)

    # Load MNIST
    train_dataset = datasets.MNIST(
        root=config.data.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = datasets.MNIST(
        root=config.data.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, scaler, device, config, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs}")

    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if config.training.mixed_precision:
            from torch.cuda.amp import autocast

            with autocast():
                loss = model(images)

            scaler.scale(loss).backward()

            # Gradient clipping
            if config.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.gradient_clip
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(images)
            loss.backward()

            # Gradient clipping
            if config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.gradient_clip
                )

            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, device, config):
    """Validation"""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for images, _ in tqdm(val_loader, desc="Validation"):
        images = images.to(device)

        loss = model(images)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def generate_samples(model, config, device, epoch):
    """Generate and save samples"""
    model.eval()

    save_dir = Path(config.logging.log_dir) / config.logging.experiment_name / "samples"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎨 Generating {config.logging.num_samples} samples...")

    # Generate with DDIM (faster)
    ddim = DDIMSampler(model)
    samples = ddim.sample_from_batch(
        batch_size=config.logging.num_samples,
        channels=config.model.in_channels,
        image_size=config.model.image_size,
        num_steps=50,  # Fast sampling
        device=device,
        show_progress=True
    )

    # Denormalize to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)

    # Save as grid
    from torchvision.utils import save_image
    save_image(
        samples,
        save_dir / f"epoch_{epoch:03d}.png",
        nrow=4,
        padding=2
    )

    print(f"   Saved to {save_dir / f'epoch_{epoch:03d}.png'}")

    return samples


def main(config):
    """Main training function"""
    print("=" * 70)
    print("Training Custom Diffusion Model")
    print("=" * 70)

    # Set seed
    set_seed(config.seed)
    print(f"🌱 Seed set to {config.seed}")

    # Device
    device = get_device(config)

    # Build model
    model = build_model(config)
    model = model.to(device)

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📏 Model parameters: {num_params:,}")

    # Build data loaders
    train_loader, val_loader = build_dataloader(config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay
    )

    # Mixed precision scaler
    from torch.cuda.amp import GradScaler
    scaler = GradScaler() if config.training.mixed_precision else None

    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')

    if config.checkpoint.resume_from:
        print(f"📂 Loading checkpoint from {config.checkpoint.resume_from}")
        checkpoint = torch.load(config.checkpoint.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"   Resumed from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, config.training.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, config, epoch
        )

        # Validate
        val_loss = validate(model, val_loader, device, config)

        epoch_time = time.time() - start_time

        print(f"\n📊 Epoch {epoch+1}/{config.training.epochs}")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Time: {epoch_time:.2f}s")

        # Generate samples
        if (epoch + 1) % config.logging.visualize_interval == 0:
            generate_samples(model, config, device, epoch + 1)

        # Save checkpoint
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        if (epoch + 1) % config.checkpoint.save_interval == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': best_loss,
                'config': config
            }

            if is_best:
                save_path = checkpoint_dir / "best_model.pth"
                torch.save(checkpoint, save_path)
                print(f"💾 Saved best model (val_loss: {val_loss:.4f})")

            if (epoch + 1) % config.checkpoint.save_interval == 0:
                save_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
                torch.save(checkpoint, save_path)
                print(f"💾 Saved checkpoint: {save_path}")

    # Final save
    final_checkpoint = {
        'epoch': config.training.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_loss': best_loss,
        'config': config
    }

    final_path = checkpoint_dir / "final_model.pth"
    torch.save(final_checkpoint, final_path)
    print(f"\n💾 Saved final model: {final_path}")

    print("\n" + "=" * 70)
    print("✅ Training completed!")
    print(f"📊 Best Val Loss: {best_loss:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Custom Diffusion model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override resume path if provided
    if args.resume:
        config.checkpoint.resume_from = args.resume

    # Run training
    main(config)
