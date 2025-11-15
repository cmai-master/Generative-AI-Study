"""
모든 생성 모델 비교 스크립트

VAE, GAN, Diffusion 모델을 동일한 데이터셋에서 비교:
1. 샘플 품질 (FID, IS)
2. 생성 속도
3. 다양성 (Precision, Recall)
4. 시각적 비교
"""

import os
import sys
import argparse
import torch
import time
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_vae_model(checkpoint_path, device):
    """Load trained VAE model"""
    from vae.basic_vae import VAE

    print(f"Loading VAE from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = VAE(
        input_dim=784,
        hidden_dims=[512, 256],
        latent_dim=20
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_gan_model(checkpoint_path, model_type, device):
    """Load trained GAN model"""
    if model_type == 'dcgan':
        from gan.dcgan import Generator
        model = Generator(latent_dim=100, feature_map_size=64, num_channels=1).to(device)
    elif model_type == 'wgan-gp':
        from gan.wgan_gp import Generator
        model = Generator(latent_dim=100, feature_map_size=64, num_channels=1).to(device)
    else:
        raise ValueError(f"Unknown GAN type: {model_type}")

    print(f"Loading {model_type.upper()} from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()

    return model


def load_diffusion_model(checkpoint_path, device):
    """Load trained diffusion model"""
    from diffusion import DDPM, DDIMSampler, UNet

    print(f"Loading DDPM from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    unet = UNet(
        in_channels=1,
        out_channels=1,
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16, 8)
    )

    ddpm = DDPM(
        model=unet,
        timesteps=1000,
        schedule_type='cosine',
        objective='noise'
    ).to(device)

    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.eval()

    return ddpm


@torch.no_grad()
def generate_vae_samples(model, num_samples, device):
    """Generate samples from VAE"""
    print(f"Generating {num_samples} samples from VAE...")
    start_time = time.time()

    # Sample from prior
    z = torch.randn(num_samples, model.latent_dim).to(device)
    samples = model.decoder(z)

    # Reshape to images
    samples = samples.view(num_samples, 1, 28, 28)

    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.2f}s ({elapsed/num_samples:.4f}s per image)")

    return samples, elapsed


@torch.no_grad()
def generate_gan_samples(model, num_samples, latent_dim, device):
    """Generate samples from GAN"""
    print(f"Generating {num_samples} samples from GAN...")
    start_time = time.time()

    # Sample from latent space
    z = torch.randn(num_samples, latent_dim).to(device)
    samples = model(z)

    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.2f}s ({elapsed/num_samples:.4f}s per image)")

    return samples, elapsed


@torch.no_grad()
def generate_diffusion_samples(ddpm, num_samples, use_ddim=True, ddim_steps=50, device='cuda'):
    """Generate samples from diffusion model"""
    if use_ddim:
        from diffusion import DDIMSampler
        print(f"Generating {num_samples} samples from DDIM ({ddim_steps} steps)...")
        ddim = DDIMSampler(ddpm)

        start_time = time.time()
        samples = ddim.sample_from_batch(
            batch_size=num_samples,
            channels=1,
            image_size=28,
            num_steps=ddim_steps,
            eta=0.0,
            device=device,
            show_progress=True
        )
        elapsed = time.time() - start_time
    else:
        print(f"Generating {num_samples} samples from DDPM (1000 steps)...")
        start_time = time.time()
        samples = ddpm.sample(
            batch_size=num_samples,
            channels=1,
            image_size=28,
            device=device,
            show_progress=True
        )
        elapsed = time.time() - start_time

    print(f"  Time: {elapsed:.2f}s ({elapsed/num_samples:.4f}s per image)")

    return samples, elapsed


def calculate_metrics(real_images, fake_images, device):
    """Calculate FID and IS"""
    from evaluation.metrics import calculate_fid, calculate_inception_score

    print("Calculating FID...")
    fid = calculate_fid(real_images, fake_images, batch_size=50, device=device)

    print("Calculating Inception Score...")
    is_mean, is_std = calculate_inception_score(fake_images, batch_size=50, device=device)

    return {
        'fid': fid,
        'is_mean': is_mean,
        'is_std': is_std
    }


def compare_models(vae_checkpoint, dcgan_checkpoint, wgan_checkpoint, ddpm_checkpoint,
                   real_data_loader, results_dir, device):
    """Compare all models"""

    print("\n" + "="*70)
    print("Loading Models")
    print("="*70)

    # Load models
    vae = load_vae_model(vae_checkpoint, device) if vae_checkpoint else None
    dcgan = load_gan_model(dcgan_checkpoint, 'dcgan', device) if dcgan_checkpoint else None
    wgan = load_gan_model(wgan_checkpoint, 'wgan-gp', device) if wgan_checkpoint else None
    ddpm = load_diffusion_model(ddpm_checkpoint, device) if ddpm_checkpoint else None

    # Generate samples
    num_samples = 64
    results = {}

    print("\n" + "="*70)
    print("Generating Samples")
    print("="*70)

    if vae:
        vae_samples, vae_time = generate_vae_samples(vae, num_samples, device)
        results['VAE'] = {
            'samples': vae_samples,
            'generation_time': vae_time,
            'time_per_image': vae_time / num_samples
        }

    if dcgan:
        dcgan_samples, dcgan_time = generate_gan_samples(dcgan, num_samples, 100, device)
        results['DCGAN'] = {
            'samples': dcgan_samples,
            'generation_time': dcgan_time,
            'time_per_image': dcgan_time / num_samples
        }

    if wgan:
        wgan_samples, wgan_time = generate_gan_samples(wgan, num_samples, 100, device)
        results['WGAN-GP'] = {
            'samples': wgan_samples,
            'generation_time': wgan_time,
            'time_per_image': wgan_time / num_samples
        }

    if ddpm:
        # DDPM (slow)
        ddpm_samples, ddpm_time = generate_diffusion_samples(ddpm, num_samples, use_ddim=False, device=device)
        results['DDPM'] = {
            'samples': ddpm_samples,
            'generation_time': ddpm_time,
            'time_per_image': ddpm_time / num_samples
        }

        # DDIM (fast)
        ddim_samples, ddim_time = generate_diffusion_samples(ddpm, num_samples, use_ddim=True, ddim_steps=50, device=device)
        results['DDIM-50'] = {
            'samples': ddim_samples,
            'generation_time': ddim_time,
            'time_per_image': ddim_time / num_samples
        }

    # Save visual comparisons
    print("\n" + "="*70)
    print("Saving Visual Comparisons")
    print("="*70)

    os.makedirs(results_dir, exist_ok=True)

    for model_name, model_results in results.items():
        save_path = os.path.join(results_dir, f'{model_name.lower().replace("-", "_")}_samples.png')
        save_image(
            model_results['samples'][:64],
            save_path,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )
        print(f"  Saved {model_name} samples to {save_path}")

    # Get real images for FID calculation
    print("\n" + "="*70)
    print("Calculating Metrics")
    print("="*70)

    real_images = []
    for images, _ in real_data_loader:
        real_images.append(images)
        if len(real_images) * images.size(0) >= 1000:
            break
    real_images = torch.cat(real_images, dim=0)[:1000].to(device)

    # Calculate metrics for each model
    metrics_results = {}

    for model_name, model_results in results.items():
        print(f"\nCalculating metrics for {model_name}...")
        fake_images = model_results['samples']

        # Repeat to get 1000 samples if needed
        if fake_images.size(0) < 1000:
            num_repeats = (1000 // fake_images.size(0)) + 1
            fake_images = fake_images.repeat(num_repeats, 1, 1, 1)[:1000]

        metrics = calculate_metrics(real_images, fake_images, device)

        metrics_results[model_name] = {
            'fid': metrics['fid'],
            'is_mean': metrics['is_mean'],
            'is_std': metrics['is_std'],
            'generation_time': model_results['generation_time'],
            'time_per_image': model_results['time_per_image']
        }

        print(f"  FID: {metrics['fid']:.2f}")
        print(f"  IS: {metrics['is_mean']:.2f} ± {metrics['is_std']:.2f}")
        print(f"  Time per image: {model_results['time_per_image']:.4f}s")

    # Save results
    results_file = os.path.join(results_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(metrics_results, f, indent=2)

    print(f"\n✅ Results saved to {results_file}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'FID':<10} {'IS':<15} {'Time/img (s)':<15}")
    print("-"*70)

    for model_name, metrics in metrics_results.items():
        print(f"{model_name:<15} {metrics['fid']:<10.2f} "
              f"{metrics['is_mean']:.2f}±{metrics['is_std']:.2f}    "
              f"{metrics['time_per_image']:<15.4f}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare all generative models')
    parser.add_argument('--vae-checkpoint', type=str, default=None,
                        help='Path to VAE checkpoint')
    parser.add_argument('--dcgan-checkpoint', type=str, default=None,
                        help='Path to DCGAN checkpoint')
    parser.add_argument('--wgan-checkpoint', type=str, default=None,
                        help='Path to WGAN-GP checkpoint')
    parser.add_argument('--ddpm-checkpoint', type=str, default=None,
                        help='Path to DDPM checkpoint')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--results-dir', type=str, default='./results/model_comparison',
                        help='Results directory')

    args = parser.parse_args()

    # Check that at least one checkpoint is provided
    if not any([args.vae_checkpoint, args.dcgan_checkpoint,
                args.wgan_checkpoint, args.ddpm_checkpoint]):
        print("❌ Error: Please provide at least one model checkpoint")
        return

    # Load dataset
    from utils.datasets import get_mnist, get_cifar10

    if args.dataset == 'mnist':
        _, test_loader = get_mnist(batch_size=128)
    elif args.dataset == 'cifar10':
        _, test_loader = get_cifar10(batch_size=128)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Run comparison
    compare_models(
        args.vae_checkpoint,
        args.dcgan_checkpoint,
        args.wgan_checkpoint,
        args.ddpm_checkpoint,
        test_loader,
        args.results_dir,
        args.device
    )


if __name__ == '__main__':
    main()
