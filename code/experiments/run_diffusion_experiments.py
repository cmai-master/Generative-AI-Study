"""
Diffusion Models (DDPM, DDIM) 실험 자동화 스크립트

다음 실험을 자동으로 실행합니다:
1. DDPM baseline 학습
2. Noise schedule 비교 (linear, cosine, sigmoid)
3. DDIM 샘플링 step 수 비교 (10, 20, 50, 100, 500)
4. DDIM eta 비교 (deterministic vs stochastic)
5. U-Net architecture ablation
"""

import os
import sys
import argparse
import subprocess
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def run_training(name, args_dict, results_dir):
    """
    DDPM 학습 실행

    Args:
        name: 실험 이름
        args_dict: 학습 인자 딕셔너리
        results_dir: 결과 저장 디렉토리
    """
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print(f"{'='*70}")

    # 결과 디렉토리 생성
    exp_dir = os.path.join(results_dir, name.replace(' ', '_').lower())
    os.makedirs(exp_dir, exist_ok=True)
    args_dict['save_dir'] = exp_dir

    # Import training function
    from diffusion import DDPM, UNet
    from utils.datasets import get_mnist, get_cifar10
    from utils.training import CheckpointManager, TrainingLogger

    # Load dataset
    dataset = args_dict.get('dataset', 'mnist')
    batch_size = args_dict.get('batch_size', 128)
    device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading {dataset} dataset...")
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist(batch_size=batch_size)
        in_channels = 1
        image_size = 28
    elif dataset == 'cifar10':
        train_loader, test_loader = get_cifar10(batch_size=batch_size)
        in_channels = 3
        image_size = 32
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create U-Net
    print("Creating U-Net...")
    unet = UNet(
        in_channels=in_channels,
        out_channels=in_channels,
        model_channels=args_dict.get('model_channels', 128),
        channel_mult=args_dict.get('channel_mult', (1, 2, 2, 2)),
        num_res_blocks=args_dict.get('num_res_blocks', 2),
        attention_resolutions=args_dict.get('attention_resolutions', (16, 8)),
        num_heads=args_dict.get('num_heads', 4)
    )

    # Create DDPM
    print("Creating DDPM...")
    ddpm = DDPM(
        model=unet,
        timesteps=args_dict.get('timesteps', 1000),
        schedule_type=args_dict.get('schedule_type', 'cosine'),
        objective=args_dict.get('objective', 'noise')
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in ddpm.parameters()):,}")

    # Setup training
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=args_dict.get('lr', 1e-4))
    checkpoint_manager = CheckpointManager(exp_dir)
    logger = TrainingLogger(exp_dir)

    epochs = args_dict.get('epochs', 50)
    save_interval = args_dict.get('save_interval', 10)

    print(f"\nStarting training for {epochs} epochs...")

    # Training loop
    for epoch in range(1, epochs + 1):
        ddpm.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, _ in train_loader:
            images = images.to(device)

            # Forward pass
            loss = ddpm(images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Log
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
        logger.log({'epoch': epoch, 'loss': avg_loss})

        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint_manager.save(
                ddpm, optimizer, epoch, avg_loss,
                filename=f'checkpoint_epoch_{epoch}.pt'
            )

            # Generate samples
            print("Generating samples...")
            ddpm.eval()
            with torch.no_grad():
                samples = ddpm.sample(
                    batch_size=16,
                    channels=in_channels,
                    image_size=image_size,
                    device=device,
                    show_progress=False
                )

            # Save samples
            from torchvision.utils import save_image
            save_image(
                samples,
                os.path.join(exp_dir, f'samples_epoch_{epoch}.png'),
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )

    print(f"✅ Training completed! Results saved to {exp_dir}")
    return exp_dir


def experiment_1_ddpm_baseline(results_dir, dataset='mnist', epochs=50, device='cuda'):
    """
    실험 1: DDPM baseline
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: DDPM Baseline")
    print("="*70)

    args_dict = {
        'dataset': dataset,
        'epochs': epochs,
        'device': device,
        'batch_size': 128,
        'lr': 1e-4,
        'timesteps': 1000,
        'schedule_type': 'cosine',
        'objective': 'noise',
        'model_channels': 128,
        'channel_mult': (1, 2, 2, 2),
        'num_res_blocks': 2,
        'attention_resolutions': (16, 8),
        'save_interval': 10,
    }

    run_training('ddpm_baseline', args_dict, results_dir)


def experiment_2_noise_schedule_comparison(results_dir, dataset='mnist', epochs=50, device='cuda'):
    """
    실험 2: Noise schedule 비교
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Noise Schedule Comparison")
    print("="*70)

    schedules = ['linear', 'cosine', 'sigmoid']

    for schedule in schedules:
        args_dict = {
            'dataset': dataset,
            'epochs': epochs,
            'device': device,
            'batch_size': 128,
            'lr': 1e-4,
            'timesteps': 1000,
            'schedule_type': schedule,
            'objective': 'noise',
            'model_channels': 128,
            'channel_mult': (1, 2, 2, 2),
            'num_res_blocks': 2,
            'attention_resolutions': (16, 8),
            'save_interval': 10,
        }

        run_training(f'ddpm_schedule_{schedule}', args_dict, results_dir)


def experiment_3_ddim_steps_comparison(ddpm_checkpoint, results_dir, device='cuda'):
    """
    실험 3: DDIM 샘플링 스텝 수 비교
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: DDIM Sampling Steps Comparison")
    print("="*70)

    from diffusion import DDPM, DDIMSampler, UNet
    import time
    from torchvision.utils import save_image

    # Load trained DDPM
    print(f"Loading DDPM from {ddpm_checkpoint}...")
    checkpoint = torch.load(ddpm_checkpoint)

    # Create model (assume MNIST for now)
    unet = UNet(in_channels=1, out_channels=1, model_channels=128,
                channel_mult=(1, 2, 2, 2), num_res_blocks=2,
                attention_resolutions=(16, 8))
    ddpm = DDPM(model=unet, timesteps=1000, schedule_type='cosine', objective='noise').to(device)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.eval()

    # Create DDIM sampler
    ddim = DDIMSampler(ddpm)

    # Test different step counts
    step_counts = [10, 20, 50, 100, 500]
    exp_dir = os.path.join(results_dir, 'ddim_steps_comparison')
    os.makedirs(exp_dir, exist_ok=True)

    results = []

    for num_steps in step_counts:
        print(f"\nTesting DDIM with {num_steps} steps...")

        # Measure time
        start_time = time.time()
        samples = ddim.sample_from_batch(
            batch_size=16,
            channels=1,
            image_size=28,
            num_steps=num_steps,
            eta=0.0,  # deterministic
            device=device,
            show_progress=True
        )
        elapsed = time.time() - start_time

        # Save samples
        save_image(
            samples,
            os.path.join(exp_dir, f'ddim_{num_steps}_steps.png'),
            nrow=4,
            normalize=True,
            value_range=(-1, 1)
        )

        results.append({
            'steps': num_steps,
            'time': elapsed,
            'time_per_image': elapsed / 16
        })

        print(f"  Time: {elapsed:.2f}s ({elapsed/16:.3f}s per image)")

    # Save results
    import json
    with open(os.path.join(exp_dir, 'timing_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ DDIM steps comparison completed! Results saved to {exp_dir}")


def experiment_4_ddim_eta_comparison(ddpm_checkpoint, results_dir, device='cuda'):
    """
    실험 4: DDIM eta 비교 (deterministic vs stochastic)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: DDIM Eta Comparison")
    print("="*70)

    from diffusion import DDPM, DDIMSampler, UNet
    from torchvision.utils import save_image

    # Load trained DDPM
    print(f"Loading DDPM from {ddpm_checkpoint}...")
    checkpoint = torch.load(ddpm_checkpoint)

    unet = UNet(in_channels=1, out_channels=1, model_channels=128,
                channel_mult=(1, 2, 2, 2), num_res_blocks=2,
                attention_resolutions=(16, 8))
    ddpm = DDPM(model=unet, timesteps=1000, schedule_type='cosine', objective='noise').to(device)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.eval()

    ddim = DDIMSampler(ddpm)

    # Test different eta values
    eta_values = [0.0, 0.3, 0.5, 1.0]
    exp_dir = os.path.join(results_dir, 'ddim_eta_comparison')
    os.makedirs(exp_dir, exist_ok=True)

    for eta in eta_values:
        print(f"\nTesting DDIM with eta={eta}...")

        samples = ddim.sample_from_batch(
            batch_size=16,
            channels=1,
            image_size=28,
            num_steps=50,
            eta=eta,
            device=device,
            show_progress=True
        )

        save_image(
            samples,
            os.path.join(exp_dir, f'ddim_eta_{eta}.png'),
            nrow=4,
            normalize=True,
            value_range=(-1, 1)
        )

    print(f"✅ DDIM eta comparison completed! Results saved to {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run Diffusion Model experiments')
    parser.add_argument('--experiments', nargs='+', type=int,
                        default=[1, 2],
                        help='Which experiments to run (1-4)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--results-dir', type=str, default='./results/diffusion_experiments',
                        help='Results directory')
    parser.add_argument('--ddpm-checkpoint', type=str, default=None,
                        help='DDPM checkpoint for sampling experiments (exp 3, 4)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Diffusion Model Experiments Configuration")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Results directory: {args.results_dir}")
    print(f"Experiments to run: {args.experiments}")

    # 결과 디렉토리 생성
    os.makedirs(args.results_dir, exist_ok=True)

    # 실험 실행
    if 1 in args.experiments:
        experiment_1_ddpm_baseline(args.results_dir, args.dataset, args.epochs, args.device)

    if 2 in args.experiments:
        experiment_2_noise_schedule_comparison(args.results_dir, args.dataset, args.epochs, args.device)

    if 3 in args.experiments:
        if args.ddpm_checkpoint is None:
            print("⚠️  Skipping experiment 3: --ddpm-checkpoint not provided")
        else:
            experiment_3_ddim_steps_comparison(args.ddpm_checkpoint, args.results_dir, args.device)

    if 4 in args.experiments:
        if args.ddpm_checkpoint is None:
            print("⚠️  Skipping experiment 4: --ddpm-checkpoint not provided")
        else:
            experiment_4_ddim_eta_comparison(args.ddpm_checkpoint, args.results_dir, args.device)

    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Results saved to: {args.results_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
