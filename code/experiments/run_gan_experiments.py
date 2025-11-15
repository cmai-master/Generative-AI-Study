"""
GAN (DCGAN, WGAN-GP) 실험 자동화 스크립트

다음 실험을 자동으로 실행합니다:
1. DCGAN vs WGAN-GP 비교
2. Learning rate ablation (Generator vs Discriminator)
3. Architecture ablation (feature map size)
4. Training stability 분석
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def run_experiment(name, script, args, results_dir):
    """
    실험 실행 헬퍼 함수

    Args:
        name: 실험 이름
        script: 실행할 스크립트 경로
        args: 스크립트 인자
        results_dir: 결과 저장 디렉토리
    """
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}")

    # 결과 디렉토리 생성
    exp_dir = os.path.join(results_dir, name.replace(' ', '_').lower())
    os.makedirs(exp_dir, exist_ok=True)

    # 실험 실행
    cmd = ['python', script] + args + ['--save-dir', exp_dir]
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {name} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} failed with error: {e}")
        return False

    return True


def experiment_1_dcgan_vs_wgan(gan_script, results_dir, dataset='mnist', epochs=100, device='cuda'):
    """
    실험 1: DCGAN vs WGAN-GP 비교
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: DCGAN vs WGAN-GP Comparison")
    print("="*70)

    models = ['dcgan', 'wgan-gp']

    for model in models:
        name = f"{model}_baseline"
        args = [
            '--model', model,
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--device', device,
            '--latent-dim', '100',
            '--batch-size', '128',
            '--lr', '2e-4',
            '--feature-map-size', '64',
            '--save-interval', '10',
            '--visualize-interval', '5',
        ]

        # WGAN-GP specific arguments
        if model == 'wgan-gp':
            args.extend([
                '--n-critic', '5',
                '--lambda-gp', '10',
            ])

        run_experiment(name, gan_script, args, results_dir)


def experiment_2_learning_rate_ablation(gan_script, results_dir, dataset='mnist', epochs=100, device='cuda'):
    """
    실험 2: Learning rate ablation (DCGAN)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Learning Rate Ablation")
    print("="*70)

    learning_rates = [1e-4, 2e-4, 5e-4]

    for lr in learning_rates:
        name = f"dcgan_lr_{lr:.0e}"
        args = [
            '--model', 'dcgan',
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--device', device,
            '--latent-dim', '100',
            '--batch-size', '128',
            '--lr', str(lr),
            '--feature-map-size', '64',
            '--save-interval', '10',
            '--visualize-interval', '5',
        ]

        run_experiment(name, gan_script, args, results_dir)


def experiment_3_architecture_ablation(gan_script, results_dir, dataset='mnist', epochs=100, device='cuda'):
    """
    실험 3: Architecture ablation (feature map size)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Architecture Ablation")
    print("="*70)

    feature_map_sizes = [32, 64, 128]

    for fms in feature_map_sizes:
        name = f"dcgan_fms_{fms}"
        args = [
            '--model', 'dcgan',
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--device', device,
            '--latent-dim', '100',
            '--batch-size', '128',
            '--lr', '2e-4',
            '--feature-map-size', str(fms),
            '--save-interval', '10',
            '--visualize-interval', '5',
        ]

        run_experiment(name, gan_script, args, results_dir)


def experiment_4_wgan_gp_lambda_ablation(gan_script, results_dir, dataset='mnist', epochs=100, device='cuda'):
    """
    실험 4: WGAN-GP gradient penalty weight ablation
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: WGAN-GP Lambda GP Ablation")
    print("="*70)

    lambda_gps = [1, 10, 50]

    for lambda_gp in lambda_gps:
        name = f"wgan_gp_lambda_{lambda_gp}"
        args = [
            '--model', 'wgan-gp',
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--device', device,
            '--latent-dim', '100',
            '--batch-size', '128',
            '--lr', '2e-4',
            '--feature-map-size', '64',
            '--n-critic', '5',
            '--lambda-gp', str(lambda_gp),
            '--save-interval', '10',
            '--visualize-interval', '5',
        ]

        run_experiment(name, gan_script, args, results_dir)


def experiment_5_latent_dim_ablation(gan_script, results_dir, dataset='mnist', epochs=100, device='cuda'):
    """
    실험 5: Latent dimension ablation
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Latent Dimension Ablation")
    print("="*70)

    latent_dims = [50, 100, 200]

    for latent_dim in latent_dims:
        name = f"wgan_gp_latent_{latent_dim}"
        args = [
            '--model', 'wgan-gp',
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--device', device,
            '--latent-dim', str(latent_dim),
            '--batch-size', '128',
            '--lr', '2e-4',
            '--feature-map-size', '64',
            '--n-critic', '5',
            '--lambda-gp', '10',
            '--save-interval', '10',
            '--visualize-interval', '5',
        ]

        run_experiment(name, gan_script, args, results_dir)


def main():
    parser = argparse.ArgumentParser(description='Run GAN experiments')
    parser.add_argument('--experiments', nargs='+', type=int,
                        default=[1, 2, 3, 4, 5],
                        help='Which experiments to run (1-5)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'cifar10', 'celeba'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--results-dir', type=str, default='./results/gan_experiments',
                        help='Results directory')

    args = parser.parse_args()

    # GAN 학습 스크립트 경로
    gan_script = os.path.join(Path(__file__).parent.parent, 'gan', 'train_gan.py')

    if not os.path.exists(gan_script):
        print(f"❌ GAN training script not found: {gan_script}")
        return

    print(f"\n{'='*70}")
    print(f"GAN Experiments Configuration")
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
        experiment_1_dcgan_vs_wgan(gan_script, args.results_dir, args.dataset, args.epochs, args.device)

    if 2 in args.experiments:
        experiment_2_learning_rate_ablation(gan_script, args.results_dir, args.dataset, args.epochs, args.device)

    if 3 in args.experiments:
        experiment_3_architecture_ablation(gan_script, args.results_dir, args.dataset, args.epochs, args.device)

    if 4 in args.experiments:
        experiment_4_wgan_gp_lambda_ablation(gan_script, args.results_dir, args.dataset, args.epochs, args.device)

    if 5 in args.experiments:
        experiment_5_latent_dim_ablation(gan_script, args.results_dir, args.dataset, args.epochs, args.device)

    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Results saved to: {args.results_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
