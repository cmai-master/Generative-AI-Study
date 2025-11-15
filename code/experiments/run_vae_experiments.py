"""
VAE 및 β-VAE 실험 자동화 스크립트

다음 실험을 자동으로 실행합니다:
1. 기본 VAE vs β-VAE 비교 (β=1, 4, 10)
2. Latent dimension ablation (10, 20, 50, 100)
3. Learning rate ablation (1e-5, 1e-4, 1e-3)
4. Hidden dimension ablation
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


def experiment_1_beta_comparison(vae_script, results_dir, dataset='mnist', epochs=50, device='cuda'):
    """
    실험 1: β-VAE β 값 비교 (β=0.5, 1.0, 4.0, 10.0)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: β-VAE Beta Comparison")
    print("="*70)

    beta_values = [0.5, 1.0, 4.0, 10.0]

    for beta in beta_values:
        name = f"beta_vae_beta_{beta}"
        args = [
            '--model', 'beta-vae',
            '--beta', str(beta),
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--device', device,
            '--latent-dim', '20',
            '--batch-size', '128',
            '--lr', '1e-4',
            '--save-interval', '10',
            '--visualize-interval', '5',
        ]

        run_experiment(name, vae_script, args, results_dir)


def experiment_2_latent_dim_ablation(vae_script, results_dir, dataset='mnist', epochs=50, device='cuda'):
    """
    실험 2: Latent dimension ablation study
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Latent Dimension Ablation")
    print("="*70)

    latent_dims = [10, 20, 50, 100]

    for latent_dim in latent_dims:
        name = f"vae_latent_{latent_dim}"
        args = [
            '--model', 'vae',
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--device', device,
            '--latent-dim', str(latent_dim),
            '--batch-size', '128',
            '--lr', '1e-4',
            '--save-interval', '10',
            '--visualize-interval', '5',
        ]

        run_experiment(name, vae_script, args, results_dir)


def experiment_3_learning_rate_ablation(vae_script, results_dir, dataset='mnist', epochs=50, device='cuda'):
    """
    실험 3: Learning rate ablation study
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Learning Rate Ablation")
    print("="*70)

    learning_rates = [1e-5, 1e-4, 1e-3]

    for lr in learning_rates:
        name = f"vae_lr_{lr:.0e}"
        args = [
            '--model', 'vae',
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--device', device,
            '--latent-dim', '20',
            '--batch-size', '128',
            '--lr', str(lr),
            '--save-interval', '10',
            '--visualize-interval', '5',
        ]

        run_experiment(name, vae_script, args, results_dir)


def experiment_4_hidden_dims_ablation(vae_script, results_dir, dataset='mnist', epochs=50, device='cuda'):
    """
    실험 4: Hidden dimensions ablation study
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Hidden Dimensions Ablation")
    print("="*70)

    hidden_configs = [
        ([256, 128], "256_128"),
        ([512, 256], "512_256"),
        ([512, 256, 128], "512_256_128"),
    ]

    for hidden_dims, name_suffix in hidden_configs:
        name = f"vae_hidden_{name_suffix}"
        args = [
            '--model', 'vae',
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--device', device,
            '--latent-dim', '20',
            '--hidden-dims', ' '.join(map(str, hidden_dims)),
            '--batch-size', '128',
            '--lr', '1e-4',
            '--save-interval', '10',
            '--visualize-interval', '5',
        ]

        run_experiment(name, vae_script, args, results_dir)


def main():
    parser = argparse.ArgumentParser(description='Run VAE/β-VAE experiments')
    parser.add_argument('--experiments', nargs='+', type=int,
                        default=[1, 2, 3, 4],
                        help='Which experiments to run (1-4)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--results-dir', type=str, default='./results/vae_experiments',
                        help='Results directory')

    args = parser.parse_args()

    # VAE 학습 스크립트 경로
    vae_script = os.path.join(Path(__file__).parent.parent, 'vae', 'train_vae.py')

    if not os.path.exists(vae_script):
        print(f"❌ VAE training script not found: {vae_script}")
        return

    print(f"\n{'='*70}")
    print(f"VAE Experiments Configuration")
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
        experiment_1_beta_comparison(vae_script, args.results_dir, args.dataset, args.epochs, args.device)

    if 2 in args.experiments:
        experiment_2_latent_dim_ablation(vae_script, args.results_dir, args.dataset, args.epochs, args.device)

    if 3 in args.experiments:
        experiment_3_learning_rate_ablation(vae_script, args.results_dir, args.dataset, args.epochs, args.device)

    if 4 in args.experiments:
        experiment_4_hidden_dims_ablation(vae_script, args.results_dir, args.dataset, args.epochs, args.device)

    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Results saved to: {args.results_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
