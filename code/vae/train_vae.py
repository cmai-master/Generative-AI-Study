"""
VAE 학습 스크립트

MNIST 데이터셋으로 VAE와 β-VAE를 학습하는 완전한 예제 코드입니다.

사용법:
    python train_vae.py --model vae --epochs 50
    python train_vae.py --model beta_vae --beta 4.0 --epochs 50
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from vae.basic_vae import VAE, train_epoch, evaluate
from vae.beta_vae import BetaVAE, train_epoch_beta_vae, evaluate_beta_vae
from utils import (
    get_dataloader,
    CheckpointManager,
    EarlyStopping,
    TrainingLogger,
    get_device,
    set_seed,
    print_model_summary,
    plot_training_curves,
    visualize_reconstruction,
    visualize_samples,
    visualize_latent_space_2d
)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Train VAE on MNIST')

    # 모델 설정
    parser.add_argument('--model', type=str, default='vae',
                        choices=['vae', 'beta_vae'],
                        help='모델 타입 (기본값: vae)')
    parser.add_argument('--latent-dim', type=int, default=20,
                        help='잠재 공간 차원 (기본값: 20)')
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[512, 256],
                        help='히든 레이어 차원 (기본값: 512 256)')
    parser.add_argument('--beta', type=float, default=4.0,
                        help='β-VAE의 β 값 (기본값: 4.0)')

    # 학습 설정
    parser.add_argument('--epochs', type=int, default=50,
                        help='학습 에포크 수 (기본값: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='배치 크기 (기본값: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='학습률 (기본값: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (기본값: 42)')

    # 데이터 설정
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='데이터 디렉토리 (기본값: ./data)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='데이터 로더 워커 수 (기본값: 4)')

    # 체크포인트 및 로깅
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='체크포인트 디렉토리 (기본값: ./checkpoints)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='로그 디렉토리 (기본값: ./logs)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='체크포인트 저장 간격 (기본값: 10)')

    # Early Stopping
    parser.add_argument('--early-stop', action='store_true',
                        help='Early Stopping 사용')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early Stopping patience (기본값: 10)')

    # 시각화
    parser.add_argument('--visualize', action='store_true',
                        help='학습 후 시각화 수행')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='시각화 결과 저장 디렉토리 (기본값: ./outputs)')

    return parser.parse_args()


def create_model(args, device):
    """모델 생성"""
    if args.model == 'vae':
        model = VAE(
            input_dim=784,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim
        )
    elif args.model == 'beta_vae':
        model = BetaVAE(
            input_dim=784,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim,
            beta=args.beta
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model.to(device)


def main():
    """메인 학습 함수"""
    # 인자 파싱
    args = parse_args()

    # 시드 설정
    set_seed(args.seed)

    # 디바이스 설정
    device = get_device()
    print(f"\nUsing device: {device}\n")

    # 데이터 로더
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_dataloader(
        'mnist',
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 모델 생성
    print(f"\nCreating {args.model.upper()} model...")
    model = create_model(args, device)
    print_model_summary(model, input_size=(1, 28, 28))

    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 체크포인트 관리자
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints=3
    )

    # 로거
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=f"{args.model}_latent{args.latent_dim}"
    )

    # Early Stopping
    early_stopping = None
    if args.early_stop:
        early_stopping = EarlyStopping(patience=args.patience)

    # 학습 히스토리
    train_losses = []
    test_losses = []
    train_recons = []
    test_recons = []
    train_kls = []
    test_kls = []

    # 학습 루프
    print(f"\n{'='*70}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'='*70}\n")

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # 학습
        if args.model == 'vae':
            train_loss, train_recon, train_kl = train_epoch(
                model, train_loader, optimizer, device, beta=1.0
            )
            test_loss, test_recon, test_kl = evaluate(
                model, test_loader, device, beta=1.0
            )
        else:  # beta_vae
            train_loss, train_recon, train_kl = train_epoch_beta_vae(
                model, train_loader, optimizer, device
            )
            test_loss, test_recon, test_kl = evaluate_beta_vae(
                model, test_loader, device
            )

        # 히스토리 저장
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_recons.append(train_recon)
        test_recons.append(test_recon)
        train_kls.append(train_kl)
        test_kls.append(test_kl)

        # 로그 기록
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=test_loss,
            metrics={
                'train_recon': train_recon,
                'train_kl': train_kl,
                'test_recon': test_recon,
                'test_kl': test_kl
            }
        )

        # 출력
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | "
              f"Test Loss: {test_loss:.4f} (Recon: {test_recon:.4f}, KL: {test_kl:.4f})")

        # 최고 모델 저장
        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint_manager.save_best(
                model, optimizer, epoch, test_loss,
                metrics={'recon': test_recon, 'kl': test_kl}
            )
            print(f"  → Best model saved! (Test Loss: {test_loss:.4f})")

        # 주기적 체크포인트 저장
        if epoch % args.save_interval == 0:
            checkpoint_manager.save(
                model, optimizer, epoch, test_loss,
                metrics={'recon': test_recon, 'kl': test_kl}
            )

        # Early Stopping 체크
        if early_stopping is not None:
            if early_stopping(test_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    # 로그 저장
    logger.save()
    print(f"\nTraining completed! Logs saved to {logger.log_file}")

    # 학습 곡선 시각화
    if args.visualize:
        print("\nGenerating visualizations...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 학습 곡선
        plot_training_curves(
            train_losses, test_losses,
            title=f'{args.model.upper()} Training Curves',
            save_path=output_dir / 'training_curves.png'
        )

        # 2. 재구성 시각화
        test_images, _ = next(iter(test_loader))
        visualize_reconstruction(
            model, test_images, device,
            n_samples=8,
            title='Reconstruction Comparison',
            save_path=output_dir / 'reconstruction.png'
        )

        # 3. 샘플 생성
        visualize_samples(
            model,
            n_samples=64,
            latent_dim=args.latent_dim,
            device=device,
            title='Generated Samples',
            save_path=output_dir / 'samples.png'
        )

        # 4. Latent Space 시각화 (2D인 경우)
        if args.latent_dim == 2:
            # 테스트 데이터 인코딩
            all_latents = []
            all_labels = []

            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    mu, _ = model.encode(images)
                    all_latents.append(mu.cpu())
                    all_labels.append(labels)

            all_latents = torch.cat(all_latents, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            visualize_latent_space_2d(
                all_latents, all_labels,
                title='2D Latent Space',
                save_path=output_dir / 'latent_space.png'
            )

        print(f"Visualizations saved to {output_dir}")

    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
