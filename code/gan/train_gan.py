"""
GAN 학습 스크립트

DCGAN과 WGAN-GP를 학습하는 완전한 예제 코드입니다.

사용법:
    python train_gan.py --model dcgan --epochs 100
    python train_gan.py --model wgan_gp --epochs 100 --n-critic 5
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

from gan.dcgan import DCGAN, train_step_dcgan
from gan.wgan_gp import WGAN_GP, train_step_wgan_gp
from utils import (
    get_dataloader,
    CheckpointManager,
    TrainingLogger,
    get_device,
    set_seed,
    print_model_summary,
    plot_training_curves,
    show_image_grid,
    AverageMeter
)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Train GAN')

    # 모델 설정
    parser.add_argument('--model', type=str, default='dcgan',
                        choices=['dcgan', 'wgan_gp'],
                        help='모델 타입 (기본값: dcgan)')
    parser.add_argument('--latent-dim', type=int, default=100,
                        help='잠재 벡터 차원 (기본값: 100)')
    parser.add_argument('--feature-map-size', type=int, default=64,
                        help='Feature map 크기 (기본값: 64)')

    # WGAN-GP 전용
    parser.add_argument('--lambda-gp', type=float, default=10.0,
                        help='Gradient Penalty 가중치 (기본값: 10.0)')
    parser.add_argument('--n-critic', type=int, default=5,
                        help='Generator 1회당 Critic 학습 횟수 (기본값: 5)')

    # 학습 설정
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에포크 수 (기본값: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='배치 크기 (기본값: 128)')
    parser.add_argument('--lr-g', type=float, default=0.0002,
                        help='Generator 학습률 (기본값: 0.0002)')
    parser.add_argument('--lr-d', type=float, default=0.0002,
                        help='Discriminator 학습률 (기본값: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1 (기본값: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2 (기본값: 0.999)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (기본값: 42)')

    # 데이터 설정
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist', 'cifar10'],
                        help='데이터셋 (기본값: mnist)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='데이터 디렉토리 (기본값: ./data)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='데이터 로더 워커 수 (기본값: 4)')

    # 체크포인트 및 로깅
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='체크포인트 디렉토리')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='로그 디렉토리')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='출력 디렉토리')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='체크포인트 저장 간격 (기본값: 10)')
    parser.add_argument('--sample-interval', type=int, default=5,
                        help='샘플 생성 간격 (기본값: 5)')

    return parser.parse_args()


def create_model(args, device):
    """모델 생성"""
    # 채널 수 결정
    if args.dataset == 'cifar10':
        num_channels = 3
    else:
        num_channels = 1

    if args.model == 'dcgan':
        model = DCGAN(
            latent_dim=args.latent_dim,
            num_channels=num_channels,
            feature_map_size=args.feature_map_size,
            device=device
        )
        return model

    elif args.model == 'wgan_gp':
        model = WGAN_GP(
            latent_dim=args.latent_dim,
            num_channels=num_channels,
            feature_map_size=args.feature_map_size,
            lambda_gp=args.lambda_gp,
            n_critic=args.n_critic,
            device=device
        )
        return model

    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    """메인 학습 함수"""
    args = parse_args()

    # 시드 설정
    set_seed(args.seed)

    # 디바이스 설정
    device = get_device()
    print(f"\nUsing device: {device}\n")

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로더
    print(f"Loading {args.dataset.upper()} dataset...")
    train_loader, _ = get_dataloader(
        args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train batches: {len(train_loader)}")

    # 모델 생성
    print(f"\nCreating {args.model.upper()} model...")
    model = create_model(args, device)

    print(f"Generator Parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")

    # Optimizer
    if args.model == 'wgan_gp':
        # WGAN-GP는 beta1=0.0 권장
        optimizer_g = optim.Adam(
            model.generator.parameters(),
            lr=args.lr_g,
            betas=(0.0, 0.9)
        )
        optimizer_d = optim.Adam(
            model.discriminator.parameters(),
            lr=args.lr_d,
            betas=(0.0, 0.9)
        )
    else:
        optimizer_g = optim.Adam(
            model.generator.parameters(),
            lr=args.lr_g,
            betas=(args.beta1, args.beta2)
        )
        optimizer_d = optim.Adam(
            model.discriminator.parameters(),
            lr=args.lr_d,
            betas=(args.beta1, args.beta2)
        )

    # 체크포인트 관리자
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints=3
    )

    # 로거
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=f"{args.model}_{args.dataset}"
    )

    # 학습 히스토리
    d_losses = []
    g_losses = []

    # 고정된 noise (샘플 생성용)
    fixed_noise = model.sample_noise(64)

    # 학습 루프
    print(f"\n{'='*70}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        model.generator.train()
        model.discriminator.train()

        d_loss_meter = AverageMeter()
        g_loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch_idx, (real_images, _) in enumerate(pbar):
            # 학습 스텝
            if args.model == 'dcgan':
                d_loss, g_loss = train_step_dcgan(
                    model.generator,
                    model.discriminator,
                    real_images,
                    optimizer_g,
                    optimizer_d,
                    model.latent_dim,
                    device
                )
                d_loss_meter.update(d_loss)
                g_loss_meter.update(g_loss)

            elif args.model == 'wgan_gp':
                # Critic을 n_critic번 학습
                update_g = (batch_idx % args.n_critic == 0)

                d_loss, g_loss, gp = train_step_wgan_gp(
                    model.generator,
                    model.critic,
                    real_images,
                    optimizer_g,
                    optimizer_d,
                    model.latent_dim,
                    model.lambda_gp,
                    device,
                    update_generator=update_g
                )

                d_loss_meter.update(d_loss)
                if update_g:
                    g_loss_meter.update(g_loss)

            # Progress bar 업데이트
            pbar.set_postfix({
                'D_loss': f'{d_loss_meter.avg:.4f}',
                'G_loss': f'{g_loss_meter.avg:.4f}'
            })

        # 에포크 평균 손실
        d_losses.append(d_loss_meter.avg)
        g_losses.append(g_loss_meter.avg)

        # 로그 기록
        logger.log_epoch(
            epoch=epoch,
            train_loss=d_loss_meter.avg,
            metrics={'g_loss': g_loss_meter.avg}
        )

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"D Loss: {d_loss_meter.avg:.4f} | "
              f"G Loss: {g_loss_meter.avg:.4f}")

        # 샘플 생성
        if epoch % args.sample_interval == 0:
            model.generator.eval()
            with torch.no_grad():
                samples = model.generator(fixed_noise)

            save_path = output_dir / f'samples_epoch_{epoch:03d}.png'
            show_image_grid(
                samples,
                nrow=8,
                title=f'Generated Samples (Epoch {epoch})',
                save_path=save_path
            )

        # 체크포인트 저장
        if epoch % args.save_interval == 0:
            checkpoint_manager.save(
                model.generator,
                optimizer_g,
                epoch,
                g_loss_meter.avg,
                filename=f'checkpoint_epoch_{epoch}.pt'
            )

    # 로그 저장
    logger.save()
    print(f"\nTraining completed! Logs saved to {logger.log_file}")

    # 학습 곡선 시각화
    plot_training_curves(
        g_losses,
        d_losses,
        title=f'{args.model.upper()} Training Curves',
        save_path=output_dir / 'training_curves.png'
    )

    # 최종 샘플 생성
    print("\nGenerating final samples...")
    model.generator.eval()
    with torch.no_grad():
        final_samples = model.generator(model.sample_noise(100))

    show_image_grid(
        final_samples,
        nrow=10,
        title='Final Generated Samples',
        save_path=output_dir / 'final_samples.png'
    )

    print(f"\nOutputs saved to {output_dir}")
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
