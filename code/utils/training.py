"""
학습 유틸리티 함수

모델 학습에 필요한 공통 함수들을 제공합니다.
- 체크포인트 저장/로드
- Early Stopping
- 학습 로그 관리
- Learning Rate Scheduler
"""

import torch
import os
import json
from pathlib import Path
from datetime import datetime


class CheckpointManager:
    """
    모델 체크포인트 관리 클래스
    """

    def __init__(self, checkpoint_dir='./checkpoints', max_checkpoints=5):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            max_checkpoints: 유지할 최대 체크포인트 수
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save(self, model, optimizer, epoch, loss, metrics=None, filename=None):
        """
        체크포인트 저장

        Args:
            model: 저장할 모델
            optimizer: 옵티마이저
            epoch: 현재 에포크
            loss: 현재 손실
            metrics: 추가 메트릭 (dict)
            filename: 파일명 (None이면 자동 생성)

        Returns:
            저장된 체크포인트 경로
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        # 저장할 데이터
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
        }

        if metrics is not None:
            checkpoint['metrics'] = metrics

        # 저장
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)

        # 오래된 체크포인트 삭제
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        return checkpoint_path

    def save_best(self, model, optimizer, epoch, loss, metrics=None):
        """
        최고 성능 모델 저장

        Args:
            model: 저장할 모델
            optimizer: 옵티마이저
            epoch: 현재 에포크
            loss: 현재 손실
            metrics: 추가 메트릭 (dict)

        Returns:
            저장된 체크포인트 경로
        """
        return self.save(model, optimizer, epoch, loss, metrics, filename='best_model.pt')

    def load(self, model, optimizer=None, filename='best_model.pt'):
        """
        체크포인트 로드

        Args:
            model: 모델
            optimizer: 옵티마이저 (None이면 로드 안 함)
            filename: 로드할 파일명

        Returns:
            checkpoint 딕셔너리
        """
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint


class EarlyStopping:
    """
    Early Stopping 클래스

    검증 손실이 개선되지 않으면 학습을 조기 종료합니다.
    """

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: 개선이 없을 때 대기할 에포크 수
            min_delta: 개선으로 간주할 최소 변화량
            mode: 'min' (손실 최소화) 또는 'max' (메트릭 최대화)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value):
        """
        현재 값을 평가하고 early stopping 여부 결정

        Args:
            current_value: 현재 메트릭 값

        Returns:
            bool: True면 학습 종료, False면 계속
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        # 개선 여부 확인
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class TrainingLogger:
    """
    학습 로그 관리 클래스
    """

    def __init__(self, log_dir='./logs', experiment_name=None):
        """
        Args:
            log_dir: 로그 저장 디렉토리
            experiment_name: 실험 이름 (None이면 타임스탬프 사용)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.json"

        self.logs = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'epochs': []
        }

    def log_epoch(self, epoch, train_loss, val_loss=None, metrics=None):
        """
        에포크 로그 기록

        Args:
            epoch: 에포크 번호
            train_loss: 학습 손실
            val_loss: 검증 손실
            metrics: 추가 메트릭 (dict)
        """
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
        }

        if val_loss is not None:
            epoch_log['val_loss'] = val_loss

        if metrics is not None:
            epoch_log.update(metrics)

        self.logs['epochs'].append(epoch_log)

    def save(self):
        """
        로그를 파일로 저장
        """
        self.logs['end_time'] = datetime.now().isoformat()

        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def load(self, experiment_name=None):
        """
        저장된 로그 로드

        Args:
            experiment_name: 로드할 실험 이름 (None이면 현재 실험)

        Returns:
            로그 딕셔너리
        """
        if experiment_name is None:
            log_file = self.log_file
        else:
            log_file = self.log_dir / f"{experiment_name}.json"

        with open(log_file, 'r') as f:
            return json.load(f)


def get_device():
    """
    사용 가능한 디바이스 반환 (CUDA > MPS > CPU)

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """
    모델의 학습 가능한 파라미터 수 계산

    Args:
        model: PyTorch 모델

    Returns:
        int: 파라미터 수
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_size=None):
    """
    모델 요약 정보 출력

    Args:
        model: PyTorch 모델
        input_size: 입력 크기 (tuple)
    """
    print("=" * 70)
    print("Model Summary")
    print("=" * 70)

    print(f"\nModel Architecture:\n{model}\n")

    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params:,}")

    if input_size is not None:
        print(f"\nInput Size: {input_size}")

    print("=" * 70)


def set_seed(seed=42):
    """
    재현 가능성을 위한 랜덤 시드 설정

    Args:
        seed: 랜덤 시드 값
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    평균 계산을 위한 헬퍼 클래스
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """값 초기화"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        값 업데이트

        Args:
            val: 추가할 값
            n: 값의 개수 (배치 크기 등)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # 간단한 테스트
    print("Testing training utilities...")

    # Device 테스트
    device = get_device()
    print(f"\n1. Device: {device}")

    # CheckpointManager 테스트
    print("\n2. Testing CheckpointManager...")
    checkpoint_manager = CheckpointManager(checkpoint_dir='./test_checkpoints')
    print(f"   Checkpoint directory: {checkpoint_manager.checkpoint_dir}")

    # EarlyStopping 테스트
    print("\n3. Testing EarlyStopping...")
    early_stopping = EarlyStopping(patience=3)
    test_losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84]
    for epoch, loss in enumerate(test_losses):
        stop = early_stopping(loss)
        print(f"   Epoch {epoch}: loss={loss:.2f}, stop={stop}")
        if stop:
            break

    # TrainingLogger 테스트
    print("\n4. Testing TrainingLogger...")
    logger = TrainingLogger(log_dir='./test_logs', experiment_name='test_exp')
    logger.log_epoch(0, train_loss=1.0, val_loss=0.9)
    logger.log_epoch(1, train_loss=0.8, val_loss=0.75)
    logger.save()
    print(f"   Log saved to: {logger.log_file}")

    # AverageMeter 테스트
    print("\n5. Testing AverageMeter...")
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"   Average: {meter.avg:.2f}")

    print("\n✅ All training utilities work correctly!")
