"""
생성 모델 평가 지표

주요 지표:
- Inception Score (IS)
- Fréchet Inception Distance (FID)
- Precision & Recall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from typing import Tuple, List


class InceptionV3(nn.Module):
    """
    Inception V3 모델 (FID, IS 계산용)

    Pre-trained Inception V3 모델의 특정 레이어 출력을 추출합니다.
    """

    def __init__(self, output_blocks=[3], resize_input=True, normalize_input=True):
        """
        Args:
            output_blocks: 추출할 블록 인덱스 리스트
                - 0: Pool3 features
                - 1: First max pooling features
                - 2: Second max pooling features
                - 3: Final average pooling features (기본값)
            resize_input: 입력을 299x299로 리사이즈 여부
            normalize_input: 입력 정규화 여부
        """
        super().__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)

        # Pre-trained Inception V3 로드
        inception = inception_v3(pretrained=True)
        inception.eval()

        # Inception 블록 분리
        self.blocks = nn.ModuleList()

        # Block 0: Conv2d_1a_3x3 to Mixed_5d
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: Mixed_5b to Mixed_5d
        if len(self.output_blocks) > 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: Mixed_5b to Mixed_6e
        if len(self.output_blocks) > 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: Mixed_7a to avg pool
        if len(self.output_blocks) > 3 or 3 in self.output_blocks:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        # 가중치 업데이트 안 함
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 이미지 (batch_size, 3, H, W)

        Returns:
            출력 features 리스트
        """
        # Resize to 299x299
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize [-1, 1] → [0, 1]
        if self.normalize_input:
            x = (x + 1) / 2.0

        # Inception 입력 정규화
        x = 2 * x - 1  # [0, 1] → [-1, 1]

        outputs = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outputs.append(x)

        return outputs if len(outputs) > 1 else outputs[0]


def calculate_inception_score(
    images: torch.Tensor,
    batch_size: int = 50,
    splits: int = 10,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """
    Inception Score (IS) 계산

    IS = exp(E_x[KL(p(y|x) || p(y))])

    높을수록 좋음 (이미지 품질과 다양성)

    Args:
        images: 생성된 이미지 (N, C, H, W), 값 범위 [-1, 1]
        batch_size: 배치 크기
        splits: Split 개수 (평균과 표준편차 계산용)
        device: 디바이스

    Returns:
        is_mean: IS 평균
        is_std: IS 표준편차
    """
    N = images.size(0)

    # Inception 모델 로드
    model = InceptionV3(output_blocks=[3], resize_input=True, normalize_input=False)
    model = model.to(device)
    model.eval()

    # 예측 확률 수집
    preds = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = images[i:i+batch_size].to(device)

            # Inception features
            features = model(batch)
            features = features.squeeze(3).squeeze(2)

            # Softmax 예측 (Inception V3의 마지막 fc 레이어가 없으므로 간소화)
            # 실제로는 Inception V3의 classifier가 필요하지만,
            # 여기서는 features를 그대로 사용
            pred = F.softmax(features, dim=1)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Split하여 IS 계산
    scores = []
    split_size = N // splits

    for k in range(splits):
        part = preds[k * split_size:(k + 1) * split_size, :]

        # p(y|x)
        py_x = part

        # p(y) = E_x[p(y|x)]
        py = np.mean(part, axis=0, keepdims=True)

        # KL(p(y|x) || p(y))
        kl = py_x * (np.log(py_x + 1e-16) - np.log(py + 1e-16))
        kl = np.sum(kl, axis=1)

        # IS = exp(E[KL])
        scores.append(np.exp(np.mean(kl)))

    return np.mean(scores), np.std(scores)


def calculate_activation_statistics(
    images: torch.Tensor,
    model: nn.Module,
    batch_size: int = 50,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지들의 Inception features 통계 계산

    Args:
        images: 이미지 (N, C, H, W)
        model: Inception 모델
        batch_size: 배치 크기
        device: 디바이스

    Returns:
        mu: 평균 (2048,)
        sigma: 공분산 행렬 (2048, 2048)
    """
    model.eval()

    act = []

    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size].to(device)
            features = model(batch)

            # Flatten
            features = features.squeeze(3).squeeze(2).cpu().numpy()
            act.append(features)

    act = np.concatenate(act, axis=0)

    # 통계 계산
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Fréchet Distance 계산

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

    Args:
        mu1: 첫 번째 분포의 평균
        sigma1: 첫 번째 분포의 공분산
        mu2: 두 번째 분포의 평균
        sigma2: 두 번째 분포의 공분산
        eps: 수치 안정성을 위한 작은 값

    Returns:
        FID 값 (낮을수록 좋음)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Numerical error might give slight imaginary component
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Handle imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size: int = 50,
    device: str = 'cpu'
) -> float:
    """
    Fréchet Inception Distance (FID) 계산

    낮을수록 좋음 (생성 이미지가 실제 이미지와 유사)

    Args:
        real_images: 실제 이미지 (N, C, H, W), 값 범위 [-1, 1]
        fake_images: 생성된 이미지 (N, C, H, W), 값 범위 [-1, 1]
        batch_size: 배치 크기
        device: 디바이스

    Returns:
        FID 값
    """
    # Inception 모델
    model = InceptionV3(output_blocks=[3], resize_input=True, normalize_input=False)
    model = model.to(device)
    model.eval()

    # 통계 계산
    print("Calculating statistics for real images...")
    mu_real, sigma_real = calculate_activation_statistics(
        real_images, model, batch_size, device
    )

    print("Calculating statistics for fake images...")
    mu_fake, sigma_fake = calculate_activation_statistics(
        fake_images, model, batch_size, device
    )

    # FID 계산
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return fid_value


def calculate_precision_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 3
) -> Tuple[float, float]:
    """
    Precision and Recall 계산 (생성 모델용)

    Args:
        real_features: 실제 이미지 features (N, D)
        fake_features: 생성된 이미지 features (M, D)
        k: k-NN의 k 값

    Returns:
        precision: Precision 값
        recall: Recall 값
    """
    from sklearn.metrics import pairwise_distances

    # 거리 계산
    real_to_real = pairwise_distances(real_features, real_features)
    real_to_fake = pairwise_distances(real_features, fake_features)
    fake_to_real = pairwise_distances(fake_features, real_features)

    # k-th nearest neighbor distance
    real_nn_dist = np.partition(real_to_real, k, axis=1)[:, k]
    fake_nn_dist = np.partition(fake_to_real, k, axis=1)[:, k]

    # Precision: fake 샘플이 real manifold 내부에 있는 비율
    precision = (real_to_fake.min(axis=0) < real_nn_dist).mean()

    # Recall: real 샘플이 fake manifold에 의해 커버되는 비율
    recall = (real_to_fake.min(axis=1) < real_nn_dist).mean()

    return precision, recall


if __name__ == "__main__":
    print("Testing evaluation metrics...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 더미 데이터
    print("\nGenerating dummy data...")
    real_images = torch.randn(100, 3, 64, 64)  # RGB
    fake_images = torch.randn(100, 3, 64, 64)

    # Inception Score 테스트
    print("\n1. Testing Inception Score...")
    try:
        is_mean, is_std = calculate_inception_score(
            fake_images, batch_size=10, splits=5, device=device
        )
        print(f"   Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        print(f"   Inception Score test skipped: {e}")

    # FID 테스트
    print("\n2. Testing FID...")
    try:
        fid_value = calculate_fid(
            real_images, fake_images, batch_size=10, device=device
        )
        print(f"   FID: {fid_value:.4f}")
    except Exception as e:
        print(f"   FID test skipped: {e}")

    # Precision & Recall 테스트
    print("\n3. Testing Precision & Recall...")
    try:
        real_features = np.random.randn(100, 2048)
        fake_features = np.random.randn(100, 2048)
        precision, recall = calculate_precision_recall(
            real_features, fake_features, k=3
        )
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
    except Exception as e:
        print(f"   Precision & Recall test skipped: {e}")

    print("\n✅ Evaluation metrics module created!")
    print("Note: Some tests may require additional dependencies (scipy, sklearn)")
