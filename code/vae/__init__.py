"""
VAE (Variational Autoencoder) Package

Basic VAE와 β-VAE 구현을 제공합니다.
"""

from .basic_vae import (
    VAE,
    Encoder,
    Decoder,
    vae_loss,
    train_epoch,
    evaluate
)

from .beta_vae import (
    BetaVAE,
    beta_vae_loss,
    train_epoch_beta_vae,
    evaluate_beta_vae
)

__all__ = [
    'VAE',
    'Encoder',
    'Decoder',
    'vae_loss',
    'train_epoch',
    'evaluate',
    'BetaVAE',
    'beta_vae_loss',
    'train_epoch_beta_vae',
    'evaluate_beta_vae',
]
