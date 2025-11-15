"""
Training script for generative models

Usage:
    python train.py --config configs/base_config.yaml
"""

import argparse
import yaml
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Import utils (assuming they're in utils/ directory)
from utils.logger import Logger, ProgressLogger
from utils.checkpoint import CheckpointManager, EarlyStopping


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
    if config.hardware.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"🖥️  Using CPU")

    return device


def build_model(config):
    """
    Build model based on config

    TODO: Implement your model here
    """
    # Example: Load from models module
    # from models import MODEL_REGISTRY
    # model = MODEL_REGISTRY[config.model.type](config)

    # Placeholder
    print(f"🏗️  Building model: {config.model.type}")
    print(f"   Latent dim: {config.model.latent_dim}")

    # TODO: Replace with actual model
    model = nn.Identity()  # Placeholder

    return model


def build_dataloader(config):
    """
    Build data loaders

    TODO: Implement your data loading here
    """
    # Example: Load standard datasets
    # from data import get_dataset
    # train_dataset = get_dataset(config.data.dataset, train=True)
    # val_dataset = get_dataset(config.data.dataset, train=False)

    # Placeholder
    print(f"📊 Loading dataset: {config.data.dataset}")

    # TODO: Replace with actual datasets
    train_loader = None  # Placeholder
    val_loader = None  # Placeholder

    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device, logger, epoch):
    """
    Train for one epoch

    TODO: Implement your training logic
    """
    model.train()

    progress = ProgressLogger(
        total=len(train_loader),
        desc=f"Epoch {epoch}",
        log_interval=10
    )

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)

        # Loss (TODO: Replace with actual loss)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update progress
        progress.update(1, {'loss': loss.item()})

        # Log to TensorBoard/W&B
        if batch_idx % logger.config.logging.log_interval == 0:
            logger.log_metrics({
                'train/loss': loss.item(),
            }, step=logger.global_step)

    return loss.item()  # Return last loss (or average)


def validate(model, val_loader, criterion, device):
    """
    Validation

    TODO: Implement validation logic
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)

    return avg_loss


def main(config):
    """Main training function"""
    print("="*70)
    print("Training Generative Model")
    print("="*70)

    # Set seed
    if hasattr(config, 'seed'):
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay
    )

    # Loss (TODO: Replace with actual loss)
    criterion = nn.MSELoss()

    # Logger
    logger = Logger(config)
    logger.info(f"Experiment: {config.logging.experiment_name}")

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir=config.checkpoint.save_dir,
        save_top_k=config.checkpoint.save_top_k,
        metric_mode='min'  # For loss
    )

    # Early stopping (if enabled)
    early_stopping = None
    if config.training.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta,
            mode='min'
        )

    # Resume from checkpoint (if specified)
    start_epoch = 0
    if config.checkpoint.resume_from:
        checkpoint = checkpoint_manager.load(
            config.checkpoint.resume_from,
            model=model,
            optimizer=optimizer,
            device=device
        )
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")

    for epoch in range(start_epoch, config.training.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, logger, epoch
        )

        # Validate
        if val_loader and (epoch % config.logging.eval_interval == 0):
            val_loss = validate(model, val_loader, criterion, device)

            logger.log_metrics({
                'val/loss': val_loss,
            }, step=epoch)

            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            # Save best checkpoint
            if config.checkpoint.save_best:
                checkpoint_manager.save_best(
                    model, optimizer, epoch,
                    metric_value=val_loss,
                    metric_name='loss'
                )

            # Early stopping check
            if early_stopping and early_stopping(val_loss):
                logger.info("Early stopping triggered!")
                break

        # Save regular checkpoint
        if (epoch % config.logging.save_interval == 0):
            checkpoint_manager.save(
                model, optimizer, epoch,
                metrics={'train_loss': train_loss}
            )

        # Generate samples (visualization)
        if (epoch % config.logging.visualize_interval == 0):
            # TODO: Implement sample generation
            pass

    # Final save
    checkpoint_manager.save(
        model, optimizer, config.training.epochs - 1,
        filename='final_model.pth'
    )

    logger.info("Training completed!")
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train generative model')
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
