"""
Checkpoint management utilities
"""

import os
import torch
from pathlib import Path
from datetime import datetime
import shutil


class CheckpointManager:
    """
    Manage model checkpoints

    Features:
    - Save/load checkpoints
    - Keep best K checkpoints
    - Resume training
    """

    def __init__(self, save_dir, save_top_k=3, metric_mode='min'):
        """
        Args:
            save_dir: Directory to save checkpoints
            save_top_k: Keep top K best checkpoints
            metric_mode: 'min' or 'max' for best metric
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_top_k = save_top_k
        self.metric_mode = metric_mode

        # Track saved checkpoints
        self.checkpoints = []  # List of (metric_value, path)

    def save(self, model, optimizer, epoch, metrics=None, filename=None, **kwargs):
        """
        Save checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dict of metrics (e.g., {'loss': 0.5, 'fid': 10.2})
            filename: Custom filename (optional)
            **kwargs: Additional items to save
        """
        # Create checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
        }

        # Add custom items
        checkpoint.update(kwargs)

        # Generate filename
        if filename is None:
            filename = f'checkpoint_epoch_{epoch:04d}.pth'

        save_path = self.save_dir / filename

        # Save
        torch.save(checkpoint, save_path)
        print(f"💾 Saved checkpoint: {save_path}")

        return save_path

    def save_best(self, model, optimizer, epoch, metric_value, metric_name, **kwargs):
        """
        Save checkpoint and maintain top-K best

        Args:
            model, optimizer, epoch: Same as save()
            metric_value: Value of metric to track
            metric_name: Name of metric (e.g., 'loss', 'fid')
            **kwargs: Additional items
        """
        # Create checkpoint
        metrics = {metric_name: metric_value}
        filename = f'best_{metric_name}_{metric_value:.4f}_epoch_{epoch:04d}.pth'

        save_path = self.save(
            model, optimizer, epoch,
            metrics=metrics,
            filename=filename,
            **kwargs
        )

        # Add to list
        self.checkpoints.append((metric_value, save_path))

        # Sort (ascending for min, descending for max)
        reverse = (self.metric_mode == 'max')
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Remove worst checkpoints
        if len(self.checkpoints) > self.save_top_k:
            # Delete oldest/worst
            to_delete = self.checkpoints[self.save_top_k:]
            for _, path in to_delete:
                if path.exists():
                    path.unlink()
                    print(f"🗑️  Deleted old checkpoint: {path}")

            self.checkpoints = self.checkpoints[:self.save_top_k]

        # Print current top-K
        print(f"\n📊 Top-{self.save_top_k} {metric_name}:")
        for i, (value, path) in enumerate(self.checkpoints, 1):
            print(f"  {i}. {metric_name}={value:.4f} - {path.name}")
        print()

    def load(self, checkpoint_path, model=None, optimizer=None, device='cpu'):
        """
        Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            device: Device to load to

        Returns:
            checkpoint: Full checkpoint dict
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"📂 Loaded checkpoint: {checkpoint_path}")

        # Load model state
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Model state loaded")

        # Load optimizer state
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"   ✅ Optimizer state loaded")

        # Print info
        if 'epoch' in checkpoint:
            print(f"   📅 Epoch: {checkpoint['epoch']}")

        if 'metrics' in checkpoint:
            print(f"   📈 Metrics: {checkpoint['metrics']}")

        return checkpoint

    def get_latest_checkpoint(self):
        """Get path to most recent checkpoint"""
        checkpoints = list(self.save_dir.glob('checkpoint_*.pth'))

        if not checkpoints:
            return None

        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

        return latest

    def get_best_checkpoint(self, metric_name='loss'):
        """Get path to best checkpoint for given metric"""
        pattern = f'best_{metric_name}_*.pth'
        checkpoints = list(self.save_dir.glob(pattern))

        if not checkpoints:
            return None

        # Find checkpoint with best metric in filename
        def extract_metric(path):
            # Parse: best_{metric}_{value}_epoch_{epoch}.pth
            parts = path.stem.split('_')
            try:
                # Find value after metric name
                idx = parts.index(metric_name) + 1
                return float(parts[idx])
            except:
                return float('inf')

        if self.metric_mode == 'min':
            best = min(checkpoints, key=extract_metric)
        else:
            best = max(checkpoints, key=extract_metric)

        return best


class EarlyStopping:
    """
    Early stopping to stop training when metric stops improving

    Example:
        early_stopping = EarlyStopping(patience=10, mode='min')

        for epoch in range(epochs):
            val_loss = validate(...)
            if early_stopping(val_loss):
                print("Early stopping!")
                break
    """

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, metric_value):
        """
        Check if should stop

        Args:
            metric_value: Current metric value

        Returns:
            bool: True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False

        # Check if improved
        if self.mode == 'min':
            improved = (metric_value < self.best_value - self.min_delta)
        else:
            improved = (metric_value > self.best_value + self.min_delta)

        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            print(f"⏳ Early stopping counter: {self.counter}/{self.patience}")

        # Check if should stop
        if self.counter >= self.patience:
            print(f"🛑 Early stopping triggered!")
            self.early_stop = True
            return True

        return False

    def reset(self):
        """Reset early stopping"""
        self.counter = 0
        self.best_value = None
        self.early_stop = False


def save_checkpoint_simple(model, optimizer, epoch, save_path, **kwargs):
    """
    Simple function to save checkpoint

    Args:
        model: Model to save
        optimizer: Optimizer
        epoch: Current epoch
        save_path: Where to save
        **kwargs: Additional items
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint.update(kwargs)

    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")


def load_checkpoint_simple(checkpoint_path, model=None, optimizer=None, device='cpu'):
    """
    Simple function to load checkpoint

    Args:
        checkpoint_path: Path to checkpoint
        model: Model (optional)
        optimizer: Optimizer (optional)
        device: Device

    Returns:
        checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"📂 Checkpoint loaded: {checkpoint_path}")

    return checkpoint
