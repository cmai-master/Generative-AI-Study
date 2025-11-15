"""
Logging utilities for training monitoring
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Universal logger for training monitoring

    Supports:
    - Console logging
    - TensorBoard
    - Weights & Biases (optional)
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration namespace/dict
        """
        self.config = config

        # Create log directory
        self.log_dir = Path(config.logging.log_dir) / config.logging.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        self.log_file = self.log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Setup TensorBoard
        self.tensorboard = None
        if config.logging.tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            self.tensorboard = SummaryWriter(log_dir=str(tb_dir))

        # Setup W&B (optional)
        self.wandb = None
        if config.logging.wandb.enabled:
            try:
                import wandb
                wandb.init(
                    project=config.logging.wandb.project,
                    entity=config.logging.wandb.entity,
                    name=config.logging.experiment_name,
                    config=self._config_to_dict(config)
                )
                self.wandb = wandb
            except ImportError:
                print("⚠️  wandb not installed. Install with: pip install wandb")

        self.global_step = 0

    def _config_to_dict(self, config):
        """Convert config namespace to dict for W&B"""
        if hasattr(config, '__dict__'):
            return {k: self._config_to_dict(v) for k, v in config.__dict__.items()}
        return config

    def info(self, message):
        """Log info message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        # Console
        print(log_message)

        # File
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def log_metrics(self, metrics, step=None, prefix=""):
        """
        Log metrics to all enabled loggers

        Args:
            metrics: Dict of metric_name -> value
            step: Global step (optional, uses internal counter if None)
            prefix: Prefix for metric names (e.g., "train/", "val/")
        """
        if step is None:
            step = self.global_step
            self.global_step += 1

        # TensorBoard
        if self.tensorboard:
            for name, value in metrics.items():
                self.tensorboard.add_scalar(f"{prefix}{name}", value, step)

        # W&B
        if self.wandb:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            wandb_metrics['step'] = step
            self.wandb.log(wandb_metrics)

    def log_images(self, images, step=None, prefix="", **kwargs):
        """
        Log images

        Args:
            images: Tensor (B, C, H, W) or list of images
            step: Global step
            prefix: Prefix for image name
            **kwargs: Additional arguments for make_grid
        """
        if step is None:
            step = self.global_step

        import torchvision

        # Make grid
        if isinstance(images, torch.Tensor):
            grid = torchvision.utils.make_grid(images, **kwargs)
        else:
            grid = torchvision.utils.make_grid(torch.stack(images), **kwargs)

        # TensorBoard
        if self.tensorboard:
            self.tensorboard.add_image(f"{prefix}samples", grid, step)

        # W&B
        if self.wandb:
            self.wandb.log({f"{prefix}samples": self.wandb.Image(grid), 'step': step})

    def log_histogram(self, name, values, step=None):
        """Log histogram of values"""
        if step is None:
            step = self.global_step

        if self.tensorboard:
            self.tensorboard.add_histogram(name, values, step)

    def log_model_gradients(self, model, step=None):
        """Log gradient statistics"""
        if step is None:
            step = self.global_step

        if self.tensorboard:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.tensorboard.add_histogram(f"gradients/{name}", param.grad, step)
                    self.tensorboard.add_scalar(
                        f"gradient_norms/{name}",
                        param.grad.norm().item(),
                        step
                    )

    def close(self):
        """Close all loggers"""
        if self.tensorboard:
            self.tensorboard.close()

        if self.wandb:
            self.wandb.finish()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name='meter'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class ProgressLogger:
    """
    Progress logger for training loops

    Example:
        progress = ProgressLogger(total=100, desc="Training")
        for i in range(100):
            # ... training ...
            progress.update(1, {'loss': loss.item()})
    """

    def __init__(self, total, desc="Progress", log_interval=10):
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.meters = {}

    def update(self, n=1, metrics=None):
        """
        Update progress

        Args:
            n: Increment amount
            metrics: Dict of metric_name -> value
        """
        self.current += n

        # Update meters
        if metrics:
            for name, value in metrics.items():
                if name not in self.meters:
                    self.meters[name] = AverageMeter(name)
                self.meters[name].update(value)

        # Log progress
        if self.current % self.log_interval == 0 or self.current == self.total:
            self._print_progress()

    def _print_progress(self):
        """Print progress bar"""
        percentage = 100.0 * self.current / self.total
        bar_length = 30
        filled = int(bar_length * self.current / self.total)
        bar = '█' * filled + '-' * (bar_length - filled)

        # Metrics string
        metrics_str = ' | '.join([str(meter) for meter in self.meters.values()])

        print(f'\r{self.desc}: [{bar}] {percentage:.1f}% | {metrics_str}', end='', flush=True)

        if self.current == self.total:
            print()  # New line at end

    def reset(self):
        """Reset progress"""
        self.current = 0
        for meter in self.meters.values():
            meter.reset()
