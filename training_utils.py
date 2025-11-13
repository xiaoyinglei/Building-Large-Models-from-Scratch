"""
Training utilities and configurations.
Provides training parameter management and optimization helpers.
"""
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    eval_freq: int = 50
    eval_iter: int = 2
    max_length: int = 256
    stride: int = 128
    start_context: str = "Every effort moves you"
    num_workers: int = 0

    @classmethod
    def get_quick_test_config(cls) -> "TrainingConfig":
        """Get config for quick local testing (1 epoch, small batch)."""
        return cls(
            num_epochs=1,
            batch_size=8,
            learning_rate=3e-4,
            weight_decay=0.1,
            eval_freq=50,
            eval_iter=2,
            max_length=128,
            stride=64,
            start_context="Every effort moves you",
            num_workers=0,
        )

    @classmethod
    def get_full_training_config(cls) -> "TrainingConfig":
        """Get config for full model training."""
        return cls(
            num_epochs=10,
            batch_size=32,
            learning_rate=5e-4,
            weight_decay=0.1,
            eval_freq=100,
            eval_iter=5,
            max_length=256,
            stride=128,
            start_context="Every effort moves you",
            num_workers=0,
        )


def create_optimizer(model, learning_rate: float = 3e-4, weight_decay: float = 0.1):
    """
    Create AdamW optimizer for model.
    
    Args:
        model: PyTorch model.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization).
    
    Returns:
        AdamW optimizer instance.
    """
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_device() -> str:
    """
    Get the best available device.
    Priority: mps (Apple Silicon) > cuda > cpu
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def print_device_info(device: str):
    """Print information about the selected device."""
    print(f"\n{'='*50}")
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device == "mps":
        print("GPU: Apple Metal Performance Shaders")
    print(f"{'='*50}\n")
