"""
Training utilities and configurations.
Provides training parameter management and optimization helpers.
"""
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


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


@dataclass
class CustomTrainingConfig(TrainingConfig):
    """
    Extended training config for custom/free training mode.
    
    Allows users to set any training parameter with full flexibility.
    Supports model architecture customization for future extensions.
    
    Example:
        custom_cfg = CustomTrainingConfig(
            num_epochs=5,
            batch_size=16,
            learning_rate=1e-3,
            weight_decay=0.05,
            eval_freq=25,
            eval_iter=3,
            max_length=512,
            stride=256,
            start_context="Once upon a time",
            num_workers=2,
            model_overrides={'n_heads': 8, 'n_layers': 6}  # Optional model tweaks
        )
    """
    model_overrides: Dict[str, Any] = field(default_factory=dict)
    """Optional overrides for model config (e.g., {'n_heads': 8, 'n_layers': 6})"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CustomTrainingConfig":
        """
        Create CustomTrainingConfig from a dictionary.
        
        Args:
            config_dict: Dictionary with training parameters and optional model_overrides.
        
        Returns:
            CustomTrainingConfig instance.
        """
        # Extract model_overrides if present
        model_overrides = config_dict.pop('model_overrides', {})
        
        # Create instance with remaining parameters
        instance = cls(**config_dict)
        instance.model_overrides = model_overrides
        return instance

    def update(self, **kwargs) -> None:
        """
        Update config parameters dynamically.
        
        Args:
            **kwargs: Parameter name-value pairs to update.
        
        Example:
            cfg.update(learning_rate=1e-4, num_epochs=20)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'eval_freq': self.eval_freq,
            'eval_iter': self.eval_iter,
            'max_length': self.max_length,
            'stride': self.stride,
            'start_context': self.start_context,
            'num_workers': self.num_workers,
            'model_overrides': self.model_overrides,
        }


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
