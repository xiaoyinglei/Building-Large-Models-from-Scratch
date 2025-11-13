"""
Configuration management for GPT model training.
Provides loading, validation, and default configurations.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """GPT model configuration."""
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GPTConfig":
        """Create GPTConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert GPTConfig to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "emb_dim": self.emb_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "drop_rate": self.drop_rate,
            "qkv_bias": self.qkv_bias,
        }


def load_config_from_file(config_path: str = "config.json") -> GPTConfig:
    """
    Load GPT configuration from a JSON file.
    
    Args:
        config_path: Path to config.json file. Defaults to "config.json" in current directory.
    
    Returns:
        GPTConfig instance.
    
    Raises:
        FileNotFoundError: If config file is not found.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    
    return GPTConfig.from_dict(config_dict)


def get_small_config() -> GPTConfig:
    """Get a small GPT configuration for quick testing."""
    return GPTConfig(
        vocab_size=50257,
        context_length=128,
        emb_dim=64,
        n_heads=4,
        n_layers=2,
        drop_rate=0.1,
        qkv_bias=False,
    )


def get_default_config() -> GPTConfig:
    """Get the default (124M) GPT configuration."""
    return GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )
