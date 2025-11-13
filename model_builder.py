"""
Model building and initialization utilities.
Encapsulates GPT model instantiation from configs.
"""
import torch
import torch.nn as nn
from typing import Union
from config import GPTConfig, load_config_from_file, get_small_config, get_default_config
from model import GPTModel


def build_model(cfg: GPTConfig, device: str = "cpu") -> nn.Module:
    """
    Build a GPT model from configuration.
    
    Args:
        cfg: GPTConfig instance.
        device: Device to place model on ("cpu", "cuda", "mps").
    
    Returns:
        GPTModel instance moved to specified device.
    """
    model = GPTModel(cfg)
    model.to(device)
    return model


def build_model_from_file(config_path: str = "config.json", device: str = "cpu") -> tuple[nn.Module, GPTConfig]:
    """
    Build a GPT model by loading config from file.
    
    Args:
        config_path: Path to config.json file.
        device: Device to place model on.
    
    Returns:
        Tuple of (model, config).
    """
    cfg = load_config_from_file(config_path)
    model = build_model(cfg, device)
    return model, cfg


def build_small_model(device: str = "cpu") -> tuple[nn.Module, GPTConfig]:
    """
    Build a small GPT model for quick testing.
    
    Args:
        device: Device to place model on.
    
    Returns:
        Tuple of (model, config).
    """
    cfg = get_small_config()
    model = build_model(cfg, device)
    return model, cfg


def build_default_model(device: str = "cpu") -> tuple[nn.Module, GPTConfig]:
    """
    Build the default (124M) GPT model.
    
    Args:
        device: Device to place model on.
    
    Returns:
        Tuple of (model, config).
    """
    cfg = get_default_config()
    model = build_model(cfg, device)
    return model, cfg


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module, cfg: GPTConfig):
    """Print model configuration and parameter count."""
    total_params = count_parameters(model)
    print("\n" + "="*50)
    print("Model Configuration")
    print("="*50)
    print(f"Vocab size:      {cfg.vocab_size}")
    print(f"Context length:  {cfg.context_length}")
    print(f"Embedding dim:   {cfg.emb_dim}")
    print(f"Num heads:       {cfg.n_heads}")
    print(f"Num layers:      {cfg.n_layers}")
    print(f"Dropout rate:    {cfg.drop_rate}")
    print(f"QKV bias:        {cfg.qkv_bias}")
    print(f"Total params:    {total_params:,}")
    print("="*50 + "\n")
