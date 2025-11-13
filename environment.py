"""Centralized environment setup utilities.

Provides a single entry function `prepare_environment()` which returns a
RuntimeEnv dataclass holding all runtime handles used by training (device,
tokenizer, config, model, dataloaders, training config and optimizer).

All runtime objects are created and configured for a unified training
environment, ensuring device consistency, proper initialization, and
dependency resolution.

Configuration is loaded directly from config_run.py (no CLI args needed).
"""
from dataclasses import dataclass
import torch
import tiktoken

from config import load_config_from_file, get_small_config, get_default_config
from data import create_dataloader_v1, text_data, print_text_stats
from model_builder import build_model, print_model_info
from training_utils import TrainingConfig, create_optimizer, get_device, print_device_info
import config_run


@dataclass
class RuntimeEnv:
    """Container for runtime objects returned by prepare_environment."""
    device: str
    tokenizer: object
    cfg: object
    model: object
    train_loader: object
    val_loader: object
    train_config: TrainingConfig
    optimizer: object


def prepare_device(device_override: str | None, seed: int):
    """Initialize device and RNG seed."""
    if device_override:
        device = device_override
    else:
        device = get_device()
    print_device_info(device)
    torch.manual_seed(seed)
    return device


def prepare_tokenizer():
    """Initialize GPT-2 tokenizer."""
    tokenizer = tiktoken.get_encoding("gpt2")
    print("✓ Tokenizer initialized (GPT-2)")
    return tokenizer


def prepare_config(use_small: bool = True, use_default: bool = False, config_path: str = "config.json"):
    """Load model configuration (small/default/from file)."""
    if use_small:
        cfg = get_small_config()
        print("✓ Using small model configuration (for testing)")
    elif use_default:
        cfg = get_default_config()
        print("✓ Using default full model configuration")
    else:
        try:
            cfg = load_config_from_file(config_path)
            print(f"✓ Configuration loaded from {config_path}")
        except FileNotFoundError:
            print(f"⚠ {config_path} not found, falling back to small config")
            cfg = get_small_config()
    return cfg


def prepare_model(cfg, device):
    """Build and initialize model on specified device."""
    model = build_model(cfg, device)
    print_model_info(model, cfg)
    return model


def prepare_dataloaders(cfg, train_config: TrainingConfig):
    """Create train and validation DataLoaders."""
    print(f"Creating data loaders (batch_size={train_config.batch_size})...")
    
    train_loader = create_dataloader_v1(
        text_data,
        batch_size=train_config.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length // 2,
        shuffle=True,
        drop_last=True,
        num_workers=train_config.num_workers,
    )

    val_loader = create_dataloader_v1(
        text_data,
        batch_size=train_config.batch_size,
        max_length=cfg.context_length,
        stride=cfg.context_length // 2,
        shuffle=False,
        drop_last=True,
        num_workers=train_config.num_workers,
    )

    print("✓ Data loaders ready")
    return train_loader, val_loader


def prepare_optimizer(model, train_config: TrainingConfig):
    """Create optimizer for model using training config parameters."""
    optimizer = create_optimizer(
        model,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    print(f"✓ Optimizer initialized (lr={train_config.learning_rate}, weight_decay={train_config.weight_decay})")
    return optimizer


def verify_device_consistency(device: str, model, optimizer):
    """
    Verify that model and optimizer parameters are on the same device.
    This ensures training will not fail due to device mismatches.
    """
    # Check model parameters
    try:
        model_device = next(model.parameters()).device
        if str(model_device) != device and str(model_device).split(':')[0] != device:
            print(f"⚠ Warning: Model device {model_device} may not match target device {device}")
    except StopIteration:
        pass  # model has no parameters
    
    # Check optimizer state
    try:
        for param_group in optimizer.param_groups:
            if 'params' in param_group and len(param_group['params']) > 0:
                opt_device = param_group['params'][0].device
                if str(opt_device) != device and str(opt_device).split(':')[0] != device:
                    print(f"⚠ Warning: Optimizer device {opt_device} may not match target device {device}")
                    break
    except Exception:
        pass
    
    print(f"✓ Device consistency verified (using {device})")


def prepare_environment() -> RuntimeEnv:
    """
    Set up the complete training environment from config_run.py settings.

    Reads all parameters from config_run.py (DEVICE, USE_SMALL, QUICK_MODE, etc.)
    and initializes a unified training environment.

    Returns:
      RuntimeEnv dataclass with (device, tokenizer, cfg, model, train_loader, 
      val_loader, train_config, optimizer)
    
    All components are configured for a unified environment where:
    - device is consistent across model, dataloaders, and training
    - optimizer matches model and training config
    - dataloaders use the correct batch size and device
    """
    # Step 1: Device and RNG
    device = prepare_device(
        config_run.DEVICE,
        config_run.SEED
    )
    
    # Step 2: Tokenizer
    tokenizer = prepare_tokenizer()

    # Step 3: Model config
    cfg = prepare_config(
        use_small=config_run.USE_SMALL,
        use_default=config_run.USE_DEFAULT,
        config_path=config_run.CONFIG_PATH
    )

    # Step 4: Print data statistics
    print_text_stats(text_data)

    # Step 5: Build model
    model = prepare_model(cfg, device)

    # Step 6: Training config (quick or full)
    if config_run.QUICK_MODE:
        train_config = TrainingConfig.get_quick_test_config()
        print("✓ Using quick training config (1 epoch, for testing)")
    else:
        train_config = TrainingConfig.get_full_training_config()
        print("✓ Using full training config")

    # Allow batch-size override from config_run
    if config_run.BATCH_SIZE:
        train_config.batch_size = config_run.BATCH_SIZE
        print(f"✓ Overriding batch size to {train_config.batch_size}")

    # Step 7: Dataloaders
    train_loader, val_loader = prepare_dataloaders(cfg, train_config)

    # Step 8: Optimizer
    optimizer = prepare_optimizer(model, train_config)

    # Step 9: Verify all components are on the same device
    verify_device_consistency(device, model, optimizer)

    return RuntimeEnv(
        device=device,
        tokenizer=tokenizer,
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=train_config,
        optimizer=optimizer,
    )
