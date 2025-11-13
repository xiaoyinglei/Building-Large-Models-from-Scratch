"""
GPT Model Training Entry Point.

This module orchestrates the complete training pipeline by delegating
environment setup to environment.py, allowing main() to focus on
training orchestration and result reporting.

To customize training parameters, edit run_config.py directly and run:
  python main.py
"""
from train import train_model_simple
from environment import prepare_environment
import config_run


def run_training(model, train_loader, val_loader, optimizer, device, train_config, tokenizer, log_every: int = 0):
    """Execute the training loop.

    Args:
        model: GPT model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer for model updates.
        device: Device to use (cpu/cuda/mps).
        train_config: Training configuration (contains num_epochs, eval_freq, etc).
        tokenizer: Token encoder/decoder.
        log_every: Log batch loss every N steps (0 = no logging).
    """
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70 + "\n")
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=train_config.num_epochs,
        eval_freq=train_config.eval_freq,
        eval_iter=train_config.eval_iter,
        start_context=train_config.start_context,
        tokenizer=tokenizer,
        log_every=log_every,
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return train_losses, val_losses, tokens_seen


def report_results(train_losses, val_losses):
    """Print training results summary."""
    print("\nTraining Summary:")
    print(f"  Final train loss: {train_losses[-1]:.4f}" if train_losses else "  No training losses recorded")
    print(f"  Final val loss:   {val_losses[-1]:.4f}" if val_losses else "  No validation losses recorded")
    
    if train_losses and val_losses:
        best_val_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
        print(f"  Best val loss:    {val_losses[best_val_idx]:.4f} (at step {best_val_idx})")


def main():
    """
    Main training entry point.
    
    Orchestrates training by:
    1. Loading parameters from config_run.py
    2. Calling prepare_environment() to set up all runtime objects
    3. Running the training loop
    4. Reporting results
    """
    print("\n" + "="*70)
    print("GPT Model Training Pipeline")
    print("="*70 + "\n")
    
    # Step 1-9: Prepare entire unified environment (reads from config_run.py)
    env = prepare_environment()

    # Step 10: Run training with log_every from config
    train_losses, val_losses, tokens_seen = run_training(
        model=env.model,
        train_loader=env.train_loader,
        val_loader=env.val_loader,
        optimizer=env.optimizer,
        device=env.device,
        train_config=env.train_config,
        tokenizer=env.tokenizer,
        log_every=config_run.LOG_EVERY,
    )
    
    # Step 11: Report results
    report_results(train_losses, val_losses)
    
    print("\n✓ Training pipeline finished successfully!")


if __name__ == "__main__":
    # Run training using parameters from config_run.py
    # To customize training, edit config_run.py
    main()

