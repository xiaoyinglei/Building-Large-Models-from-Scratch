"""
Visualization helpers for training.
Provides simple loss plotting using matplotlib.
"""
from typing import List, Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def plot_losses(train_losses: List[float], val_losses: List[float], batch_losses: Optional[List[float]] = None, out_path: str = "training_losses.png"):
    """
    Plot training and validation losses on the same figure.
    
    Args:
        train_losses: List of training losses (per evaluation)
        val_losses: List of validation losses (per evaluation)
        batch_losses: Optional list of batch-level losses
        out_path: Path to save the plot image
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it with `pip install matplotlib`.")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Train and validation losses (eval-level)
    if train_losses or val_losses:
        if train_losses:
            ax1.plot(train_losses, marker='o', linewidth=2, label="Training Loss", color='#1f77b4')
        if val_losses:
            ax1.plot(val_losses, marker='s', linewidth=2, label="Validation Loss", color='#ff7f0e')
        
        ax1.set_xlabel("Evaluation Step / Epoch", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Loss", fontsize=11, fontweight='bold')
        ax1.set_title("Training vs Validation Loss", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    # Right plot: Batch-level losses (if available)
    if batch_losses:
        # Plot a subsampled version of batch losses if too many
        n = len(batch_losses)
        step = max(1, n // 500)  # Subsample to ~500 points for clarity
        x_indices = list(range(0, n, step))
        y_values = [batch_losses[i] for i in x_indices]
        
        ax2.plot(x_indices, y_values, linewidth=1, label="Batch Loss", color='#2ca02c', alpha=0.7)
        ax2.set_xlabel("Batch Index", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Loss", fontsize=11, fontweight='bold')
        ax2.set_title("Batch-level Loss (subsampled)", fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No batch losses recorded\n(use --log-every flag to enable)', 
                ha='center', va='center', fontsize=11, transform=ax2.transAxes)
        ax2.set_xlabel("Batch Index", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Loss", fontsize=11, fontweight='bold')
        ax2.set_title("Batch-level Loss (subsampled)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"  ✓ Loss plot saved: {out_path}")
