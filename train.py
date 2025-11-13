"""
Training loop and utilities for GPT model.
Includes model training, evaluation, text generation and simple visualization hooks.
"""
import torch
from typing import Any, Iterable, Tuple, List, Dict, Optional
from text_to_token_ids import calc_loss_batch, calc_loss_loader, text_to_token_ids, token_ids_to_text, generate_text_simple

# Optional dependencies: tqdm for progress bars, visualize for plotting
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

def move_to_device(x: Any, device: str) -> Any:
    """
    Recursively move tensors (or containers of tensors) to device.
    Supports: torch.Tensor, dict, list, tuple.
    
    Ensures all model inputs and data are on the same device as the model
    to avoid "input tensor is on CPU but expected on device" errors.
    """
    if isinstance(x, torch.Tensor):
        # Ensure tensor is on the correct device
        if x.device != torch.device(device):
            return x.to(device)
        return x
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [move_to_device(i, device) for i in x]
        return type(x)(t)
    return x  # leave alone if not tensor/iterable

def safe_numel(x: Any) -> int:
    """
    Try to compute number of tokens/elements in the batch.
    - If tensor: use numel()
    - If dict with 'input_ids'/'input' keys: use length of that field
    - If list/tuple: sum lengths where possible
    - Fallback: return 0
    """
    try:
        if isinstance(x, torch.Tensor):
            return x.numel()
        if isinstance(x, dict):
            for key in ("input_ids", "inputs", "input"):
                if key in x:
                    v = x[key]
                    if isinstance(v, torch.Tensor):
                        return v.numel()
                    try:
                        return sum(len(item) for item in v)
                    except Exception:
                        pass
            # fallback: sum tensor-like fields
            total = 0
            for v in x.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel()
            return total
        if isinstance(x, (list, tuple)):
            s = 0
            for item in x:
                try:
                    s += safe_numel(item)
                except Exception:
                    pass
            return s
    except Exception:
        pass
    return 0

try:
    from visualize import plot_losses
except Exception:
    plot_losses = None

def safe_loss_value(loss: Any) -> float:
    """Return scalar float from loss in a defensive way."""
    try:
        return float(loss.item())
    except Exception:
        try:
            return float(loss)
        except Exception:
            return float('nan')
        
def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device: str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context,
    tokenizer,
    log_every: int = 0,
    max_batch_losses_saved: Optional[int] = None,
    gen_config=None
) -> Tuple[List[float], List[float], List[int]]:
    """
    Initialize lists to track losses and tokens seen
    Robust version of your training loop with:
    - safe device transfers
    - safe tqdm import/usage
    - defensive handling of eval_freq/log_every
    - safer tokens counting
    - optional limit on saved batch_losses to avoid unbounded memory growth
    - optional GenerationConfig for flexible text generation control
    
    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer for model updates.
        device: Device to use (cpu/cuda/mps).
        num_epochs: Number of training epochs.
        eval_freq: Evaluate every N steps.
        eval_iter: Number of batches for evaluation.
        start_context: Initial text for generation samples.
        tokenizer: Token encoder/decoder.
        log_every: Log batch loss every N steps (0 = no logging).
        max_batch_losses_saved: Max batch losses to keep in memory.
        gen_config: GenerationConfig instance for flexible text generation (optional).
    """
    # ensure model on device
    model.to(device)

    # protect eval_freq / log_every being zero or negative
    eval_freq_enabled = bool(eval_freq and eval_freq > 0)
    log_every_enabled = bool(log_every and log_every > 0)
    train_losses, val_losses, track_tokens_seen = [], [], []
    batch_losses = []
    tokens_seen, global_step = 0, 0

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        num_batches = len(train_loader) if hasattr(train_loader, '__len__') else None
        if num_batches is not None:
            print(f"Epoch {epoch+1} start - {num_batches} batches in this epoch")

        # Use tqdm if available for a nice progress bar
        iterator = train_loader
        if _tqdm is not None and num_batches is not None:
            iterator = _tqdm(train_loader, total=num_batches, desc=f"Epoch {epoch+1}")
        else:
            iterator = train_loader
            
        for batch in iterator:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input_batch, target_batch = batch[0], batch[1]
            else:
                # If custom dataset returns dict with 'input'/'target', try to extract
                if isinstance(batch, dict) and ('input' in batch or 'input_ids' in batch):
                    input_batch = batch
                    target_batch = batch.get('labels', None)
                else:
                    # fallback: treat batch as input only
                    input_batch, target_batch = batch, None

            # move to device (critical: ensure inputs match model device)
            input_batch = move_to_device(input_batch, device)
            if target_batch is not None:
                target_batch = move_to_device(target_batch, device)
            
            # Ensure model is on correct device
            model_device = next(model.parameters()).device
            assert str(model_device) == device or str(model_device).split(':')[0] == device, \
                f"Device mismatch: model on {model_device}, but expected {device}"

            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            
            # tokens counting (robust)
            tokens_seen += safe_numel(input_batch)
            global_step += 1

            # Optional per-batch logging: print batch loss every `log_every` steps
            if log_every_enabled and (global_step % log_every == 0):
                batch_loss_value = safe_loss_value(loss)
                try:
                    batch_loss_value = loss.item()
                except Exception:
                    batch_loss_value = float('nan')
                # update tqdm postfix if available
                if _tqdm is not None and num_batches is not None:
                    try:
                        iterator.set_postfix({"batch_loss": f"{batch_loss_value:.4f}"})
                    except Exception:
                        pass
                else:
                    print(f"Ep {epoch+1} (Step {global_step:06d}): Batch loss {batch_loss_value:.4f}")

            # collect batch losses for optional visualization
            batch_loss_value = safe_loss_value(loss)
            batch_losses.append(batch_loss_value)
            
            if max_batch_losses_saved and len(batch_losses) > max_batch_losses_saved:
                # keep latest N to avoid memory growing unbounded
                batch_losses = batch_losses[-max_batch_losses_saved:]
            
            # Optional evaluation step: evaluate every `eval_freq` steps (skip initial step 0)
            if eval_freq_enabled and (global_step % eval_freq == 0):
                try:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                except Exception as e:
                    print(f"Warning: evaluation failed at step {global_step} with error: {e}")
        
        # Ensure we have at least one recorded evaluation per epoch.
        # This appends a train/val loss entry even if eval_freq did not trigger.
        if eval_freq_enabled:
            try:
                train_loss_epoch, val_loss_epoch = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss_epoch)
                val_losses.append(val_loss_epoch)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (End Epoch): Train loss {train_loss_epoch:.3f}, Val loss {val_loss_epoch:.3f}")
            except Exception as e:
                # If evaluation fails for any reason, skip but continue training
                 print(f"Warning: end-of-epoch evaluation failed: {e}")

    # Print a sample text after each epoch
    try:
        generate_and_print_sample(model, tokenizer, device, start_context, gen_config=gen_config)
    except Exception as e:
        print(f"Warning: sample generation failed: {e}")


    # Visualization (if available) -- save a plot of losses
    try:
        if plot_losses is not None and (train_losses or val_losses or batch_losses):
            # prefer plotting eval losses; fall back to batch losses if eval not recorded
            plot_path = "training_losses.png"
            plot_losses(train_losses, val_losses, batch_losses=batch_losses, out_path=plot_path)
            print(f"✓ Saved training loss plot to {plot_path}")
    except Exception:
        pass

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context, gen_config=None):
    """
    Generate and print a sample text from the model.
    
    Args:
        model: The language model to generate from.
        tokenizer: Token encoder/decoder.
        device: Device to use for generation.
        start_context: Initial text to start generation.
        gen_config: GenerationConfig instance (optional). If None, uses default.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # Prefer the centralized generation utilities if available
        try:
            from generation import sample_sequence
            # Support both GenerationConfig and legacy parameter style
            if gen_config is not None:
                token_ids = sample_sequence(model=model, idx=encoded, 
                                           context_size=context_size, config=gen_config)
            else:
                token_ids = sample_sequence(model=model, idx=encoded, 
                                           max_new_tokens=50, context_size=context_size, strategy='greedy')
        except Exception:
            token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()