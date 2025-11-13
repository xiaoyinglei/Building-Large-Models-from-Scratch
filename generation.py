"""Text generation utilities (sampling strategies).

Provides several sampling methods to generate tokens from a model's logits:
- greedy
- top_k sampling
- top_p (nucleus) sampling

Main API: sample_sequence() with optional GenerationConfig parameter
"""
from typing import Optional, Union, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from training_utils import GenerationConfig


def greedy_step(logits: torch.Tensor) -> torch.Tensor:
    # logits: (batch, vocab)
    return torch.argmax(logits, dim=-1, keepdim=True)


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply temperature scaling to logits.
    
    Args:
        logits: (batch, vocab) tensor of logits
        temperature: Temperature parameter (>0). Lower = more deterministic, Higher = more random
                    1.0 = no change, <1.0 = sharpen distribution, >1.0 = flatten distribution
    
    Returns:
        Temperature-scaled logits
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    if temperature == 1.0:
        return logits
    return logits / temperature


def top_k_step(logits: torch.Tensor, k: int = 50, temperature: float = 1.0, filter_value: float = float('-Inf')) -> torch.Tensor:
    """ 
    logits: (batch, vocab)
    Top-k sampling step.
    Args:
        logits: (batch, vocab)
        k: keep top-k tokens (if <=0 then greedy)
        temperature: temperature for scaling (>0)
        filter_value: value to set for filtered logits (default -inf)
    Returns:
        next token ids: shape (batch, 1)
    """
    # validate shapes / types
    if logits.dim() != 2:
        raise ValueError("logits must be 2-D tensor of shape (batch, vocab)")
    
    batch, vocab_size = logits.shape

    if k <= 0:
        scaled = apply_temperature(logits, temperature)
        return greedy_step(scaled)
    
    # ensure k doesn't exceed vocab size
    k = min(int(k), vocab_size)

    # apply temperature
    logits = apply_temperature(logits, temperature)

    # find k-th largest value per row and mask others
    values, _ = torch.topk(logits, k, dim=-1)
    min_values = values[:, -1].unsqueeze(1)
    logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)
    probs = F.softmax(logits, dim=-1)
    
    # torch.multinomial expects non-negative probs rows that sum > 0
    # rows with all -inf would become NaN/zeros; topk ensures at least k tokens remain,
    # so this should be safe. Still, we add a tiny eps fallback to be robust.
    if torch.any(torch.isnan(probs)):
        # numerical fallback: replace NaNs with 0 and renormalize
        probs = torch.nan_to_num(probs, nan=0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        # if any row sums are zero (very unlikely), set uniform over vocab
        zero_rows = (row_sums == 0).squeeze(1)
        if zero_rows.any():
            probs[zero_rows] = torch.ones(vocab_size, device=probs.device) / vocab_size
        else:
            probs = probs / row_sums

    return torch.multinomial(probs, num_samples=1)


def top_p_step(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    filter_value: float = float('-Inf')
) -> torch.Tensor:
    """
    Apply nucleus (top-p) sampling step.

    Args:
        logits: (batch, vocab) tensor of logits
        p: cumulative probability threshold for nucleus sampling (0 < p <= 1)
        temperature: temperature scaling factor (>0)
        filter_value: value used to mask filtered logits (-Inf by default)

    Returns:
        Next token IDs sampled according to top-p probabilities.
    """
     # --- 参数检查 ---
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    if not (0 < p <= 1.0):
        raise ValueError(f"Top-p value must be in (0, 1], got {p}")

    # logits: (batch, vocab)
    logits = apply_temperature(logits, temperature)

    # p=1.0 时，不进行截断，直接按 softmax 采样
    if p == 1.0:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative prob above p
    sorted_indices_to_remove = cumulative_probs > p
    # shift right to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, filter_value)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def sample_sequence(model, idx: torch.Tensor, max_new_tokens: Optional[int] = None, 
                    context_size: int = 256, strategy: Optional[str] = None, 
                    k: Optional[int] = None, p: Optional[float] = None,
                    temperature: Optional[float] = None,
                    config: Optional["GenerationConfig"] = None) -> torch.Tensor:
    """Generate tokens using chosen sampling strategy.

    Flexible interface supporting both old-style parameters and new GenerationConfig approach.

    Args:
        model: the language model (in eval mode)
        idx: (batch, seq_len) tensor of token ids
        max_new_tokens: number of new tokens to generate (deprecated, use config instead)
        context_size: window size to crop context
        strategy: 'greedy' | 'top_k' | 'top_p' (deprecated, use config instead)
        k: top-k parameter (deprecated, use config instead)
        p: top-p nucleus parameter (deprecated, use config instead)
        temperature: temperature parameter (deprecated, use config instead)
        config: GenerationConfig instance (preferred approach)

    Returns:
        tensor of shape (batch, seq_len + max_new_tokens)

    Examples:
        # Old style (still supported for backward compatibility):
        tokens = sample_sequence(model, idx, max_new_tokens=50, 
                                context_size=256, strategy='top_k', k=50, temperature=0.8)
        
        # New style with GenerationConfig:
        from training_utils import GenerationConfig
        gen_cfg = GenerationConfig(max_new_tokens=50, strategy='top_k', top_k=50, temperature=0.8)
        tokens = sample_sequence(model, idx, context_size=256, config=gen_cfg)
    """
    # Handle config vs legacy parameters
    if config is not None:
        max_tokens = config.max_new_tokens
        strat = config.strategy
        top_k = config.top_k
        top_p = config.top_p
        temp = config.temperature
    else:
        # Use legacy parameters (with defaults for backward compatibility)
        max_tokens = max_new_tokens if max_new_tokens is not None else 50
        strat = strategy if strategy is not None else 'greedy'
        top_k = k if k is not None else 50
        top_p = p if p is not None else 0.9
        temp = temperature if temperature is not None else 1.0
    
    # basic validation
    if max_tokens is None:
        raise ValueError("max_tokens must be set")

    for _ in range(max_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # (batch, vocab)

        if strat == 'greedy':
            next_ids = greedy_step(logits)
        elif strat == 'top_k':
            next_ids = top_k_step(logits, k=top_k, temperature=temp)
        elif strat == 'top_p':
            next_ids = top_p_step(logits, p=top_p, temperature=temp)
        else:
            raise ValueError(f"Unknown generation strategy: {strat}")

        idx = torch.cat((idx, next_ids), dim=1)

    return idx
