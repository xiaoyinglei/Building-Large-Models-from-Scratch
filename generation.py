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


def top_k_step(logits: torch.Tensor, k: int = 50, filter_value: float = -float('Inf')) -> torch.Tensor:
    # logits: (batch, vocab)
    if k <= 0:
        return greedy_step(logits)
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_p_step(logits: torch.Tensor, p: float = 0.9, filter_value: float = -float('Inf')) -> torch.Tensor:
    # logits: (batch, vocab)
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
        config: GenerationConfig instance (preferred approach)

    Returns:
        tensor of shape (batch, seq_len + max_new_tokens)

    Examples:
        # Old style (still supported for backward compatibility):
        tokens = sample_sequence(model, idx, max_new_tokens=50, 
                                context_size=256, strategy='top_k', k=50)
        
        # New style with GenerationConfig:
        from training_utils import GenerationConfig
        gen_cfg = GenerationConfig(max_new_tokens=50, strategy='top_k', top_k=50)
        tokens = sample_sequence(model, idx, context_size=256, config=gen_cfg)
    """
    # Handle config vs legacy parameters
    if config is not None:
        max_tokens = config.max_new_tokens
        strat = config.strategy
        top_k = config.top_k
        top_p = config.top_p
    else:
        # Use legacy parameters (with defaults for backward compatibility)
        max_tokens = max_new_tokens if max_new_tokens is not None else 50
        strat = strategy if strategy is not None else 'greedy'
        top_k = k if k is not None else 50
        top_p = p if p is not None else 0.9

    for _ in range(max_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if strat == 'greedy':
            next_ids = greedy_step(logits)
        elif strat == 'top_k':
            next_ids = top_k_step(logits, k=top_k)
        elif strat == 'top_p':
            next_ids = top_p_step(logits, p=top_p)
        else:
            raise ValueError(f"Unknown generation strategy: {strat}")

        idx = torch.cat((idx, next_ids), dim=1)

    return idx
