"""Text generation utilities (sampling strategies).

Provides several sampling methods to generate tokens from a model's logits:
- greedy
- top_k sampling
- top_p (nucleus) sampling

Main API: sample_sequence(model, idx, max_new_tokens, context_size, strategy='greedy', **kwargs)
"""
from typing import Optional
import torch
import torch.nn.functional as F


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


def sample_sequence(model, idx: torch.Tensor, max_new_tokens: int, context_size: int,
                    strategy: str = 'greedy', k: int = 50, p: float = 0.9) -> torch.Tensor:
    """Generate tokens using chosen sampling strategy.

    Args:
        model: the language model (in eval mode)
        idx: (batch, seq_len) tensor of token ids
        max_new_tokens: number of new tokens to generate
        context_size: window size to crop context
        strategy: 'greedy' | 'top_k' | 'top_p'
        k: top-k parameter
        p: top-p nucleus parameter

    Returns:
        tensor of shape (batch, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if strategy == 'greedy':
            next_ids = greedy_step(logits)
        elif strategy == 'top_k':
            next_ids = top_k_step(logits, k=k)
        elif strategy == 'top_p':
            next_ids = top_p_step(logits, p=p)
        else:
            raise ValueError(f"Unknown generation strategy: {strategy}")

        idx = torch.cat((idx, next_ids), dim=1)

    return idx
