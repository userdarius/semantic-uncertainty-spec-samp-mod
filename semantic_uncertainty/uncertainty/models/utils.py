import torch
from torch.nn import functional as F
import logging


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits


def norm_logits(
    logits: torch.Tensor, temperature: float, top_k: float, top_p: float
) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs: torch.Tensor, num_samples: int = 1):
    """Sample from probability distribution with validation."""
    try:
        # Check for valid probability distribution
        if torch.isnan(probs).any():
            raise ValueError("NaN values in probability distribution")

        if (probs == 0).all():
            raise ValueError("All zero probability distribution")

        sum_probs = probs.sum(dim=1)
        if not torch.allclose(sum_probs, torch.ones_like(sum_probs), rtol=1e-5):
            raise ValueError(f"Probabilities don't sum to 1: {sum_probs.item()}")

        idx_next = torch.multinomial(probs, num_samples=num_samples)

        if idx_next.item() == 0:
            # Instead of raising RuntimeError, try to sample again excluding 0
            probs[:, 0] = 0  # Zero out probability of selecting 0
            if probs.sum() > 0:  # Check if we still have valid probabilities
                probs = probs / probs.sum()  # Renormalize
                idx_next = torch.multinomial(probs, num_samples=num_samples)
            else:
                # If no valid probabilities left, sample from uniform distribution
                idx_next = torch.randint(1, probs.size(-1), (1,))

        return idx_next

    except Exception as e:
        logging.error("Sampling failed:")
        logging.error(f"Probability shape: {probs.shape}")
        logging.error(
            f"Probability stats - min: {probs.min().item()}, max: {probs.max().item()}, sum: {probs.sum().item()}"
        )
        raise RuntimeError(f"Sampling failed: {str(e)}")


def max_fn(x):
    """
    norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum
