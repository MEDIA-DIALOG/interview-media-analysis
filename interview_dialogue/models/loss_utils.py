import torch
from typing import Tuple, Callable


def null_log(*args, **kwargs):
    pass


def compute_log_probability(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    mask: torch.BoolTensor = None,
    debug_fxn: Callable[[object, str], None] = null_log,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """
    Compute sum of log probs from model logits

    Arguments:
        logits (torch.FloatTensor): Model output logits (B x T x V)
        targets (torch.LongTensor): Target tokens (B x T)
        mask (torch.BoolTensor): Mask revealing only the utterance tokens (B x T)
        debug_fxn (callable): Logging function

    Returns:
        torch.FloatTensor: Target log probabilities (B x T)
        torch.LongTensor: Number of utterance tokens (1)
    """
    # Get log probability from logits via log softmax
    log_probs = torch.log_softmax(logits, dim=-1)
    debug_fxn(log_probs, 'log_probs')
    debug_fxn(targets, 'targets')

    # Extract target token probability - (B x T)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    debug_fxn(target_log_probs, 'target_log_probs')

    # Mask to utterance tokens
    if mask is not None:
        target_log_probs = target_log_probs.masked_select(mask)
        debug_fxn(target_log_probs, 'target_log_probs (masked)')
        n_tokens = mask.sum()
    else:
        n_tokens = target_log_probs.numel()
    debug_fxn(n_tokens, 'n_tokens')

    return target_log_probs, n_tokens
