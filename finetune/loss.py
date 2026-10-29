import torch
from torch.nn import functional as F


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
):
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight

    # Moshi deliberately fills invalid delayed positions with NaN. Exclude
    # them BEFORE cross entropy; masking the loss afterwards can poison gradients.
    logits = logits.reshape(-1, logits.size(-1)).float()
    target = target.reshape(-1)
    weights = weights.reshape(-1)
    selected = weights > 0
    logits, target, weights = logits[selected], target[selected], weights[selected]
    if weights.numel() == 0:
        # Differentiable zero without reading any invalid logit values.
        return logits.sum()
    losses = F.cross_entropy(logits, target, reduction="none")
    if not torch.isfinite(losses).all():
        raise FloatingPointError(f"Nonfinite {mode} loss at an unmasked target")
    return (losses * weights).sum() / weights.sum()
