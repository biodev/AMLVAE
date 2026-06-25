"""Training loss helpers (kept separate from VAE ELBO)."""

import torch.nn.functional as F


def adversarial_mse(pred, target, reduction="mean"):
    """MSE adversarial loss for continuous nuisance targets."""
    return F.mse_loss(pred, target, reduction=reduction)
