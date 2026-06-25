import math
import json
import os

import torch
from torch import nn
import torch.nn.functional as F

from amlvae.models.utils import get_nonlin, get_norm
from amlvae.models.MLP import MLP

LOG2PI = math.log(2.0 * math.pi)

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.alpha)


class VAE(nn.Module):
    """Variational Autoencoder with a proper Gaussian likelihood.

    The decoder outputs the mean of a Gaussian over input features; per-feature
    log-variances are free parameters. This lets `beta=1` be a proper ELBO on
    continuous (e.g. z-scored log-FPKM) data and avoids the MSE/KL scale
    mismatch that can otherwise collapse the posterior.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers,
        latent_dim,
        norm="layer",
        nonlin="elu",
        variational=True,
        dropout=0.0,
        min_log_var_x=-8.0,
        max_log_var_x=4.0,
        conditional_dim=0,
        advesarial_dim=0,
        grl_alpha=1.0,
    ):
        super().__init__()

        self._init_kwargs = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "latent_dim": latent_dim,
            "norm": norm,
            "nonlin": nonlin,
            "variational": variational,
            "dropout": dropout,
            "min_log_var_x": min_log_var_x,
            "max_log_var_x": max_log_var_x,
            "conditional_dim": conditional_dim,
            "advesarial_dim": advesarial_dim,
            "grl_alpha": grl_alpha,
        }

        self.variational = variational
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.min_log_var_x = float(min_log_var_x)
        self.max_log_var_x = float(max_log_var_x)
        self.conditional_dim = conditional_dim
        self.advesarial_dim = advesarial_dim
        self.grl_alpha = grl_alpha
        
        nonlin_cls = get_nonlin(nonlin)
        norm_cls = get_norm(norm)

        self.encoder = MLP(
            in_channels=input_dim + conditional_dim,
            hidden_channels=hidden_dim,
            out_channels=latent_dim * 2,
            layers=n_layers,
            dropout=dropout,
            nonlin=nonlin_cls,
            norm=norm_cls,
            bias=True,
        )

        self.decoder = MLP(
            in_channels=latent_dim + conditional_dim,
            hidden_channels=hidden_dim,
            out_channels=input_dim,
            layers=n_layers,
            dropout=dropout,
            nonlin=nonlin_cls,
            norm=norm_cls,
            bias=True,
        )

        if self.variational:
            self.log_var_x = nn.Parameter(torch.zeros(input_dim))

        if self.advesarial_dim > 0:
            self.GRL = GradientReversalLayer(alpha=self.grl_alpha)
            self.clf = MLP(
                in_channels=latent_dim,
                hidden_channels=hidden_dim,
                out_channels=advesarial_dim,
                layers=n_layers,
                dropout=dropout,
                nonlin=nonlin_cls,
                norm=norm_cls,
                bias=True,
            )

    def _require_x_cond(self, x_cond):
        if self.conditional_dim > 0 and x_cond is None:
            raise ValueError(
                "x_cond is required when conditional_dim > 0 "
                f"(expected last dim {self.conditional_dim})"
            )

    def encode(self, x, x_cond=None):
        self._require_x_cond(x_cond)
        if x_cond is not None:
            x = torch.cat([x, x_cond], dim=-1)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_cond=None):
        self._require_x_cond(x_cond)
        if x_cond is not None:
            z = torch.cat([z, x_cond], dim=-1)
        return self.decoder(z)

    def predict(self, x, x_cond=None):
        """Deterministic reconstruction (use posterior mean)."""
        mu, _ = self.encode(x.view(-1, x.size(1)), x_cond=x_cond)
        return self.decode(mu, x_cond=x_cond)

    def forward(self, x, x_cond=None):
        mu, logvar = self.encode(x.view(-1, x.size(1)), x_cond=x_cond)

        if self.variational:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        xhat = self.decode(z, x_cond=x_cond)

        if self.advesarial_dim > 0:
            zstar = self.GRL(z)
            adv_pred = self.clf(zstar)

            return {"xhat": xhat, "mu": mu, "logvar": logvar, "adv_pred": adv_pred}

        return {"xhat": xhat, "mu": mu, "logvar": logvar}

    def loss(self, x, xhat, mu, logvar, beta=1.0, free_bits=0.0, **_):
        """ELBO with proper Gaussian likelihood.

        Returns a dict with:
            - loss: nll + beta * kld_clamped  (training objective)
            - nll: Gaussian negative log-likelihood (per-sample, mean over batch)
            - kld: analytic KL divergence (per-sample, mean over batch)
            - kld_raw: unclamped KL (for monitoring)
            - elbo: nll + kld (no beta, for model selection)
            - recon_mse: mean squared error (reported only)
        """
        x = x.view(-1, x.size(1))

        if self.variational:
            log_var_x = self.log_var_x.clamp(self.min_log_var_x, self.max_log_var_x)
            var_x = log_var_x.exp()
            nll_per_dim = 0.5 * ((x - xhat) ** 2 / var_x + log_var_x + LOG2PI)
            nll = nll_per_dim.sum(dim=1).mean()

            kld_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
            kld_raw = kld_per_dim.sum(dim=1).mean()
            if free_bits and free_bits > 0:
                kld_clamped = kld_per_dim.mean(dim=0).clamp(min=free_bits).sum()
            else:
                kld_clamped = kld_raw

            loss = nll + beta * kld_clamped
            elbo = nll + kld_raw
        else:
            nll = F.mse_loss(xhat, x, reduction="sum") / x.size(0)
            kld_raw = torch.zeros((), device=x.device, dtype=x.dtype)
            kld_clamped = kld_raw
            loss = nll
            elbo = nll

        recon_mse = F.mse_loss(xhat, x, reduction="mean")

        return {
            "loss": loss,
            "nll": nll,
            "kld": kld_clamped,
            "kld_raw": kld_raw,
            "elbo": elbo,
            "recon_mse": recon_mse,
        }

    def save(self, path):
        """Persist model as state_dict + init kwargs.

        Writes `<path>` with the torch state dict and, alongside it,
        `<path without .pt>_kwargs.json` containing the constructor kwargs.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "kwargs": self._init_kwargs},
            path,
        )
        kwargs_path = os.path.splitext(path)[0] + "_kwargs.json"
        with open(kwargs_path, "w") as f:
            json.dump(self._init_kwargs, f, indent=2)

    @classmethod
    def load(cls, path, map_location="cpu"):
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**blob["kwargs"])
        model.load_state_dict(blob["state_dict"])
        return model
