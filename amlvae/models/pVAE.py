import torch 
from torch import nn
import torch.nn.functional as F
from amlvae.models.utils import get_nonlin, get_norm
import numpy as np 
from sklearn.metrics import r2_score
from amlvae.models.VAE import VAE 


class pVAE(VAE):
    """Poisson VAE."""
    def __init__(self, input_dim, hidden_dims, latent_dim, norm='layer', nonlin='elu'):
        super().__init__(input_dim, hidden_dims, latent_dim, norm=norm, nonlin=nonlin)
        
    def encode(self, x):
        x = torch.log2(x + 1)
        h = self.encoder(x)
        return self.fc21(h), self.fc22(h)

    def decode(self, z):

        return 2**self.decoder(z)

    def loss(self, x, xhat, mu, logvar, beta=1):
        """
        x:     (batch_size, n_genes) raw counts (integers)
        xhat:  (batch_size, n_genes) the predicted Poisson rate (lambda)
        mu, logvar: from the encoder
        beta:  weighting factor on KLD (beta-VAE)
        """

        # assume xhat is the Poisson rate (lambda) and is positive

        # Poisson negative log-likelihood (ignoring log(x!))
        # Sum over genes/features, then average over batch.
        NLL = torch.sum(xhat - x * torch.log(xhat), dim=1).mean()

        # KL divergence term
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        return NLL + beta * KLD

    def eval_(self, x, xhat):
        """
        For evaluation/monitoring, you can still compute MSE or correlation
        even though you're training with Poisson NLL.
        """
        # We can do a simple MSE just to see how 'close' the predicted rates are to actual counts.
        mse_val = F.mse_loss(xhat, x.view(-1, x.size(1)), reduction='mean')
        
        # Pearson correlation across genes, averaged
        r_vals = []
        x_np = x.detach().cpu().numpy()
        xhat_np = xhat.detach().cpu().numpy()
        for i in range(xhat.size(1)):
            # correlation of the i-th gene across the batch
            corr = np.corrcoef(x_np[:, i], xhat_np[:, i])[0, 1]
            r_vals.append(corr)
        r_avg = np.mean(r_vals)

        # R^2 score (multi-output)
        r2_val = r2_score(x_np, xhat_np, multioutput='uniform_average')

        return {
            'MSE': mse_val,
            'r': r_avg,
            'r2': r2_val
        }