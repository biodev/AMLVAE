import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score

def get_nonlin(name):
    """Dummy nonlinearity factory (replace with your own)."""
    if name == 'elu':
        return nn.ELU
    elif name == 'relu':
        return nn.ReLU
    elif name == 'leakyrelu':
        return nn.LeakyReLU
    # add more if you want
    return nn.ReLU  # default

def get_norm(name):
    """Dummy normalization factory (replace with your own)."""
    if name == 'layer':
        return nn.LayerNorm
    elif name == 'batch':
        return nn.BatchNorm1d
    return None  # no normalization

class VectorQuantizer(nn.Module):
    """
    Basic vector quantization module.
    - codebook: an Embedding table [num_embeddings, embedding_dim]
    - Straight-through or simple nearest-neighbor quantization
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook of embeddings: shape [num_embeddings, embedding_dim]
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize embeddings
        nn.init.uniform_(self.codebook.weight, -1, 1)

    def forward(self, inputs):
        """
        inputs: (batch_size, embedding_dim)
        returns:
          quantized: (batch_size, embedding_dim) - the quantized vectors
          vq_loss: scalar for commitment/codebook loss
          indices: (batch_size,) indices of chosen codebook vectors
        """
        # Flatten if needed (in case of multiple dims). 
        # For simplicity, assume inputs is already [B, embedding_dim].
        # Compute distances to codebook vectors
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                     - 2 * torch.matmul(inputs, self.codebook.weight.t())
                     + torch.sum(self.codebook.weight**2, dim=1))
        # Get indices of nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantize
        quantized = self.codebook(encoding_indices)

        # Compute VQ losses:
        # 1) Commitment loss (how far inputs is from quantized)
        # 2) Codebook loss (optional: in some variants you also measure how far codebook is from inputs)
        # For simplicity, use a single term: ||sg(inputs) - quantized||^2 + commitment_cost*||inputs - sg(quantized)||^2
        # A simpler approach: MSE(inputs, quantized). We'll just do the “commitment” part.
        # Stop gradient from quantized back to encoder (straight-through).
        vq_loss = F.mse_loss(inputs.detach(), quantized) + self.commitment_cost * F.mse_loss(inputs, quantized.detach())

        # Straight-through estimator: we want gradient w.r.t. decoder to flow, but not w.r.t. the codebook in this path
        quantized_st = inputs + (quantized - inputs).detach()  # i.e. the "copy" trick

        return quantized_st, vq_loss, encoding_indices


class VQVAE(nn.Module):
    """
    Minimal Vector-Quantized VAE in a style similar to your standard VAE.
    """
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 latent_dim,       # embedding_dim in VQ context
                 num_embeddings,   # size of the codebook
                 norm='layer',
                 nonlin='elu',
                 commitment_cost=0.25):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        # Activation + Normalization layers
        nonlin_layer = get_nonlin(nonlin)
        norm_layer = get_norm(norm)

        # ----------------------
        # Encoder
        # ----------------------
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            if norm_layer:
                encoder_layers.append(norm_layer(h_dim))
            encoder_layers.append(nonlin_layer())
            prev_dim = h_dim
        # final layer to produce embeddings of size 'latent_dim'
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ----------------------
        # Vector Quantizer
        # ----------------------
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost=commitment_cost)

        # ----------------------
        # Decoder
        # ----------------------
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            if norm_layer:
                decoder_layers.append(norm_layer(h_dim))
            decoder_layers.append(nonlin_layer())
            prev_dim = h_dim
        # final layer to reconstruct input_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """
        Maps input x -> continuous embedding e -> quantized embedding q
        """
        e = self.encoder(x)
        return e  # continuous embedding before quantization

    def quantize(self, e):
        """
        Pass embedding e through VectorQuantizer
        """
        q, vq_loss, indices = self.vq(e)
        return q, vq_loss, indices

    def decode(self, q):
        """
        Reconstruct from quantized embedding
        """
        return self.decoder(q)

    def forward(self, x):
        """
        Full forward pass:
          1) encode -> e
          2) quantize e -> q
          3) decode q -> xhat
        """
        e = self.encode(x.view(-1, x.size(1)))
        q, vq_loss, indices = self.quantize(e)
        xhat = self.decode(q)
        return {'xhat': xhat, 'vq_loss': vq_loss, 'indices': indices}

    def loss(self, x, xhat, vq_loss, beta=1.0, **kwargs):
        """
        VQ-VAE total loss = reconstruction loss + beta * VQ loss.
        Typically, VQ loss includes the commitment and codebook losses.
        """
        recon_loss = F.mse_loss(xhat, x.view(-1, x.size(1)), reduction='mean')
        # Optionally you could do sum over batch, but mean is typical
        return recon_loss + beta * vq_loss

    def eval_(self, x, xhat):
        """
        Like your existing eval function, we can compute MSE, correlation, etc.
        """
        mse_val = F.mse_loss(xhat, x.view(-1, x.size(1)), reduction='mean')
        
        # Pearson correlation across features
        r_vals = []
        x_np = x.detach().cpu().numpy()
        xhat_np = xhat.detach().cpu().numpy()
        for i in range(xhat.size(1)):
            corr = np.corrcoef(x_np[:, i], xhat_np[:, i])[0, 1]
            r_vals.append(corr)
        r_avg = np.mean(r_vals)

        r2_val = r2_score(x_np, xhat_np, multioutput='uniform_average')

        return {
            'MSE': mse_val,
            'r': r_avg,
            'r2': r2_val
        }
