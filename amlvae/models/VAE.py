import torch 
from torch import nn
import torch.nn.functional as F
from amlvae.models.utils import get_nonlin, get_norm
import numpy as np 
from sklearn.metrics import r2_score
from amlvae.models.MLP import MLP
from scvi.distributions import NegativeBinomial

    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, latent_dim, 
                 norm='layer', nonlin='elu',
                 dropout=0., norm_first=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.latent_dim = latent_dim 

        nonlin = get_nonlin(nonlin)
        norm = get_norm(norm)

        self.encoder = MLP(in_channels      = input_dim,
                           hidden_channels  = hidden_dim, 
                           out_channels     = latent_dim*2,
                           layers           = n_layers,
                           norm             = norm,
                           dropout          = 0, 
                           nonlin           = nonlin, 
                           bias             = True,
                           norm_first       = norm_first)
        
        self.decoder = MLP(in_channels      = latent_dim,
                           hidden_channels  = hidden_dim, 
                           out_channels     = input_dim*2, # mu, theta
                           layers           = n_layers,
                           norm             = norm,
                           dropout          = dropout, 
                           nonlin           = nonlin, 
                           bias             = True,
                           norm_first       = norm_first)
    
        
    def encode(self, x):

        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1) 
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5 * logvar)
        P = torch.distributions.Normal(mu, std)
        return P.rsample(), P
    
    def decode(self, z):
        return self.decoder(z)

    
    def predict(self, x): 
        # for reconstrunction evaluation 
        mu, logvar = self.encode(x.view(-1, x.size(1)))
        out = self.decode(mu)
        n_mu, n_var_ = torch.chunk(out, 2, dim=-1) 

        n_var = torch.nn.functional.softplus(n_var_) + 1e-2

        P = torch.distributions.Normal(n_mu, n_var)
        return P.mean

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.size(1)))
        
        z, P_z = self.reparameterize(mu, logvar)

        out = self.decode(z) 

        n_mu, n_var_ = torch.chunk(out, 2, dim=-1) 

        n_var = torch.nn.functional.softplus(n_var_) + 1e-2

        P = torch.distributions.Normal(n_mu, n_var)

        return {'xhat': P.mean, 'mu': mu, 'logvar': logvar, 'P': P, 'nll': -P.log_prob(x).mean(), 'P_z': P_z}

    @staticmethod
    def loss(P_z, nll, beta=1., **kwargs):
        """
        Computes the VAE loss = recon_loss + KL_divergence.
        """
        B = P_z.loc.size(0)

        # KL for gene latent
        Q = torch.distributions.Normal(torch.zeros_like(P_z.loc), torch.ones_like(P_z.scale)) # prior
        kld = torch.distributions.kl.kl_divergence(P_z, Q).sum() / B

        total_loss = nll + beta*kld 
        return total_loss, nll, kld

    @staticmethod 
    def eval_posterior_collapse(mu, logvar, t=0.1):
        std = torch.exp(0.5 * logvar)
        P = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1) # posterior
        Q = torch.distributions.Independent(torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std)), 1) # prior
        kld = torch.distributions.kl.kl_divergence(P, Q).sum(0) / mu.size(0)
        return (kld > t).mean().item()

    @staticmethod
    def eval_(x, xhat, mu, logvar, nll, beta=1., **kwargs):

        B = mu.size(0)

        # KL for gene latent
        std = torch.exp(0.5 * logvar)
        P = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1) # posterior
        Q = torch.distributions.Independent(torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std)), 1) # prior
        kld = torch.distributions.kl.kl_divergence(P, Q).sum() / B

        eval_dict = {'MSE': F.mse_loss(xhat, x.view(-1, x.size(1)), reduction='mean'),
                     'r2': r2_score(x.detach().cpu().numpy(), xhat.detach().cpu().numpy(), multioutput='uniform_average'),
                     'nll': nll.item(),
                     'elbo': nll.item() + beta*kld.item(),
                     'kld': kld.item()}
    
        return eval_dict
