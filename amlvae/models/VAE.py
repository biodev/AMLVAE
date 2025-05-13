import torch 
from torch import nn
import torch.nn.functional as F
from amlvae.models.utils import get_nonlin, get_norm
import numpy as np 
from sklearn.metrics import r2_score
from amlvae.models.MLP import MLP

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        # Save alpha (scale factor) for backward
        ctx.alpha = alpha
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient by multiplying with -alpha
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        # We call our custom autograd Function
        return GradientReverseFunction.apply(x, self.alpha)
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, latent_dim, conditions={}, 
                 norm='layer', nonlin='elu', variational=True,
                 dropout=0.):
        super().__init__()

        self.variational = variational
        self.latent_dim = latent_dim 

        nonlin = get_nonlin(nonlin)
        norm_layer = get_norm(norm)

        if len(conditions) > 0: 
            mlp = lambda d: torch.nn.Sequential(nn.Linear(latent_dim, latent_dim*4), nonlin(), norm_layer(latent_dim*4), nn.Linear(latent_dim*4, d))
            self.adv_dict = torch.nn.ModuleDict({key: mlp(cond_dim) for key, cond_dim in conditions.items()})
        
        self.encoder = MLP(in_channels      = input_dim,
                           hidden_channels  = hidden_dim, 
                           out_channels     = latent_dim*2,
                           layers           = n_layers,
                           dropout          = 0, 
                           nonlin           = nonlin, 
                           bias             = True)
        
        self.decoder = MLP(in_channels      = latent_dim,
                           hidden_channels  = hidden_dim, 
                           out_channels     = input_dim,
                           layers           = n_layers,
                           dropout          = dropout, 
                           nonlin           = nonlin, 
                           bias             = True)
        
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1) 
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    @DeprecationWarning 
    def adversarial(self, z):

        # perform gradient reversal layer (backward gradients will be reversed)
        z = self.GRL(z)
        adv_preds = {key: self.adv_dict[key](z) for key in self.adv_dict.keys()}
        return adv_preds
    
    def predict(self, x): 
        # for reconstrunction evaluation 
        mu, logvar = self.encode(x.view(-1, x.size(1)))
        out = self.decode(mu)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.size(1)))
        z = self.reparameterize(mu, logvar)

        if self.variational: 
            out = self.decode(z)
        else: 
            out = self.decode(mu)

        if hasattr(self, 'adv_dict'):
            adv_preds = self.adversarial(mu)
        else: 
            adv_preds = {} 

        return {**{'xhat': out, 'mu': mu, 'logvar': logvar}, **adv_preds}

    @staticmethod
    def loss(x, xhat, mu, logvar, beta=1., **kwargs):
        """
        Computes the VAE loss = recon_loss + KL_divergence.
        """
        # Reconstruction loss (MSE here; choose appropriate loss for gene expression)
        recon_loss = F.mse_loss(xhat, x, reduction='sum') / x.size(0)

        std = torch.exp(0.5 * logvar)

        # KL for gene latent
        P = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1) # posterior
        Q = torch.distributions.Independent(torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std)), 1) # prior
        kld = torch.distributions.kl.kl_divergence(P, Q).sum() / x.size(0)
                                                                                      
        total_loss = recon_loss + beta*kld 
        return total_loss, recon_loss, kld #, recon_loss, kld

    def eval_(self, x, xhat):

        eval_dict = {'MSE': F.mse_loss(xhat, x.view(-1, x.size(1)), reduction='mean'),
                     'r': np.mean([np.corrcoef(x[:, i].detach().cpu().numpy(), xhat[:,i].detach().cpu().numpy()) for i in range(xhat.size(1))]),
                     'r2': r2_score(x.detach().cpu().numpy(), xhat.detach().cpu().numpy(), multioutput='uniform_average')}
    
        return eval_dict
