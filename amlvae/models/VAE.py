import torch 
from torch import nn
import torch.nn.functional as F
from amlvae.models.utils import get_nonlin, get_norm
import numpy as np 
from sklearn.metrics import r2_score

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, norm='layer', nonlin='elu'):
        super().__init__()

        nonlin = get_nonlin(nonlin)
        norm_layer = get_norm(norm)
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            if norm_layer:
                encoder_layers.append(norm_layer(h_dim))
            encoder_layers.append(nonlin())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc21 = nn.Linear(hidden_dims[-1], latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dims[-1], latent_dim)  # Log variance
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            if norm_layer:
                decoder_layers.append(norm_layer(h_dim))
            decoder_layers.append(nonlin())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc21(h), self.fc22(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.size(1)))
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return {'xhat': out, 'mu': mu, 'logvar': logvar}

    def loss(self, x, xhat, mu, logvar, beta=1):
        MSE = F.mse_loss(xhat, x.view(-1, x.size(1)), reduction='sum') / x.size(0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return MSE + beta*KLD
    
    def eval_(self, x, xhat):

        eval_dict = {'MSE': F.mse_loss(xhat, x.view(-1, x.size(1)), reduction='mean'),
                     'r': np.mean([np.corrcoef(x[:, i].detach().cpu().numpy(), xhat[:,i].detach().cpu().numpy()) for i in range(xhat.size(1))]),
                     'r2': r2_score(x.detach().cpu().numpy(), xhat.detach().cpu().numpy(), multioutput='uniform_average')}
    
        return eval_dict
