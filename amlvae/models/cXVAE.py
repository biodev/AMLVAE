'''
Implementation of: 

Zuqi Li, Sonja Katz, Edoardo Saccenti, David W Fardo, Peter Claes, Vitor A P Martins dos Santos, Kristel Van Steen, 
Gennady V Roshchupkin, Novel multi-omics deconfounding variational autoencoders can obtain meaningful disease subtyping, 
Briefings in Bioinformatics, Volume 25, Issue 6, November 2024, bbae512, https://doi.org/10.1093/bib/bbae512
'''

import torch 
from torch import nn
import torch.nn.functional as F
from amlvae.models.utils import get_nonlin, get_norm
import numpy as np 
from sklearn.metrics import r2_score

def _make_encoder(input_dim, cond_dim, hidden_dims, norm_layer, nonlin):
    encoder_layers = []
    prev_dim = input_dim + cond_dim
    for h_dim in hidden_dims:
        encoder_layers.append(nn.Linear(prev_dim, h_dim))
        if norm_layer:
            encoder_layers.append(norm_layer(h_dim))
        encoder_layers.append(nonlin())
        prev_dim = h_dim
    return nn.Sequential(*encoder_layers)

def _make_decoder(hidden_dims, cond_dim, latent_dim, input_dim, norm_layer, nonlin):
    decoder_layers = []
    prev_dim = latent_dim + cond_dim
    for h_dim in reversed(hidden_dims):
        decoder_layers.append(nn.Linear(prev_dim, h_dim))
        if norm_layer:
            decoder_layers.append(norm_layer(h_dim))
        decoder_layers.append(nonlin())
        prev_dim = h_dim
    decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
    return nn.Sequential(*decoder_layers)



class cXVAE(nn.Module):
    def __init__(self, input_dim_dict, loss_dict, eval_dict, cond_dim, hidden_dims, latent_dim, norm='layer', nonlin='elu'):
        super().__init__()

        nonlin = get_nonlin(nonlin)
        norm_layer = get_norm(norm)
        self.loss_dict = loss_dict
        self.eval_dict = eval_dict

        encoder_dict = {}
        for key, input_dim in input_dim_dict.items():
            encoder_dict[key] = _make_encoder(input_dim, cond_dim, hidden_dims, norm_layer, nonlin)
        self.encoder_dict = torch.nn.ModuleDict(encoder_dict)

        self.fc21 = nn.Linear(hidden_dims[-1], latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dims[-1], latent_dim)  # Log variance

        decoder_dict = {}
        for key, input_dim in input_dim_dict.items():
            decoder_dict[key] = _make_decoder(hidden_dims, cond_dim, latent_dim, input_dim, norm_layer, nonlin)
        self.decoder_dict = torch.nn.ModuleDict(decoder_dict)

    def encode(self, x, c, key):
        h = self.encoder_dict[key](torch.cat([x, c], dim=1))
        return self.fc21(h), self.fc22(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c, key):
        return self.decoder_dict[key](torch.cat([z, c], dim=1))
    
    def forward(self, x, c, key):
        mu, logvar = self.encode(x, c, key)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, c, key)
        return {'xhat': out, 'mu': mu, 'logvar': logvar, 'key': key}

    def loss(self, x, xhat, mu, logvar, key, beta=1):

        recon_loss = self.loss_dict[key](x, xhat)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta*KLD
    
    def eval_(self, x, xhat, key):

        eval_dict = self.eval_dict[key](x, xhat)

        return eval_dict
