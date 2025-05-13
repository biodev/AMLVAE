import torch 
from torch import nn
import torch.nn.functional as F
from amlvae.models.utils import get_nonlin, get_norm
import numpy as np 
from sklearn.metrics import r2_score


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
    


class ZeroDiagonalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_matrix):
        # We just pass the data forward unchanged
        return input_matrix.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Make a copy of the gradient
        grad = grad_output.clone()
        
        # For a square matrix NxN: zero-out the diagonal
        n = grad.shape[0]
        # If your matrix isn't guaranteed to be square, you can do:
        # m = min(grad.shape[0], grad.shape[1])
        # grad[range(m), range(m)] = 0
        
        grad[range(n), range(n)] = 0
        
        # Return the modified gradient
        return grad

class ZeroDiagonalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ZeroDiagonalFunction.apply(x)
    

class advAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, norm='layer', nonlin='elu', alpha=1., dropout=0.):
        super().__init__()

        nonlin = get_nonlin(nonlin)
        norm_layer = get_norm(norm)

        self.GRL = GradientReversalLayer(alpha=alpha)
        self.ZDL = ZeroDiagonalLayer()
        W = torch.randn(latent_dim, latent_dim)
        # set the diagonal of W to 0 
        W = W - torch.diag(torch.diag(W))
        self.W = nn.Parameter(W)
        self.dropout = nn.Dropout(dropout)

        
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
        return self.fc21(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def adversarial(self, z):

        # perform gradient reversal layer (backward gradients will be reversed)
        z = self.GRL(z)
        adv_preds = {key: self.adv_dict[key](z) for key in self.adv_dict.keys()}
        return adv_preds

    
    def forward(self, x):
        z = self.encode(x.view(-1, x.size(1)))
        out = self.decode(z)

        zstar = self.GRL(z)
        zhat = torch.mm(zstar, self.dropout(self.ZDL(self.W)))
        adv_loss = F.mse_loss(zhat, z)

        return {'xhat': out, 'z': z, 'adv_loss': adv_loss}

    def loss(self, x, xhat, mu=None, logvar=None, beta=1., adv_loss=None, **kwargs):
        MSE = F.mse_loss(xhat, x.view(-1, x.size(1)), reduction='sum') / x.size(0)
        return MSE + beta*adv_loss
    
    def eval_(self, x, xhat):

        eval_dict = {'MSE': F.mse_loss(xhat, x.view(-1, x.size(1)), reduction='mean'),
                     'r': np.mean([np.corrcoef(x[:, i].detach().cpu().numpy(), xhat[:,i].detach().cpu().numpy()) for i in range(xhat.size(1))]),
                     'r2': r2_score(x.detach().cpu().numpy(), xhat.detach().cpu().numpy(), multioutput='uniform_average')}
    
        return eval_dict
