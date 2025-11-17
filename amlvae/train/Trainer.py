

import torch 
import numpy as np 
from sklearn.metrics import r2_score
import math 
import pandas as pd 
from amlvae.models.VAE import VAE
import tempfile
import time

try: 
    # versioning issues 
    from ray.tune import Checkpoint
except: 
    from ray.train import Checkpoint

from ray import tune
import os

def freeze_(model):
    for param in model.parameters():
        param.requires_grad = False 

def unfreeze_(model):
    for param in model.parameters():
        param.requires_grad = True

def check_convergence(
    losses: list[float],
    patience: int = 5,           # steps with no improvement → stop
    min_delta: float = 1e-4,     # “improvement” means drop ≥ min_delta
    max_steps: int = 50          # hard safety cap
) -> bool:
    """
    Return True when training has converged.

    Converged ⇔
    • Best loss hasn't improved by ≥ min_delta for `patience` steps, OR
    • We already took `max_steps` steps (safety guard).
    """
    n = len(losses)
    if n == 0:
        return False            # nothing to judge yet

    # --- Hard upper-bound on iterations -------------------------------------
    if n >= max_steps:
        return True

    # --- Patience rule -------------------------------------------------------
    best_so_far = np.min(losses)
    best_step  = np.argmin(losses)

    # Has the best loss improved in the last `patience` iterations?
    steps_since_best = n - 1 - best_step
    if steps_since_best < patience:
        return False            # still improving frequently enough

    # Even if we haven’t beaten the *absolute* best lately,
    # check if the *recent* improvement is ≥ min_delta.
    recent_window = losses[-patience-1:-1]   # last `patience` old points
    if np.min(recent_window) - losses[-1] >= min_delta:
        return False            # we *did* improve in that span

    return True                 # no meaningful drop for `patience` steps


class Trainer(): 

    def __init__(self, root, dataset_name='aml', checkpoint=False, log_every=250, epochs=500, verbose=False, patience=100, return_best_model=False): 

        data = pd.read_csv(f'{root}/{dataset_name}_expr.csv')
        data = data.set_index(data.columns[0])
        partitions = torch.load(f'{root}/{dataset_name}_partitions.pt', weights_only=False)
        self.X_train = torch.tensor( 
            data.loc[partitions['train_ids'], :].values, dtype=torch.float32
        )
        self.X_val = torch.tensor(
            data.loc[partitions['val_ids'], :].values, dtype=torch.float32
        )
        self.X_test = torch.tensor(
            data.loc[partitions['test_ids'], :].values, dtype=torch.float32
        )

        self.checkpoint = checkpoint
        self.epochs = epochs
        self.log_every = log_every
        self.verbose = verbose
        self.patience = patience
        self.return_best_model = return_best_model


    def train_epoch(self, model, optim, batch_size, device, beta): 
        
        model.train()
        for ixs in torch.split(torch.randperm(len(self.X_train)), batch_size):

            optim.zero_grad()
            x = self.X_train[ixs]
            out = model(x)
            loss, nll, kld = model.loss(beta=beta, **out)

            loss.backward()
            optim.step()
        

    def eval(self, model, device, partition='val'): 

        if partition == 'train':
            X = self.X_train
        elif partition == 'val':
            X = self.X_val
        elif partition == 'test':
            X = self.X_test
        else:
            raise ValueError(f'Unknown partition: {partition}')
        
        model.eval() 
        with torch.no_grad():
            out = model(X.to(device))
            # total_loss, recon_loss, kld, lm
            metrics = model.eval_(X.to(device), out['xhat'], out['mu'], out['logvar'], out['nll'])

        return metrics

    def __call__(self, config):

        model_kwargs = {
            'input_dim'   : self.X_train.size(1),
            'hidden_dim'  : config['n_hidden'],
            'n_layers'    : config['n_layers'],
            'latent_dim'  : config['n_latent'],
            'norm'        : config['norm'],
            'dropout'     : config['dropout'],
            'nonlin'      : config['nonlin'],
            'norm_first'  : config['norm_first'],
        }

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = VAE(**model_kwargs).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['l2'])

        self.X_train = self.X_train.to(device)
        self.X_val = self.X_val.to(device)
        self.X_test = self.X_test.to(device)

        best_elbo = float('inf')
        patience_count = 0 
        best_model = None 

        T = int(self.epochs*0.75)
        for epoch in range(self.epochs): 

            if config['anneal']: 
                if epoch < T:
                    fraction = epoch / T
                    beta = 0.5 * (1 - math.cos(fraction * math.pi)) * config['beta']
                else:
                    beta = config['beta'] 
            else: 
                beta = config['beta']

            self.train_epoch(
                model, optim, config['batch_size'], device, beta
            )

            val_metrics = self.eval(
                model, device, partition='val'
            )

            if val_metrics['elbo'] < best_elbo:
                best_elbo = val_metrics['elbo']
                best_model = {k:v.detach().cpu() for k,v in model.state_dict().items()}
                patience_count = 0

            if self.verbose: 
                print(f'epoch: {epoch}, val mse: {val_metrics['MSE']:.4f}, val r2: {val_metrics['r2']:.2f}, kld: {val_metrics['kld']:.2f}, nll: {val_metrics['nll']:.2f}, elbo: {val_metrics['elbo']:.2f}, beta: {beta:.2E}', end='\r')

            if patience_count > self.patience:
                break             


        if self.return_best_model:
            model.load_state_dict(best_model)
            return model
        else: 
            return model
        