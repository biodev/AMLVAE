import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sbn
import numpy as np 

import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
import umap
from sklearn.decomposition import PCA
from amlvae.models.VAE import VAE 

import argparse 


def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--data', type=str, default='../../data/', 
                           help='path to data dir')
    argparser.add_argument('--proc', type=str, default='../proc/',
                            help='path to proc dir')
    argparser.add_argument('--out', type=str, default='../output/',
                            help='path to output dir')
    argparser.add_argument('--model_path', type=str, default=None,
                            help='path to model dir')
    argparser.add_argument('--dataset', type=str, default='aml',
                            help='dataset name')

    return argparser.parse_args()


def eval_vae(model, X_train, X_val, X_test): 

    with torch.no_grad(): xhat = model.predict(X_test.to('cuda')).cpu()
    r2_test = r2_score(X_test.cpu().numpy(), xhat.cpu().numpy(), multioutput='variance_weighted')
    mse_test = F.mse_loss(X_test, xhat).item()

    with torch.no_grad(): xhat = model.predict(X_train.to('cuda')).cpu()
    r2_train = r2_score(X_train.cpu().numpy(), xhat.cpu().numpy(), multioutput='variance_weighted')
    mse_train = F.mse_loss(X_train, xhat).item()

    with torch.no_grad(): xhat = model.predict(X_val.to('cuda')).cpu()
    r2_val = r2_score(X_val.cpu().numpy(), xhat.cpu().numpy(), multioutput='variance_weighted')
    mse_val = F.mse_loss(X_val, xhat).item()

    res = pd.DataFrame({'model':['vae'], 'r2_train':[r2_train], 'r2_val':[r2_val], 'r2_test':[r2_test],
           'mse_train':[mse_train], 'mse_val':[mse_val], 'mse_test':[mse_test]})
    
    return res 

def eval_pca(X_train, X_val, X_test, n_components): 

    # ensure proper shapes 
    X_train = X_train.reshape(X_train.shape[0], -1) 
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(X_train.cpu().numpy())
    xhat = pca.inverse_transform(pc) 

    mse_train = (
        (xhat - X_train.cpu().numpy())**2
    ).mean()
    r2_train = r2_score(X_train.cpu().numpy(), xhat, multioutput='variance_weighted')

    xhat = pca.inverse_transform(pca.transform(X_test.cpu().numpy()))
    mse_test = (
        (xhat - X_test.cpu().numpy())**2
    ).mean()
    r2_test = r2_score(X_test.cpu().numpy(), xhat, multioutput='variance_weighted')

    xhat = pca.inverse_transform(pca.transform(X_val.cpu().numpy()))
    mse_val = (
        (xhat - X_val.cpu().numpy())**2
    ).mean()
    r2_val = r2_score(X_val.cpu().numpy(), xhat, multioutput='variance_weighted')

    res = pd.DataFrame({'model':['pca'], 'r2_train':[r2_train], 'r2_val':[r2_val], 'r2_test':[r2_test],
           'mse_train':[mse_train], 'mse_val':[mse_val], 'mse_test':[mse_test]}) 
    
    return res 

def load(args): 

    data = pd.read_csv(f'{args.proc}/{args.dataset}_expr.csv')
    data = data.set_index(data.columns[0])
    partitions = torch.load(f'{args.proc}/{args.dataset}_partitions.pt', weights_only=False)
    X_train = torch.tensor( 
        data.loc[partitions['train_ids'], :].values, dtype=torch.float32
    )
    X_val = torch.tensor(
        data.loc[partitions['val_ids'], :].values, dtype=torch.float32
    )
    X_test = torch.tensor(
        data.loc[partitions['test_ids'], :].values, dtype=torch.float32
    )

    return X_train, X_val, X_test


if __name__ == '__main__': 

    print()
    print('---------------------------------------------')
    print('VAE: Evaluation')
    print('---------------------------------------------')
    print() 
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')
    
    X_train, X_val, X_test = load(args) 
    model = torch.load(args.model_path, weights_only=False, map_location='cuda')
    model = model.eval()

    vae_res = eval_vae(model, X_train, X_val, X_test)
    pca_res = eval_pca(X_train, X_val, X_test, n_components=model.latent_dim)
    eval_res = pd.concat([vae_res, pca_res], axis=0) 
    eval_res = eval_res.assign(path=args.model_path)
    eval_res.to_csv(f'{args.out}/eval.csv', index=False)

    print() 
    print('---------------------------------------------')
    print('Evaluation results:')
    print(eval_res)
    print('---------------------------------------------')

    print('evaluation complete.')
    print('---------------------------------------------')
    print()