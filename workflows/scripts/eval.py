import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from amlvae.data.clin_cond import load_clin_cond
from amlvae.models.VAE import VAE


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--proc', type=str, required=True,
                           help='directory with <dataset>_expr.csv + <dataset>_partitions.pt')
    argparser.add_argument('--out', type=str, required=True,
                           help='output directory')
    argparser.add_argument('--model_path', type=str, required=True,
                           help='trained model.pt (state_dict + kwargs)')
    argparser.add_argument('--dataset', type=str, default='aml')
    return argparser.parse_args()


def eval_vae(model, X_train, X_val, X_test, device, C_train=None, C_val=None, C_test=None):
    def _predict(X, C):
        x_cond = C.to(device) if C is not None else None
        return model.predict(X.to(device), x_cond=x_cond).cpu()

    with torch.no_grad():
        xhat_train = _predict(X_train, C_train)
        xhat_val   = _predict(X_val, C_val)
        xhat_test  = _predict(X_test, C_test)

    def _metrics(X, xhat):
        return (
            r2_score(X.cpu().numpy(), xhat.cpu().numpy(), multioutput='variance_weighted'),
            F.mse_loss(X, xhat).item(),
        )

    r2_train, mse_train = _metrics(X_train, xhat_train)
    r2_val,   mse_val   = _metrics(X_val,   xhat_val)
    r2_test,  mse_test  = _metrics(X_test,  xhat_test)

    return pd.DataFrame({
        'model':     ['vae'],
        'r2_train':  [r2_train],
        'r2_val':    [r2_val],
        'r2_test':   [r2_test],
        'mse_train': [mse_train],
        'mse_val':   [mse_val],
        'mse_test':  [mse_test],
    })


def eval_pca(X_train, X_val, X_test, n_components):
    X_train = X_train.reshape(X_train.shape[0], -1).cpu().numpy()
    X_val   = X_val.reshape(X_val.shape[0], -1).cpu().numpy()
    X_test  = X_test.reshape(X_test.shape[0], -1).cpu().numpy()

    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    def _eval(X):
        xhat = pca.inverse_transform(pca.transform(X))
        return (
            r2_score(X, xhat, multioutput='variance_weighted'),
            ((xhat - X) ** 2).mean(),
        )

    r2_train, mse_train = _eval(X_train)
    r2_val,   mse_val   = _eval(X_val)
    r2_test,  mse_test  = _eval(X_test)

    return pd.DataFrame({
        'model':     ['pca'],
        'r2_train':  [r2_train],
        'r2_val':    [r2_val],
        'r2_test':   [r2_test],
        'mse_train': [mse_train],
        'mse_val':   [mse_val],
        'mse_test':  [mse_test],
    })


def load(args):
    data = pd.read_csv(f'{args.proc}/{args.dataset}_expr.csv')
    data = data.set_index(data.columns[0])
    partitions = torch.load(f'{args.proc}/{args.dataset}_partitions.pt', weights_only=False)

    X_train = torch.tensor(data.loc[partitions['train_ids'], :].values, dtype=torch.float32)
    X_val   = torch.tensor(data.loc[partitions['val_ids'], :].values,   dtype=torch.float32)
    X_test  = torch.tensor(data.loc[partitions['test_ids'], :].values,  dtype=torch.float32)

    return (
        X_train, X_val, X_test,
        partitions['train_ids'], partitions['val_ids'], partitions['test_ids'],
    )


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

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, X_val, X_test, train_ids, val_ids, test_ids = load(args)
    model = VAE.load(args.model_path, map_location=device).to(device).eval()

    C_train = C_val = C_test = None
    if model.conditional_dim > 0:
        C_train = load_clin_cond(args.proc, args.dataset, train_ids)
        C_val = load_clin_cond(args.proc, args.dataset, val_ids)
        C_test = load_clin_cond(args.proc, args.dataset, test_ids)

    vae_res = eval_vae(
        model, X_train, X_val, X_test, device, C_train, C_val, C_test
    )
    pca_res = eval_pca(X_train, X_val, X_test, n_components=model.latent_dim)
    eval_res = pd.concat([vae_res, pca_res], axis=0)
    eval_res = eval_res.assign(path=args.model_path)
    eval_res.to_csv(f'{args.out}/eval.csv', index=False)

    def _encode_mu(X, C):
        x_cond = C.to(device) if C is not None else None
        return model.encode(X.to(device), x_cond=x_cond)[0].cpu()

    with torch.no_grad():
        z_train = _encode_mu(X_train, C_train)
        z_val   = _encode_mu(X_val, C_val)
        z_test  = _encode_mu(X_test, C_test)

    z = pd.DataFrame(
        np.concatenate([z_train, z_val, z_test], axis=0),
        index=np.concatenate([train_ids, val_ids, test_ids], axis=0),
        columns=[f'z{i+1}' for i in range(model.latent_dim)],
    )
    z.to_csv(f'{args.out}/{args.dataset}_z.csv')

    print()
    print('---------------------------------------------')
    print('Evaluation results:')
    print(eval_res)
    print('---------------------------------------------')
    print('evaluation complete.')
    print('---------------------------------------------')
    print()
