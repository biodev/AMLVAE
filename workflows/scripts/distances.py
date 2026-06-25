import argparse
import os

import pandas as pd
import torch

from amlvae.data.clin_cond import load_clin_cond
from amlvae.models.VAE import VAE


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--proc', type=str, required=True,
                           help='directory containing <dataset>_expr.csv')
    argparser.add_argument('--out', type=str, required=True,
                           help='output directory')
    argparser.add_argument('--model_path', type=str, required=True,
                           help='path to trained model.pt (state_dict + kwargs)')
    argparser.add_argument('--dataset', type=str, default='aml',
                           help='dataset prefix')
    return argparser.parse_args()


def load(args):
    data = pd.read_csv(f'{args.proc}/{args.dataset}_expr.csv')
    data = data.set_index(data.columns[0])
    X = torch.tensor(data.values, dtype=torch.float32)
    return X, data.index


if __name__ == '__main__':
    print()
    print('---------------------------------------------')
    print('VAE: Encoding latent embeddings')
    print('---------------------------------------------')
    print()
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.out, exist_ok=True)

    X, ids = load(args)
    model = VAE.load(args.model_path, map_location=device).to(device).eval()

    x_cond = None
    if model.conditional_dim > 0:
        x_cond = load_clin_cond(args.proc, args.dataset, ids).to(device)

    with torch.no_grad():
        z = model.encode(X.to(device), x_cond=x_cond)[0].cpu().numpy()

    z_df = pd.DataFrame(z, index=ids, columns=[f'z{i+1}' for i in range(z.shape[1])])
    z_df.to_csv(f'{args.out}/{args.dataset}_z.csv', index=True, header=True)

    print(f'Saved latent embeddings to {args.out}/{args.dataset}_z.csv')
