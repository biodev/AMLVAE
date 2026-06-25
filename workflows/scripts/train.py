import argparse
import os

import torch

from amlvae.train.Trainer import Trainer


def _parse_bool(name, value):
    if isinstance(value, bool):
        return value
    if value in ('true', 'True', 'TRUE', '1'):
        return True
    if value in ('false', 'False', 'FALSE', '0'):
        return False
    raise ValueError(f'Unknown value for {name}: {value}')


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--proc', type=str, required=True,
                           help='directory containing <dataset_name>_expr.csv + partitions')
    argparser.add_argument('--out', type=str, required=True,
                           help='directory to write model.pt into')
    argparser.add_argument('--dataset_name', type=str, default='aml',
                           help='dataset name prefix in proc dir')
    argparser.add_argument('--epochs', type=int, default=1000)
    argparser.add_argument('--patience', type=int, default=1000)
    argparser.add_argument('--n_hidden', type=int, default=512)
    argparser.add_argument('--n_latent', type=int, default=12)
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--norm', type=str, default='layer')
    argparser.add_argument('--variational', type=str, default='true')
    argparser.add_argument('--anneal', type=str, default='true')
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--nonlin', type=str, default='elu')
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--l2', type=float, default=0.0)
    argparser.add_argument('--beta', type=float, default=1.0)
    argparser.add_argument('--batch_size', type=int, default=256)
    argparser.add_argument('--free_bits', type=float, default=0.0,
                           help='per-dimension KL free-bits floor (0 disables)')
    argparser.add_argument('--model_selection_metric', type=str, default='recon_mse',
                           choices=['recon_mse', 'nll', 'elbo'],
                           help='validation metric used for best-model selection')
    argparser.add_argument('--conditional', type=str, default='false',
                           help='enable cVAE conditioning from clin_cond.csv')
    argparser.add_argument('--adversarial', type=str, default='false',
                           help='enable GRL adversarial head (MSE on clin_cond.csv)')
    argparser.add_argument('--lambda_adv', type=float, default=1.0,
                           help='weight on adversarial MSE loss')
    argparser.add_argument('--grl_alpha', type=float, default=1.0,
                           help='gradient reversal layer scale')

    args = argparser.parse_args()
    args.anneal = _parse_bool('anneal', args.anneal)
    args.variational = _parse_bool('variational', args.variational)
    args.conditional = _parse_bool('conditional', args.conditional)
    args.adversarial = _parse_bool('adversarial', args.adversarial)
    return args


if __name__ == '__main__':
    print()
    print('---------------------------------------------')
    print('AML-VAE: model training')
    print('---------------------------------------------')
    print()
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    os.makedirs(args.out, exist_ok=True)

    trainer = Trainer(
        root=args.proc,
        checkpoint=False,
        epochs=args.epochs,
        verbose=True,
        patience=args.patience,
        return_best_model=True,
        dataset_name=args.dataset_name,
        model_selection_metric=args.model_selection_metric,
        metrics_log_path=os.path.join(args.out, 'training_metrics.csv'),
        use_conditional=args.conditional,
        use_adversarial=args.adversarial,
        lambda_adv=args.lambda_adv,
    )

    config = {
        'n_hidden': args.n_hidden,
        'n_layers': args.n_layers,
        'n_latent': args.n_latent,
        'norm': args.norm,
        'variational': args.variational,
        'anneal': args.anneal,
        'dropout': args.dropout,
        'nonlin': args.nonlin,
        'lr': args.lr,
        'l2': args.l2,
        'beta': args.beta,
        'batch_size': args.batch_size,
        'free_bits': args.free_bits,
        'grl_alpha': args.grl_alpha,
    }

    model = trainer(config)

    # persist state dict + kwargs (portable across code changes)
    model.save(f'{args.out}/model.pt')

    print()
    print()
    print('training complete.')
    print('---------------------------------------------')
