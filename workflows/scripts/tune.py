import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import torch 
import os 
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
import pandas as pd 
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from amlvae.train.Trainer import Trainer
import argparse 



####################################################################################
####################################################################################
__SEARCH_SPACE__ = {
        "lr"                : hp.choice('lr', [5e-5, 1e-4, 5e-4]),
        "l2"                : hp.choice('l2', [0, 1e-6, 1e-2]),
        "n_hidden"          : hp.choice('n_hidden', [256, 512, 1024, 2048]),
        "n_layers"          : hp.choice('n_layers', [1, 2, 4]),
        "batch_size"        : hp.choice('batch_size', [128, 256]),
        "aggresive_updates" : hp.choice('aggresive_updates', [False]),
        "norm"              : hp.choice('norm', ["batch", "layer", "none"]),
        "variational"       : hp.choice('variational', [True]),
        "anneal"            : hp.choice('anneal', [True, False]),
        "dropout"           : hp.choice('dropout', [0.0, 0.1, 0.2, 0.3]),
        "nonlin"            : hp.choice('nonlin', ["elu", "gelu"]),
        "beta"              : hp.choice('beta', [1]),
    }
####################################################################################
####################################################################################

def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--data', type=str, default='../../data/', 
                           help='path to data dir')
    argparser.add_argument('--proc', type=str, default='../proc/',
                            help='path to proc dir')
    argparser.add_argument('--out', type=str, default='../output/',
                            help='path to output dir')
    argparser.add_argument('--target_metric', type=str, default='elbo',
                            help='target metric for tuning')
    argparser.add_argument('--num_samples', type=int, default=100,
                            help='number of samples for tuning')
    argparser.add_argument('--patience', type=int, default=100,
                            help='patience for early stopping')
    argparser.add_argument('--gpus', type=int, default=1,
                            help='number of gpus to use')
    argparser.add_argument('--cpus', type=int, default=10,
                            help='number of cpus to use')
    argparser.add_argument('--epochs', type=int, default=1000,
                            help='number of epochs to train for')
    argparser.add_argument('--n_latent', type=int, default=12,
                            help='number of latent units')
    argparser.add_argument('--dataset_name', type=str, default='aml',
                            help='dataset name')
                            
    return argparser.parse_args()

if __name__ == '__main__': 

    print()
    print('---------------------------------------------')
    print('AML-VAE: Hyperparameter tuning')
    print('---------------------------------------------')
    print() 
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    # add n_latent to search space, for now we will use only 1 value 
    __SEARCH_SPACE__['n_latent'] = hp.choice('n_latent', [args.n_latent])

    trainer = Trainer(
        root=args.proc,
        checkpoint=True,
        epochs=args.epochs,
        verbose=False, 
        patience=args.patience,
        dataset_name=args.dataset_name
        )

    if args.target_metric == 'mse': 
        metric = 'val_mse'
        mode = 'min'
    elif args.target_metric == 'r2':
        metric = 'val_r2'
        mode = 'max'
    elif args.target_metric == 'elbo':
        metric = 'val_elbo'
        mode = 'min'

    hyperopt_search = HyperOptSearch(__SEARCH_SPACE__, metric=metric, mode=mode)

    tuner = tune.Tuner(
        tune.with_resources(trainer, {"cpu":args.cpus, "gpu": args.gpus}),
        tune_config=tune.TuneConfig(num_samples=args.num_samples, 
                                    search_alg=hyperopt_search
        ), 
    )
    results = tuner.fit()   

    dfs = {result.path: result.metrics_dataframe for result in results}
    res = {'val_mse': [], 'val_r2':[], 'val_elbo':[], 'batch_size': [], 'beta': [], 'dropout': [], 'l2': [], 'lr': [], 'n_hidden': [], 'n_latent': [], 'n_layers': [], 'nonlin': [], 'norm': [], 'variational': [], 'n_epochs': [], 'path':[], 'aggresive_updates': [], 'anneal': []}
    for name, df in dfs.items():
        best_trial = df.loc[df.val_mse.idxmin()]
        res['val_mse'].append(best_trial.val_mse)
        res['val_r2'].append(best_trial.val_r2)
        res['val_elbo'].append(best_trial.val_elbo)
        res['batch_size'].append(best_trial['config/batch_size'])
        res['beta'].append(best_trial['config/beta'])
        res['dropout'].append(best_trial['config/dropout'])
        res['l2'].append(best_trial['config/l2'])
        res['lr'].append(best_trial['config/lr'])
        res['n_hidden'].append(best_trial['config/n_hidden'])
        res['n_latent'].append(best_trial['config/n_latent'])
        res['n_layers'].append(best_trial['config/n_layers'])
        res['nonlin'].append(best_trial['config/nonlin'])
        res['norm'].append(best_trial['config/norm'])
        res['variational'].append(best_trial['config/variational'])
        res['n_epochs'].append(best_trial['training_iteration'])
        res['aggresive_updates'].append(best_trial['config/aggresive_updates'])
        res['anneal'].append(best_trial['config/anneal'])
        res['path'].append(name)

    res = pd.DataFrame(res).sort_values(by=metric, ascending=True).reset_index(drop=True)

    res.to_csv(f'{args.out}/amlvae_tune_results.csv', index=False)

    print('hyperparameter tuning complete.')
    print('---------------------------------------------')
    print() 



