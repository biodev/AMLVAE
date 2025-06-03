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
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from amlvae.models.VAE import VAE 
from amlvae.train.Trainer import Trainer

from amlvae.data.ExprProcessor import ExprProcessor
from amlvae.data.ClinProcessor import ClinProcessor

import argparse 




def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--data', type=str, default='../../data/', 
                           help='path to data dir')
    argparser.add_argument('--proc', type=str, default='../proc/',
                            help='path to proc dir')
    argparser.add_argument('--out', type=str, default='../output/',
                            help='path to output dir')
    argparser.add_argument('--epochs', type=int, default=1000,
                            help='number of epochs to train for')
    argparser.add_argument('--patience', type=int, default=1000,
                            help='patience for early stopping')
    argparser.add_argument('--n_hidden', type=int, default=512,
                            help='number of hidden units')
    argparser.add_argument('--n_latent', type=int, default=12,  
                            help='number of latent units')
    argparser.add_argument('--n_layers', type=int, default=2,
                            help='number of layers')
    argparser.add_argument('--norm', type=str, default='layer',
                            help='normalization method')
    argparser.add_argument('--variational', type=str, default='true',
                            help='variational method')
    argparser.add_argument('--anneal', type=str, default='true',
                            help='annealing method')
    argparser.add_argument('--dropout', type=float, default=0.0,
                            help='dropout rate')
    argparser.add_argument('--nonlin', type=str, default='elu',
                            help='nonlinear activation function')
    argparser.add_argument('--lr', type=float, default=1e-4,
                            help='learning rate')
    argparser.add_argument('--l2', type=float, default=0.0,
                            help='l2 regularization')
    argparser.add_argument('--beta', type=float, default=1.0,
                            help='beta parameter')
    argparser.add_argument('--batch_size', type=int, default=256,
                            help='batch size')
    argparser.add_argument('--dataset_name', type=str, default='aml',
                            help='dataset name')
    argparser.add_argument('--masked_prob', type=float, default=0.0,
                            help='probability of masking an input feature')
    
    args = argparser.parse_args()
    
    if args.anneal in ['true', 'True', 'TRUE', '1']:
        args.anneal = True
    elif args.anneal in ['false', 'False', 'FALSE', '0']:
        args.anneal = False
    else:
        raise ValueError(f'Unknown value for anneal: {args.anneal}')
    
    if args.variational in ['true', 'True', 'TRUE', '1']:
        args.variational = True
    elif args.variational in ['false', 'False', 'FALSE', '0']:
        args.variational = False
    else:
        raise ValueError(f'Unknown value for variational: {args.variational}')
                            
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

    trainer = Trainer(
        root=args.proc,
        checkpoint=False,
        epochs=args.epochs,
        verbose=True, 
        patience=args.patience,
        return_best_model=True,
        dataset_name=args.dataset_name,
        ) 
    
    config = {
            'n_hidden'   : args.n_hidden,
            'n_layers'   : args.n_layers,
            'n_latent'   : args.n_latent,
            'norm'       : args.norm,
            'variational': args.variational,
            'anneal'     : args.anneal,
            'dropout'    : args.dropout,
            'nonlin'     : args.nonlin,
            'lr'         : args.lr,
            'l2'         : args.l2,
            'beta'       : args.beta,
            'batch_size' : args.batch_size,
            'masked_prob': args.masked_prob,
        }

    model = trainer(config)

    torch.save(model, f'{args.out}/model.pt')
    
    print() 
    print() 
    print('training complete.')
    print('---------------------------------------------')
        

    
