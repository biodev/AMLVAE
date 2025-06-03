import pandas as pd 
import numpy as np 
from amlvae.data.ExprProcessor import ExprProcessor
from amlvae.data.ClinProcessor import ClinProcessor
from matplotlib import pyplot as plt
import torch 
import os 


import argparse 


def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--data', type=str, default='../../data/', 
                           help='path to data dir')
    argparser.add_argument('--out', type=str, default='../proc/',
                            help='path to output dir')
    argparser.add_argument('--target_type', type=str, default='tpm_unstranded',
                           help='target type for expression data')
    argparser.add_argument('--gene_selection_method', type=str, default='variance',
                           help='gene selection method')
    argparser.add_argument('--num_top_genes', type=int, default=1000,
                            help='number of top genes to select')
    argparser.add_argument('--norm_method', type=str, default='zscore',
                            help='normalization method')
    argparser.add_argument('--dataset_name', type=str, default='aml',
                            help='dataset name')
    argparser.add_argument('--gene_id_type', type=str, default='gene_name',
                           help='gene id type for expression data')
    argparser.add_argument('--sample_id_type', type=str, default='id',
                           help='sample id type for expression data')
    return argparser.parse_args()

if __name__ == '__main__': 

    print()
    print('---------------------------------------------')
    print('Data pre-processing')
    print('---------------------------------------------')
    print() 
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    expr_long1 = pd.read_csv(f'{args.data}/{args.dataset_name}_train.csv')
    expr_long2 = pd.read_csv(f'{args.data}/{args.dataset_name}_validation.csv')
    expr_long3 = pd.read_csv(f'{args.data}/{args.dataset_name}_test.csv')

    if args.dataset_name == 'aml':
        counts_name = 'unstranded'
    elif args.dataset_name == 'mds':
        counts_name = 'counts'

    eproc = ExprProcessor(expr_long1, 
                          target        = args.target_type,             # options: 'FPKM' (mds); aml-> 'unstranded','stranded_first','stranded_second','tpm_unstranded','fpkm_unstranded','fpkm_uq_unstranded'
                          counts_name   = counts_name,                  # options: 'unstranded' (aml), 'counts' (mds)
                          gene_col      = args.gene_id_type,            # options: 'gene_id' (mds),'gene_name' (aml)
                          sample_id_col = args.sample_id_type)          # options: 'array_id' (mds), 'id' (aml)
    
    eproc.select_genes_(args.gene_selection_method, top_n=args.num_top_genes)   # options: 'tcga', 'variance' 
    eproc.normalize_(args.norm_method)                                          # options: 'minmax', 'zscore'

    X_train, train_ids = eproc.get_data()
    X_train = torch.tensor(X_train, dtype=torch.float32)

    assert X_train.shape[0] == len(train_ids), 'X_train and train_ids do not match in length'
    assert not torch.isnan(X_train).any(), 'X_train contains NaN values'
    assert not torch.isinf(X_train).any(), 'X_train contains Inf values'
    print('train set:')
    print('\tmin value:', X_train.min())
    print('\tmax value:', X_train.max())
    print('\tmean value:', X_train.mean())
    print('\tstd value:', X_train.std())

    X_val, val_ids = eproc.process_new(expr_long2)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    assert X_val.shape[0] == len(val_ids), 'X_val and val_ids do not match in length'
    assert not torch.isnan(X_val).any(), 'X_val contains NaN values'
    assert not torch.isinf(X_val).any(), 'X_val contains Inf values'
    print('validation set:')
    print('\tmin value:', X_val.min())
    print('\tmax value:', X_val.max())
    print('\tmean value:', X_val.mean())
    print('\tstd value:', X_val.std())

    X_test, test_ids = eproc.process_new(expr_long3)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    assert X_test.shape[0] == len(test_ids), 'X_test and test_ids do not match in length'
    assert not torch.isnan(X_test).any(), 'X_test contains NaN values'
    assert not torch.isinf(X_test).any(), 'X_test contains Inf values'
    print('test set:')
    print('\tmin value:', X_test.min())
    print('\tmax value:', X_test.max())
    print('\tmean value:', X_test.mean())
    print('\tstd value:', X_test.std())

    X = torch.cat([X_train, X_val, X_test], dim=0).detach().numpy()
    ids = np.concatenate([train_ids, val_ids, test_ids], axis=0)

    df = pd.DataFrame(X, index=ids, columns = eproc.selected_genes)

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(f'{args.out}/{args.dataset_name}_expr.csv', index=True)
    torch.save({'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}, f'{args.out}/{args.dataset_name}_partitions.pt')

    print('pre-processing complete.')
    print('---------------------------------------------')
    print() 