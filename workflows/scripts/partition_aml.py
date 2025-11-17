from typing import Any


import pandas as pd 
import numpy as np 
import argparse 
import sklearn 
import os 
from amlvae.data.ExprProcessor import ExprProcessor
import torch 

'''
example data: 

id,gene_id,gene_name,gene_type,unstranded,stranded_first,stranded_second,tpm_unstranded,fpkm_unstranded,fpkm_uq_unstranded
9507c9d3-5021-49d9-b0e6-e04304077840,ENSG00000000003.15,TSPAN6,protein_coding,4,0,4,0.113,0.0316,0.0279
9507c9d3-5021-49d9-b0e6-e04304077840,ENSG00000000005.6,TNMD,protein_coding,0,0,0,0.0,0.0,0.0

'''

def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--fpath', type=str, default='../../data/aml_dummy.csv', 
                           help='path to data file')
    argparser.add_argument('--out', type=str, default='../output/aml_partitions/',
                            help='path to output dir')
    argparser.add_argument('--k', type=int, default=5,
                            help='number of folds for cross-validation')
    argparser.add_argument('--seed', type=int, default=0,
                            help='random seed for cross-validation')
    argparser.add_argument('--n_val', type=int, default=25,
                            help='number of validation samples')
    argparser.add_argument('--id_type_name', type=str, default='id',
                            help='name of the id type in the file')
    argparser.add_argument('--num_top_genes', type=int, default=2000,
                            help='number of top genes to select')
    argparser.add_argument('--gene_selection_method', type=str, default='variance',
                            help='gene selection method')
    argparser.add_argument('--norm_method', type=str, default='zscore',
                            help='normalization method')
    argparser.add_argument('--dataset_name', type=str, default='aml',
                            help='dataset name')
    argparser.add_argument('--target_name', type=str, default='FPKM',
                            help='target name')
    argparser.add_argument('--counts_name', type=str, default='unstranded',
                            help='counts name')
    argparser.add_argument('--gene_col', type=str, default='gene_name',
                            help='gene column name')
    argparser.add_argument('--id_col', type=str, default='id',
                            help='id column name')

    return argparser.parse_args()


def proc(expr_train, expr_val, expr_test, target_name, counts_name, gene_col, id_col, 
            num_top_genes, gene_selection_method, norm_method, verbose=True):

    if verbose: 
        print('\tpre-processing expression data...') 

    eproc = ExprProcessor(expr_train, 
                          target        = target_name,               # options: 'FPKM' (mds); aml-> 'unstranded','stranded_first','stranded_second','tpm_unstranded','fpkm_unstranded','fpkm_uq_unstranded'
                          counts_name   = counts_name,         # options: 'unstranded' (aml), 'counts' (mds)
                          gene_col      = gene_col,          # options: 'gene_id' (mds),'gene_name' (aml)
                          sample_id_col = id_col)                 # options: 'array_id' (mds), 'id' (aml)
    
    eproc.select_genes_(gene_selection_method, top_n=num_top_genes)   # options: 'tcga', 'variance' 
    eproc.normalize_(norm_method)                                          # options: 'minmax', 'zscore'

    X_train, train_ids = eproc.get_data()
    X_train = torch.tensor(X_train, dtype=torch.float32)

    assert X_train.shape[0] == len(train_ids), 'X_train and train_ids do not match in length'
    assert not torch.isnan(X_train).any(), 'X_train contains NaN values'
    assert not torch.isinf(X_train).any(), 'X_train contains Inf values'
    if verbose: 
        print('\ttrain set:')
        print('\t\tmin value:', X_train.min())
        print('\t\tmax value:', X_train.max())
        print('\t\tmean value:', X_train.mean())
        print('\t\tstd value:', X_train.std())

    X_val, val_ids = eproc.process_new(expr_val)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    assert X_val.shape[0] == len(val_ids), 'X_val and val_ids do not match in length'
    assert not torch.isnan(X_val).any(), 'X_val contains NaN values'
    assert not torch.isinf(X_val).any(), 'X_val contains Inf values'
    if verbose: 
        print('\tvalidation set:')
        print('\t\tmin value:', X_val.min())
        print('\t\tmax value:', X_val.max())
        print('\t\tmean value:', X_val.mean())
        print('\t\tstd value:', X_val.std())

    X_test, test_ids = eproc.process_new(expr_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    assert X_test.shape[0] == len(test_ids), 'X_test and test_ids do not match in length'
    assert not torch.isnan(X_test).any(), 'X_test contains NaN values'
    assert not torch.isinf(X_test).any(), 'X_test contains Inf values'
    if verbose: 
        print('\ttest set:')
        print('\t\tmin value:', X_test.min())
        print('\t\tmax value:', X_test.max())
        print('\t\tmean value:', X_test.mean())
        print('\t\tstd value:', X_test.std())

    X = torch.cat([X_train, X_val, X_test], dim=0).detach().numpy()
    ids = np.concatenate([train_ids, val_ids, test_ids], axis=0)

    df = pd.DataFrame(X, index=ids, columns = eproc.selected_genes)

    if verbose: 
        print('\tpre-processing complete.')

    partition_dict = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}

    return df, ids, partition_dict


def partition(expr, ids, train_ids, test_ids, n_val, id_col, verbose=True):
    
    if verbose: 
        print('\tpartitioning data...')

    # select validation ids from train ids 
    val_ids = np.random.choice(train_ids, n_val, replace=False)
    train_ids = np.setdiff1d(train_ids, val_ids)

    train_ids = ids[train_ids]
    test_ids = ids[test_ids]
    val_ids = ids[val_ids]

    expr_train = expr[expr[id_col].isin(train_ids)]
    expr_val = expr[expr[id_col].isin(val_ids)]
    expr_test = expr[expr[id_col].isin(test_ids)] 

    assert len(train_ids) == len(np.unique(train_ids)), 'train ids are not unique'
    assert len(val_ids) == len(np.unique(val_ids)), 'val ids are not unique'
    assert len(test_ids) == len(np.unique(test_ids)), 'test ids are not unique'
    assert len(np.intersect1d(train_ids, val_ids)) == 0, 'train and val ids overlap'
    assert len(np.intersect1d(train_ids, test_ids)) == 0, 'train and test ids overlap'
    assert len(np.intersect1d(val_ids, test_ids)) == 0, 'val and test ids overlap'
    assert len(np.intersect1d(train_ids, test_ids)) == 0, 'train and test ids overlap'
    assert len(np.intersect1d(train_ids, ids)) == len(train_ids), 'train ids not in ids'
    assert len(np.intersect1d(val_ids, ids)) == len(val_ids), 'val ids not in ids'
    assert len(np.intersect1d(test_ids, ids)) == len(test_ids), 'test ids not in ids'
    assert len(np.intersect1d(train_ids, np.concatenate([val_ids, test_ids]))) == 0, 'train ids overlap with val and test ids'
    assert len(np.intersect1d(val_ids, np.concatenate([train_ids, test_ids]))) == 0, 'val ids overlap with train and test ids'
    assert len(np.intersect1d(test_ids, np.concatenate([train_ids, val_ids]))) == 0, 'test ids overlap with train and val ids'

    return expr_train, expr_val, expr_test


if __name__ == '__main__': 

    print()
    print('---------------------------------------------')
    print(f'partitioning (K-fold cross-validation)')
    print('---------------------------------------------')
    print() 
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    # set random seed
    np.random.seed(args.seed)

    # load data
    expr = pd.read_csv(args.fpath, sep=',')

    ids = expr[args.id_type_name].unique().tolist()
    print(f'Number of unique {args.id_type_name}: {len(ids)}')

    # convert to array for indexing 
    ids = np.array(ids)

    os.makedirs(args.out, exist_ok=True)

    for i, (train_ids, test_ids) in enumerate(sklearn.model_selection.KFold(n_splits=args.k, shuffle=True, random_state=args.seed).split(ids)): 
        
        print()
        print('---------------------------------------------')
        print(f'Generating partition fold {i+1}/{args.k}')
        
        fold_out_dir = f'{args.out}/fold_{i}/'
        os.makedirs(fold_out_dir, exist_ok=True) 

        expr_train, expr_val, expr_test = partition(expr, ids, train_ids, test_ids, args.n_val, args.id_col) 

        df, ids, partition_dict = proc(expr_train, expr_val, expr_test, args.target_name, args.counts_name, args.gene_col, args.id_col, 
                        args.num_top_genes, args.gene_selection_method, args.norm_method)

        df.to_csv(f'{fold_out_dir}/proc/{args.dataset_name}_expr.csv', index=True)
        torch.save(partition_dict, f'{fold_out_dir}/proc/{args.dataset_name}_partitions.pt')

        
    print()
    print()
    print('partitioning complete.')
    print('---------------------------------------------')
    print()








