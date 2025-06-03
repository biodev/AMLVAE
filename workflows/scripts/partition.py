import pandas as pd 
import numpy as np 
import argparse 
import sklearn 
import os 

'''
example data: 

array_id        gene_id         FPKM
MLL_00003       DDX11L1         0.6867988628712519
MLL_00003       WASH7P          1.303650285434622
MLL_00003       MIR6859-3       1.112555172813955
'''

def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--fpath', type=str, default='../../data/mds_dummy.csv', 
                           help='path to data file')
    argparser.add_argument('--source_path', type=str, default='../../data/mds_source.xlsx',
                            help='path to source file with sample metadata')
    argparser.add_argument('--out', type=str, default='../output/mds_partitions/',
                            help='path to output dir')
    argparser.add_argument('--k', type=int, default=5,
                            help='number of folds for cross-validation')
    argparser.add_argument('--seed', type=int, default=0,
                            help='random seed for cross-validation')
    argparser.add_argument('--n_val', type=int, default=25,
                            help='number of validation samples')
    argparser.add_argument('--id_type_name', type=str, default='id',
                            help='name of the id type in the file')

                            
    return argparser.parse_args()

if __name__ == '__main__': 

    print()
    print('---------------------------------------------')
    print('MDS partitioning; K-fold cross-validation')
    print('---------------------------------------------')
    print() 
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    # set random seed
    np.random.seed(args.seed)

    # load data
    expr = pd.read_csv(args.fpath, sep='\t')

    ids = expr[args.id_type_name].unique().tolist()
    print(f'Number of unique {args.id_type_name} [BM + PB]: {len(ids)}')

    # -----------------------------------------------------------------------------------------------------------------------
    # 6/3/25 - remove periphereal blood samples (bone marrow samples only) as their are not many PB samples in the dataset. 
    mds_source = pd.read_excel(args.source_path, sheet_name=0)
    BM_ids = mds_source[lambda x: x.material == 'BM']['exam_array'].unique().tolist()
    ids = list( set(ids).intersection(set(BM_ids))  )
    print(f'Number of unique {args.id_type_name} [BM only]: {len(ids)}')
    # -----------------------------------------------------------------------------------------------------------------------

    # convert to array for indexing 
    ids = np.array(ids)

    os.makedirs(args.out, exist_ok=True)

    for i, (train_ids, test_ids) in enumerate(sklearn.model_selection.KFold(n_splits=args.k, shuffle=True, random_state=args.seed).split(ids)): 
        print(f'Generating partition fold {i+1}/{args.k}', end='\r')

        fold_out_dir = f'{args.out}/fold_{i}/'
        os.makedirs(fold_out_dir, exist_ok=True) 
        
        # select validation ids from train ids 
        n_val = args.n_val 
        val_ids = np.random.choice(train_ids, n_val, replace=False)
        train_ids = np.setdiff1d(train_ids, val_ids)

        train_ids = ids[train_ids]
        test_ids = ids[test_ids]
        val_ids = ids[val_ids]

        expr_train = expr[expr[args.id_type_name].isin(train_ids)]
        expr_val = expr[expr[args.id_type_name].isin(val_ids)]
        expr_test = expr[expr[args.id_type_name].isin(test_ids)] 

        expr_train.to_csv(f'{fold_out_dir}/mds_train.csv', index=False)
        expr_val.to_csv(f'{fold_out_dir}/mds_validation.csv', index=False)
        expr_test.to_csv(f'{fold_out_dir}/mds_test.csv', index=False)

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

    print()
    print()
    print('partitioning complete.')
    print('---------------------------------------------')
    print()








