import os
import argparse

import numpy as np
import pandas as pd
import sklearn.model_selection

'''
example data row:

array_id        gene_id         FPKM
MLL_00003       DDX11L1         0.6867988628712519
MLL_00003       WASH7P          1.303650285434622
MLL_00003       MIR6859-3       1.112555172813955
'''


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--fpath', type=str, required=True,
                           help='path to long-form expression TSV/CSV')
    argparser.add_argument('--source_path', type=str, required=True,
                           help='path to source file with sample metadata')
    argparser.add_argument('--out', type=str, required=True,
                           help='path to output dir (will be created)')
    argparser.add_argument('--k', type=int, default=5,
                           help='number of folds for cross-validation')
    argparser.add_argument('--seed', type=int, default=0,
                           help='random seed for cross-validation')
    argparser.add_argument('--n_val', type=int, default=25,
                           help='number of validation samples')
    argparser.add_argument('--id_type_name', type=str, default='array_id',
                           help='name of the sample-id column in the expression file')
    argparser.add_argument('--dataset_name', type=str, default='mds',
                           help='prefix used for output CSV filenames '
                                '(<dataset_name>_{train,validation,test}.csv)')
    return argparser.parse_args()


if __name__ == '__main__':
    print()
    print('---------------------------------------------')
    print('K-fold partitioning')
    print('---------------------------------------------')
    print()
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    np.random.seed(args.seed)

    expr = pd.read_csv(args.fpath, sep='\t')

    ids = expr[args.id_type_name].unique().tolist()
    print(f'Number of unique {args.id_type_name} [BM + PB]: {len(ids)}')

    # -----------------------------------------------------------------------------------------------------------------------
    # 6/3/25 - remove peripheral blood samples (bone marrow samples only) as there are not many PB samples in the dataset.
    mds_source = pd.read_excel(args.source_path, sheet_name=0)
    BM_ids = mds_source[lambda x: x.material == 'BM']['exam_array'].unique().tolist()
    ids = list(set(ids).intersection(set(BM_ids)))
    print(f'Number of unique {args.id_type_name} [BM only]: {len(ids)}')
    # -----------------------------------------------------------------------------------------------------------------------

    ids = np.array(ids)

    os.makedirs(args.out, exist_ok=True)

    kf = sklearn.model_selection.KFold(
        n_splits=args.k, shuffle=True, random_state=args.seed
    )

    for i, (train_idx, test_idx) in enumerate(kf.split(ids)):
        print(f'Generating partition fold {i + 1}/{args.k}', end='\r')

        fold_out_dir = os.path.join(args.out, f'fold_{i}')
        os.makedirs(fold_out_dir, exist_ok=True)

        val_idx = np.random.choice(train_idx, args.n_val, replace=False)
        train_idx = np.setdiff1d(train_idx, val_idx)

        train_ids = ids[train_idx]
        val_ids = ids[val_idx]
        test_ids = ids[test_idx]

        expr_train = expr[expr[args.id_type_name].isin(train_ids)]
        expr_val = expr[expr[args.id_type_name].isin(val_ids)]
        expr_test = expr[expr[args.id_type_name].isin(test_ids)]

        expr_train.to_csv(f'{fold_out_dir}/{args.dataset_name}_train.csv', index=False)
        expr_val.to_csv(f'{fold_out_dir}/{args.dataset_name}_validation.csv', index=False)
        expr_test.to_csv(f'{fold_out_dir}/{args.dataset_name}_test.csv', index=False)

        assert len(np.intersect1d(train_ids, val_ids)) == 0, 'train/val ids overlap'
        assert len(np.intersect1d(train_ids, test_ids)) == 0, 'train/test ids overlap'
        assert len(np.intersect1d(val_ids, test_ids)) == 0, 'val/test ids overlap'

    print()
    print()
    print('partitioning complete.')
    print('---------------------------------------------')
    print()
