import os
import argparse

import numpy as np
import pandas as pd
import torch

from amlvae.data.ExprProcessor import ExprProcessor

_GENDER_COL = 'Gender 1=female;2=male'
_AGE_COL = 'Age'


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--data', type=str, required=True,
                           help='directory containing <dataset_name>_{train,validation,test}.csv')
    argparser.add_argument('--out', type=str, required=True,
                           help='output directory')
    argparser.add_argument('--target_type', type=str, default='tpm_unstranded',
                           help='expression column name to use')
    argparser.add_argument('--gene_selection_method', type=str, default='variance',
                           help='gene selection method: variance | tcga | wgcna')
    argparser.add_argument('--num_top_genes', type=int, default=1000,
                           help='number of top genes to keep')
    argparser.add_argument('--norm_method', type=str, default='zscore',
                           help='zscore | minmax')
    argparser.add_argument('--dataset_name', type=str, default='aml',
                           help='dataset name prefix')
    argparser.add_argument('--gene_id_type', type=str, default='gene_name',
                           help='gene id column name')
    argparser.add_argument('--sample_id_type', type=str, default='id',
                           help='sample id column name in expression tables')
    argparser.add_argument('--clin_path', type=str, default=None,
                           help='clinical data path')
    argparser.add_argument(
        '--conditional_feats',
        nargs='*',
        default=None,
        help='conditional feature column names (e.g. gender age)',
    )
    argparser.add_argument('--clin_id_column', type=str, default='MLL ID',
                           help='sample id column in clinical table')
    argparser.add_argument('--source_path', type=str, default=None,
                           help='source material xlsx mapping expression ids to clinical ids')
    return argparser.parse_args()


_COUNTS_NAME = {
    'aml': 'unstranded',
    'mds': 'counts',
}


def _read_table(path):
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    if path.endswith('.csv'):
        return pd.read_csv(path)
    raise ValueError('Unsupported file format. Use .xlsx or .csv.')


def _clean_clin_feats(df):
    out = df.copy()
    if _GENDER_COL in out.columns:
        out['gender'] = pd.to_numeric(out[_GENDER_COL], errors='coerce') - 1
        out['gender'] = out['gender'].fillna(0)

    if _AGE_COL in out.columns:
        out['age'] = pd.to_numeric(out[_AGE_COL], errors='coerce') / 100
        out['age'] = out['age'].fillna(0.5)

    return out


def _sample_ids_in_fold(data_dir, dataset_name, sample_id_col):
    ids = []
    for split in ('train', 'validation', 'test'):
        path = f'{data_dir}/{dataset_name}_{split}.csv'
        chunk = pd.read_csv(path, usecols=[sample_id_col])
        ids.extend(chunk[sample_id_col].unique().tolist())
    return pd.Index(ids).unique()


def _resolve_column(df, preferred, aliases=()):
    if preferred in df.columns:
        return preferred
    for name in aliases:
        if name in df.columns:
            return name
    return None


def _load_source_id_map(source_path, sample_id_col, clin_id_col, clin):
    src = _read_table(source_path)
    array_col = _resolve_column(
        src, sample_id_col, aliases=('exam_array', 'array_id')
    )
    if array_col is None:
        raise KeyError(
            f"Expression id column not found in source table. Tried "
            f"{sample_id_col!r}, 'exam_array', 'array_id'. Columns: {list(src.columns)}"
        )

    src_clin_col = _resolve_column(
        src, clin_id_col, aliases=('MLL_ID', 'mll_id', 'MLL id')
    )
    if src_clin_col is not None:
        mapping = src[[array_col, src_clin_col]].drop_duplicates()
        mapping = mapping.dropna(subset=[array_col, src_clin_col])
        return mapping.rename(
            columns={array_col: sample_id_col, src_clin_col: clin_id_col}
        )

    # Some cohorts use the same identifier in expression and clinical tables.
    if clin_id_col not in clin.columns:
        raise KeyError(
            f"Clinical id {clin_id_col!r} not in clinical or source tables."
        )
    array_ids = src[array_col].dropna().unique()
    clin_ids = set(clin[clin_id_col].dropna().unique())
    shared = [i for i in array_ids if i in clin_ids]
    if not shared:
        raise KeyError(
            f"Could not map {sample_id_col!r} to {clin_id_col!r}. "
            f"Add {clin_id_col!r} to the source table or align id naming. "
            f"Source columns: {list(src.columns)}"
        )
    return pd.DataFrame({sample_id_col: shared, clin_id_col: shared})


def _build_clin_cond(clin, args, sample_ids):
    clin = _clean_clin_feats(clin)
    feat_cols = list(args.conditional_feats)
    missing_feats = [c for c in feat_cols if c not in clin.columns]
    if missing_feats:
        raise KeyError(
            f"conditional features not in clinical table after cleaning: {missing_feats}"
        )

    id_out = args.sample_id_type
    clin_id = args.clin_id_column

    if id_out in clin.columns:
        out = clin[[id_out] + feat_cols].copy()
    elif clin_id in clin.columns:
        if not args.source_path:
            raise ValueError(
                f"Clinical table has {clin_id!r} but not {id_out!r}; "
                "pass --source_path to map expression sample ids to clinical ids."
            )
        id_map = _load_source_id_map(args.source_path, id_out, clin_id, clin)
        clin_sub = clin[[clin_id] + feat_cols].drop_duplicates(subset=[clin_id])
        out = id_map.merge(clin_sub, on=clin_id, how='inner')
        out = out[[id_out] + feat_cols]
    else:
        raise KeyError(
            f"Neither expression id {id_out!r} nor clinical id {clin_id!r} "
            f"in clinical columns: {list(clin.columns)}"
        )

    out = out.drop_duplicates(subset=[id_out]).set_index(id_out)
    out = out.loc[out.index.isin(sample_ids)]
    if out.empty:
        raise ValueError(
            'No clinical rows matched expression sample ids for this fold. '
            f'Check --source_path and id columns ({id_out!r}, {clin_id!r}).'
        )
    missing_ids = sample_ids.difference(out.index)
    if len(missing_ids):
        raise KeyError(
            f"{len(missing_ids)} expression sample id(s) missing from clinical "
            f"conditional table (e.g. {list(missing_ids[:5])})"
        )
    return out


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

    os.makedirs(args.out, exist_ok=True)

    sample_ids = _sample_ids_in_fold(
        args.data, args.dataset_name, args.sample_id_type
    )

    if args.clin_path and args.conditional_feats:
        clin = _read_table(args.clin_path)
        clin_cond = _build_clin_cond(clin, args, sample_ids)
        clin_cond.to_csv(f'{args.out}/{args.dataset_name}_clin_cond.csv')

    expr_long1 = pd.read_csv(f'{args.data}/{args.dataset_name}_train.csv')
    expr_long2 = pd.read_csv(f'{args.data}/{args.dataset_name}_validation.csv')
    expr_long3 = pd.read_csv(f'{args.data}/{args.dataset_name}_test.csv')

    counts_name = _COUNTS_NAME.get(args.dataset_name, 'counts')

    eproc = ExprProcessor(
        expr_long1,
        target        = args.target_type,
        counts_name   = counts_name,
        gene_col      = args.gene_id_type,
        sample_id_col = args.sample_id_type,
    )

    eproc.select_genes_(args.gene_selection_method, top_n=args.num_top_genes)
    eproc.normalize_(args.norm_method)

    X_train, train_ids = eproc.get_data()
    X_train = torch.tensor(X_train, dtype=torch.float32)

    assert X_train.shape[0] == len(train_ids), 'X_train and train_ids length mismatch'
    assert not torch.isnan(X_train).any(), 'X_train contains NaN values'
    assert not torch.isinf(X_train).any(), 'X_train contains Inf values'
    print('train set:')
    print('\tmin value:', X_train.min().item())
    print('\tmax value:', X_train.max().item())
    print('\tmean value:', X_train.mean().item())
    print('\tstd value:', X_train.std().item())

    X_val, val_ids = eproc.process_new(expr_long2)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    assert X_val.shape[0] == len(val_ids), 'X_val / val_ids length mismatch'
    assert not torch.isnan(X_val).any(), 'X_val contains NaN values'
    assert not torch.isinf(X_val).any(), 'X_val contains Inf values'
    print('validation set:')
    print('\tmin value:', X_val.min().item())
    print('\tmax value:', X_val.max().item())
    print('\tmean value:', X_val.mean().item())
    print('\tstd value:', X_val.std().item())

    X_test, test_ids = eproc.process_new(expr_long3)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    assert X_test.shape[0] == len(test_ids), 'X_test / test_ids length mismatch'
    assert not torch.isnan(X_test).any(), 'X_test contains NaN values'
    assert not torch.isinf(X_test).any(), 'X_test contains Inf values'
    print('test set:')
    print('\tmin value:', X_test.min().item())
    print('\tmax value:', X_test.max().item())
    print('\tmean value:', X_test.mean().item())
    print('\tstd value:', X_test.std().item())

    X = torch.cat([X_train, X_val, X_test], dim=0).detach().numpy()
    ids = np.concatenate([train_ids, val_ids, test_ids], axis=0)

    df = pd.DataFrame(X, index=ids, columns=eproc.selected_genes)

    df.to_csv(f'{args.out}/{args.dataset_name}_expr.csv', index=True)
    torch.save(
        {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids},
        f'{args.out}/{args.dataset_name}_partitions.pt',
    )

    print('pre-processing complete.')
    print('---------------------------------------------')
    print()
