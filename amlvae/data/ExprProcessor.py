"""Expression data processor.

The core ``ExprProcessor`` class is intended to be **fitted on training data
only** – you construct it with train-only long-form expression, call
``select_genes_()`` + ``normalize_()``, and then push held-out splits through
``process_new(...)`` which re-uses the stored gene list and normalization
statistics. Constructing it on a concatenation of train/val/test leaks
validation/test information.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def pivot_expression(
    expr_long: pd.DataFrame,
    sample_id_col: str = 'id',
    gene_col: str = 'gene_name',
    value_col: str = 'fpkm_uq_unstranded',
) -> pd.DataFrame:
    """
    Turn a long-form DataFrame (sample x gene) into a samples-by-genes matrix.

    Also drops all-NaN genes/samples with a printed count so that downstream
    variance ranking / normalization does not produce silent NaNs.
    """
    df = (
        expr_long[[sample_id_col, gene_col, value_col]]
        .groupby([sample_id_col, gene_col])
        .mean()
        .reset_index()
        .pivot(index=sample_id_col, columns=gene_col, values=value_col)
    )

    before_genes = df.shape[1]
    before_samples = df.shape[0]
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    dropped_genes = before_genes - df.shape[1]
    dropped_samples = before_samples - df.shape[0]
    if dropped_genes or dropped_samples:
        print(
            f'[pivot_expression] dropped {dropped_genes} all-NaN genes '
            f'and {dropped_samples} all-NaN samples'
        )

    return df


def select_genes_wgcna_protocol(
    expr: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    top_n: int = 1000,
    min_count: int = 10,
    min_total_count: int = 15,
    min_prop: float = 0.66,
) -> Sequence[str]:
    """Reproduce the edgeR + WGCNA gene-filtering workflow used in the
    original R script.

    1. edgeR-style low-count filter (via CPM).
    2. log2(FPKM+1) transform on the remaining genes.
    3. Remove flat genes (MAD == 0).
    4. Rank by variance and keep the ``top_n`` most variable genes.
    """
    lib_sizes = counts.sum(axis=1)
    min_lib = lib_sizes.min()
    cpm = counts.div(lib_sizes, axis=0) * 1e6

    k = min_count * 1e6 / min_lib
    n_samples = counts.shape[0]
    keep = (cpm >= k).sum(axis=0) >= np.ceil(min_prop * n_samples)
    keep &= counts.sum(axis=0) >= min_total_count

    if not keep.any():
        raise RuntimeError('No genes passed the edgeR low-count filter.')
    expr_filt = expr.loc[:, keep[keep].index]

    log_expr = np.log2(expr_filt + 1.0)

    med = np.median(log_expr.values, axis=0)
    mad_vals = np.median(np.abs(log_expr.values - med), axis=0)
    nz_mask = mad_vals > 0
    if not nz_mask.any():
        raise RuntimeError('All genes have MAD == 0 after log transform.')
    log_expr = log_expr.loc[:, log_expr.columns[nz_mask]]

    gene_var = log_expr.var(axis=0)
    top_genes = gene_var.sort_values(ascending=False).index[:top_n]

    return list(top_genes)


def select_genes_tcga(
    expr: pd.DataFrame,
    noise_threshold: float = 0.2,
    median_threshold: float = 10.0,
    top_n: int = 1000,
) -> Sequence[str]:
    """TCGA-style noise + median + CV filter."""
    q75 = expr.quantile(0.75, axis=0)
    keep1 = q75 > noise_threshold
    expr1 = expr.loc[:, keep1]

    keep2 = expr1.median(axis=0) >= median_threshold
    expr2 = expr1.loc[:, keep2]

    with np.errstate(divide='ignore', invalid='ignore'):
        cv = expr2.std(axis=0) / expr2.mean(axis=0)
    cv = cv.replace([np.inf, -np.inf], np.nan).dropna()
    return list(cv.sort_values(ascending=False).index[:top_n])


def select_genes_by_variance(
    expr: pd.DataFrame,
    top_n: int = 1000,
) -> Sequence[str]:
    """Rank genes by variance on ``log2(x+1)`` after dropping flat genes.

    Ranking on raw FPKM variance is dominated by highly expressed genes,
    so we log-transform first (matching the WGCNA/TCGA convention).
    """
    log_expr = np.log2(expr.fillna(0.0) + 1.0)

    # filter MAD == 0 (flat genes)
    med = np.median(log_expr.values, axis=0)
    mad_vals = np.median(np.abs(log_expr.values - med), axis=0)
    keep = mad_vals > 0
    if not keep.any():
        raise RuntimeError('All genes have MAD == 0 after log transform.')
    log_expr = log_expr.loc[:, log_expr.columns[keep]]

    var = log_expr.var(axis=0)
    return list(var.sort_values(ascending=False).index[:top_n])


def normalize_zscore(expr: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """log2(x+1) then z-score per gene. Returns (normed_df, params)."""
    logged = np.log2(expr + 1)
    mu = logged.mean(axis=0)
    sd = logged.std(axis=0)
    normed = (logged - mu) / (sd + 1e-8)
    return normed, {'mu': mu, 'sd': sd, 'method': 'zscore'}


def normalize_minmax(expr: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """log2(x+1) then min-max scale per gene to [0,1]."""
    expr = np.log2(expr + 1)
    mn = expr.min(axis=0)
    mx = expr.max(axis=0)
    normed = (expr - mn) / (mx - mn + 1e-8)
    return normed, {'min': mn, 'max': mx, 'method': 'minmax'}


class ExprProcessor:
    """Process a long-form expression DataFrame into a numpy array ready for
    modeling.

    Intended usage (fit on train only):

    >>> ep = ExprProcessor(train_long, target='FPKM',
    ...                    gene_col='gene_id', sample_id_col='array_id')
    >>> ep.select_genes_('wgcna', top_n=2000)
    >>> ep.normalize_('zscore')
    >>> X_train, train_ids = ep.get_data()
    >>> X_val,   val_ids   = ep.process_new(val_long)
    >>> X_test,  test_ids  = ep.process_new(test_long)
    """

    def __init__(
        self,
        expr_long: pd.DataFrame,
        target: str = 'fpkm_uq_unstranded',
        counts_name: str = 'unstranded',
        gene_col: str = 'gene_name',
        sample_id_col: str = 'id',
    ):
        self.raw_expr = pivot_expression(
            expr_long, value_col=target,
            gene_col=gene_col, sample_id_col=sample_id_col,
        )
        self._raw_counts = pivot_expression(
            expr_long, value_col=counts_name,
            gene_col=gene_col, sample_id_col=sample_id_col,
        )

        self.sample_ids = list(self.raw_expr.index)
        self.target = target
        self.gene_col = gene_col
        self.sample_id_col = sample_id_col

        self.selected_genes: Optional[Sequence[str]] = None
        self.expr: Optional[pd.DataFrame] = None
        self.transform_params: Optional[dict] = None

    def select_genes_(self, method: str = 'variance', top_n: int = 1000):
        """Select genes using one of {'variance', 'tcga', 'wgcna'}."""
        if method == 'variance':
            genes = select_genes_by_variance(self.raw_expr, top_n=top_n)
        elif method == 'tcga':
            genes = select_genes_tcga(self.raw_expr, top_n=top_n)
        elif method == 'wgcna':
            genes = select_genes_wgcna_protocol(
                self.raw_expr, self._raw_counts, top_n=top_n,
            )
        else:
            raise ValueError(f"Unknown method '{method}'")

        self.selected_genes = list(genes)

    def normalize_(self, method: str = 'zscore'):
        """Fit normalization on ``self.raw_expr`` (train only) and apply it."""
        if method == 'zscore':
            self.expr, self.transform_params = normalize_zscore(self.raw_expr)
        elif method == 'minmax':
            self.expr, self.transform_params = normalize_minmax(self.raw_expr)
        else:
            raise ValueError(f"Unknown normalization method '{method}'")

    def get_data(self):
        """Return ``(X: np.ndarray, sample_ids: list)`` for the training fit."""
        if self.expr is None:
            raise ValueError('Data has not been normalized yet; call normalize_().')
        if self.selected_genes is None:
            raise ValueError('Gene selection has not been performed yet; call select_genes_().')
        return self.expr[self.selected_genes].values, self.sample_ids

    def process_new(self, expr_long: pd.DataFrame):
        """Apply the fitted pivot -> normalize -> subset pipeline to new data."""
        if self.transform_params is None or self.selected_genes is None:
            raise ValueError(
                'ExprProcessor must be fitted (select_genes_() + normalize_()) '
                'before process_new() can be called.'
            )

        raw_expr = pivot_expression(
            expr_long,
            value_col=self.target,
            gene_col=self.gene_col,
            sample_id_col=self.sample_id_col,
        )

        missing = [g for g in self.selected_genes if g not in raw_expr.columns]
        if missing:
            raise ValueError(
                f'{len(missing)} selected genes are missing from the new data '
                f'(e.g. {missing[:5]!r}). Refusing to silently fill with NaN.'
            )

        if self.transform_params['method'] == 'zscore':
            logged = np.log2(raw_expr + 1)
            normed = (logged - self.transform_params['mu']) / (
                self.transform_params['sd'] + 1e-8
            )
        elif self.transform_params['method'] == 'minmax':
            mn, mx = self.transform_params['min'], self.transform_params['max']
            logged = np.log2(raw_expr + 1)
            normed = (logged - mn) / (mx - mn + 1e-8)
            normed = normed.clip(lower=0.0, upper=1.0)
        else:
            raise ValueError(
                f"Unknown transform method '{self.transform_params['method']}'"
            )

        return normed[self.selected_genes].values, list(normed.index)
