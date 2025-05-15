
import numpy as np
import pandas as pd
from typing import Optional, Sequence


def pivot_expression(
    expr_long: pd.DataFrame,
    sample_id_col: str = 'id',
    gene_col: str = 'gene_name',
    value_col: str = 'fpkm_uq_unstranded',
) -> pd.DataFrame:
    """
    Turn a long-form DataFrame (sample × gene) into a samples×genes matrix.
    """
    df = (
        expr_long[[sample_id_col, gene_col, value_col]]
        .groupby([sample_id_col, gene_col])
        .mean()
        .reset_index()
        .pivot(index=sample_id_col, columns=gene_col, values=value_col)
    )
    return df


def select_genes_tcga(
    expr: pd.DataFrame,
    noise_threshold: float = 0.2,
    median_threshold: float = 10.0,
    top_n: int = 1000
) -> Sequence[str]:
    """
    "For mRNA-seq data, we removed genes expressed at or below a noise threshold of RPKM≤0.2 in at least 75% of
    samples, then identified the most-variant 25% of genes (N = 1728) by ranking expressed genes having a median RPKM
    of at least 10 by the coefficient of variation." 

    Divergence: we will select top_n genes instead of 25% of the total.
    """
    n = expr.shape[0]
    # 1) Noise filter
    q75 = expr.quantile(0.75, axis=0)
    remove = q75 <= noise_threshold
    keep1 = ~remove

    expr1 = expr.loc[:, keep1]

    # 2) Median filter
    keep2 = expr1.median(axis=0) >= median_threshold
    expr2 = expr1.loc[:, keep2]

    # 3) Top CV
    cv = expr2.std(axis=0) / expr2.mean(axis=0)
    return list(cv.sort_values(ascending=False).index[:top_n])


def select_genes_by_variance(
    expr: pd.DataFrame,
    top_n: int = 1000
) -> Sequence[str]:
    """
    Pick the top_n most variable genes (by raw variance).
    """
    var = expr.var(axis=0)
    return list(var.sort_values(ascending=False).index[:top_n])


def normalize_zscore(
    expr: pd.DataFrame
) -> (pd.DataFrame, dict):
    """
    1) log2(x+1)
    2) z‑score per gene
    Returns (normalized_df, params) where params={'mu':…, 'sd':…}.
    """
    logged = np.log2(expr + 1)
    mu = logged.mean(axis=0)
    sd = logged.std(axis=0)
    normed = (logged - mu) / (sd + 1e-8)
    return normed, {'mu': mu, 'sd': sd, 'method': 'zscore'}


def normalize_minmax(
    expr: pd.DataFrame
) -> (pd.DataFrame, dict):
    """
    Scale each gene to [0,1].
    Returns (normalized_df, params) where params={'min':…, 'max':…}.
    """
    expr = np.log2(expr + 1)
    mn = expr.min(axis=0)
    mx = expr.max(axis=0)
    normed = (expr - mn) / (mx - mn + 1e-8)
    return normed, {'min': mn, 'max': mx, 'method': 'minmax'}


class ExprProcessor:
    """
    Process a long-form expression DataFrame into a numpy array ready for modeling.
    
    Parameters
    ----------
    expr_long : pd.DataFrame
        Must contain at least ['id', 'gene_name', target] columns.
    target : str
        Column name for expression values.
    norm : {'zscore_log2', 'scale'}
        How to normalize after gene selection.
    feature_selection : {'tcga', 'variance', 'custom'}
        Method to pick which genes to keep.
    top_n : int
        Number of genes to pick (for 'variance') or
        fraction multiplier is fixed in 'tcga' (25%).
    custom_genes : Sequence[str], optional
        List of gene names to keep if feature_selection == 'custom'.
    """
    def __init__(
        self,
        expr_long: pd.DataFrame,
        target: str = 'fpkm_uq_unstranded', 
        gene_col: str = 'gene_name',
        sample_id_col: str = 'id',):
        
        # 1) pivot to get raw expression
        self.raw_expr = pivot_expression(expr_long, 
                                         value_col      = target, 
                                         gene_col       = gene_col, 
                                         sample_id_col  = sample_id_col)
        self.sample_ids = list(self.raw_expr.index)
        self.target = target
        self.gene_col = gene_col
        self.sample_id_col = sample_id_col


    def select_genes_(self, method='variance', top_n=1000):
        """
        Select genes based on variance.
        """
        if method == 'variance':
            genes = select_genes_by_variance(self.raw_expr, top_n=top_n)
        elif method == 'tcga':
            genes = select_genes_tcga(self.raw_expr, top_n=top_n)
        else:
            raise ValueError(f"Unknown method '{method}'")

        self.selected_genes = genes 

    def normalize_(self, method='zscore'):
        """
        Normalize the expression data.
        """
        if method == 'zscore':
            self.expr, self.transform_params = normalize_zscore(self.raw_expr)
        elif method == 'minmax':
            self.expr, self.transform_params = normalize_minmax(self.raw_expr)
        else:
            raise ValueError(f"Unknown normalization method '{method}'")

    def get_data(self):
        """Returns (X: np.ndarray, sample_ids: list)."""
        if hasattr(self, 'expr'): 
            if not hasattr(self, 'selected_genes'):
                raise ValueError("Gene selection has not been performed yet.")
            
            return self.expr[self.selected_genes].values, self.sample_ids
        else: 
            raise ValueError("Data has not been normalized yet.")

    def get_gene_names(self):
        """Returns the list of gene names in this order."""
        return self.gene_names

    def process_new(self, expr_long: pd.DataFrame):
        """
        Apply the same pivot → normalize → subset pipeline to a new dataset.
        """
        raw_expr = pivot_expression(expr_long, 
                                    value_col=self.target, 
                                    gene_col=self.gene_col, 
                                    sample_id_col=self.sample_id_col)
        # re‑pivoting uses same target name as originally passed

        if self.transform_params['method'] == 'zscore':
            logged = np.log2(raw_expr + 1)
            normed = (logged - self.transform_params['mu']) / (self.transform_params['sd'] + 1e-8)
        elif self.transform_params['method'] == 'minmax':
            mn, mx = self.transform_params['min'], self.transform_params['max']
            logged = np.log2(raw_expr + 1)
            normed = (logged - mn) / (mx - mn + 1e-8)
            # clip to [0,1]
            normed = np.clip(normed, 0, 1)
        else:
            raise ValueError(f"Unknown transform method '{self.transform_params['method']}'")

        return normed[self.selected_genes].values, list(normed.index)
