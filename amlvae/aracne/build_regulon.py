"""Convert ARACNe-AP network.txt to pyviper Interactome regulon."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyviper import Interactome
from scipy.stats import spearmanr

REGULON_CACHE = "regulon_mds_aracne_human_symbol.parquet.gzip"


def _read_network(network_path: Path | str) -> pd.DataFrame:
    """Parse ARACNe-AP consolidated network.txt."""
    network_path = Path(network_path)
    if not network_path.exists():
        raise FileNotFoundError(network_path)

    # network.txt: Regulator Target MI pvalue (whitespace-separated, header row)
    net = pd.read_csv(
        network_path,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["regulator", "target", "mi", "pvalue"],
    )
    net["regulator"] = net["regulator"].astype(str)
    net["target"] = net["target"].astype(str)
    return net


def _spearman_mor(
    net: pd.DataFrame,
    expr_df: pd.DataFrame,
) -> pd.Series:
    """Spearman correlation (MoR) for each regulator-target pair."""
    expr_df = expr_df.astype(float)
    expr_df.index = expr_df.index.astype(str)
    mor_values = []
    for reg, tgt in zip(net["regulator"], net["target"]):
        reg, tgt = str(reg), str(tgt)
        if reg not in expr_df.index or tgt not in expr_df.index:
            mor_values.append(np.nan)
            continue
        x = expr_df.loc[reg].values
        y = expr_df.loc[tgt].values
        if np.std(x) == 0 or np.std(y) == 0:
            mor_values.append(0.0)
            continue
        rho, _ = spearmanr(x, y)
        mor_values.append(float(rho) if np.isfinite(rho) else 0.0)
    return pd.Series(mor_values, index=net.index)


def network_to_regulon(
    network_path: Path | str,
    expr_df: pd.DataFrame,
    name: str = "regulon_mds",
) -> tuple[pd.DataFrame, Interactome]:
    """Build pyviper regulon from consolidated ARACNe network.

    Parameters
    ----------
    network_path
        Path to ARACNe-AP ``network.txt``.
    expr_df
        Genes x samples matrix used for ARACNe (same normalization as VIPER input).
    name
        Interactome name.

    Returns
    -------
    (net_table, interactome) with columns regulator, target, mor, likelihood.
    """
    net = _read_network(network_path)
    net = net[net["regulator"] != net["target"]].copy()

    expr_df = expr_df.copy()
    expr_df.index = expr_df.index.astype(str)

    print(f"Computing Spearman MoR for {len(net):,} edges...")
    net["mor"] = _spearman_mor(net, expr_df)

    # likelihood = MI normalized to [0, 1] within each regulator (ARACNe-AP convention)
    net["likelihood"] = net.groupby("regulator")["mi"].transform(
        lambda s: s / s.max() if s.max() > 0 else 0.0
    )

    net_table = net[["regulator", "target", "mor", "likelihood"]].dropna(subset=["mor"])
    if net_table.empty:
        n_missing = (~net["regulator"].astype(str).isin(expr_df.index.astype(str))).sum()
        raise ValueError(
            f"No regulon edges after MoR computation. "
            f"Check that network gene IDs match expr_df.index "
            f"({len(net)} network edges; regulators missing from expression: {n_missing})."
        )
    interactome = Interactome(name, net_table)
    return net_table, interactome


def save_regulon(net_table: pd.DataFrame, path: Path | str) -> Path:
    """Save regulon net_table as parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net_table.to_parquet(path, compression="gzip", index=False)
    print(f"Wrote {path} ({len(net_table):,} edges)")
    return path


def load_aracne_regulon(repo_root: Path | str | None = None) -> Interactome:
    """Load cohort ARACNe regulon from repo cache."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    cache_path = Path(repo_root) / "data" / "regulons" / REGULON_CACHE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run notebooks/ARACNE.ipynb to build the regulon."
        )
    return Interactome("regulon_mds", pd.read_parquet(cache_path))
