"""Load per-sample conditional / adversarial clinical features from proc output."""

from __future__ import annotations

import os

import pandas as pd
import torch


def clin_cond_path(proc_dir: str, dataset_name: str) -> str:
    return os.path.join(proc_dir, f"{dataset_name}_clin_cond.csv")


def load_clin_cond(
    proc_dir: str,
    dataset_name: str,
    ids,
    *,
    dtype=torch.float32,
) -> torch.Tensor:
    """Load ``{dataset}_clin_cond.csv`` and align rows to ``ids``.

    Parameters
    ----------
    proc_dir : str
        Directory containing the proc outputs (same as Trainer ``root``).
    dataset_name : str
        Dataset prefix (e.g. ``mds``).
    ids : array-like
        Sample IDs in the desired row order.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(len(ids), D)``.
    """
    path = clin_cond_path(proc_dir, dataset_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Clinical conditional file not found: {path}. "
            "Run proc with --clin_path and --conditional_feats."
        )

    clin = pd.read_csv(path, index_col=0)
    ids = list(ids)
    missing = [i for i in ids if i not in clin.index]
    if missing:
        raise KeyError(
            f"{len(missing)} sample id(s) missing from {path} "
            f"(e.g. {missing[:5]})"
        )

    values = clin.loc[ids].values.astype("float32")
    return torch.tensor(values, dtype=dtype)
