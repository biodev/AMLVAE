"""
tune_utils.py
Helper functions for extracting the best hyper‑params from `amlvae_tune_results.csv`
and formatting them for `train.py`.

Example
-------
>>> from tune_utils import best_config, config_to_cli
>>> cfg = best_config("runs/2025‑05‑02_baseline/tune/amlvae_tune_results.csv",
...                   metric="val_elbo")
>>> print(cfg)
{'n_hidden': 512, 'n_latent': 12, 'n_layers': 2, ...}

# build a CLI string you can append when calling train.py
>>> cli = config_to_cli(cfg)
>>> print(cli)
--n_hidden 512 --n_latent 12 --n_layers 2 --norm layer --variational true ...
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# All columns that train.py understands -----------------------------
_TRAIN_ARGS: List[str] = [
    "n_hidden",
    "n_latent",
    "n_layers",
    "norm",
    "variational",
    "anneal",
    "aggresive_updates",
    "dropout",
    "nonlin",
    "lr",
    "l2",
    "beta",
    "batch_size",
]

# Metrics where **higher is better**; everything else is minimised
_MAXIMISE = {"val_r2"}


def best_config(csv_file: str | Path,
                metric: str = "val_elbo") -> Dict[str, Any]:
    """
    Read the Ray/Hyperopt results CSV and return a dict with the
    best‑performing hyper‑parameters for `train.py`.

    Parameters
    ----------
    csv_file : str | Path
        Path to `amlvae_tune_results.csv`.
    metric : str
        Column name to optimise – one of 'val_elbo', 'val_mse', 'val_r2'.

    Returns
    -------
    Dict[str, Any]
        Keys exactly match the CLI flags of `train.py`.
    """
    assert metric in {"val_elbo", "val_mse", "val_r2"}, \
        f"Invalid metric '{metric}'. Must be one of 'val_elbo', 'val_mse', 'val_r2'."

    csv_file = Path(csv_file)
    if not csv_file.is_file():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {csv_file}.")

    idx = df[metric].idxmax() if metric in _MAXIMISE else df[metric].idxmin()
    row = df.loc[idx]

    # keep only columns that train.py accepts
    cfg = {k: row[k] for k in _TRAIN_ARGS if k in row}

    # Ensure Python types are clean (e.g. numpy.bool_ -> bool)
    for k, v in cfg.items():
        cfg[k] = bool(v) if isinstance(v, (bool, int)) and str(v) in {"True", "False"} else v

    return cfg


def config_to_cli(cfg: Dict[str, Any]) -> str:
    """
    Convert the hyper‑parameter dictionary into a CLI argument string.

    Booleans are rendered as 'true'/'false' to match argparse.

    Returns
    -------
    str
        e.g. "--n_hidden 512 --dropout 0.1 --variational true ..."
    """
    def _fmt(key, val):
        if isinstance(val, bool):
            return f"--{key} {str(val).lower()}"
        return f"--{key} {val}"

    return " ".join(_fmt(k, v) for k, v in cfg.items())
