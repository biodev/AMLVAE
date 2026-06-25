"""Helpers for extracting the best hyper-params from ``amlvae_tune_results.csv``
and formatting them for ``train.py``.

Example
-------
>>> from amlvae.utils.tune_parsing import best_config, config_to_cli
>>> cfg = best_config(".../amlvae_tune_results.csv", metric="val_recon_mse")
>>> cli = config_to_cli(cfg)
>>> # `cli` can be appended to a train.py invocation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# CLI flags that train.py understands -------------------------------
_TRAIN_ARGS: List[str] = [
    "n_hidden",
    "n_latent",
    "n_layers",
    "norm",
    "variational",
    "anneal",
    "dropout",
    "nonlin",
    "lr",
    "l2",
    "beta",
    "batch_size",
    "free_bits",
]

# Columns we know are booleans regardless of the dtype pandas inferred.
_BOOL_ARGS = {"variational", "anneal"}
# Columns we know are integer-typed.
_INT_ARGS = {"n_hidden", "n_latent", "n_layers", "batch_size"}
# Columns we know are float-typed.
_FLOAT_ARGS = {"dropout", "lr", "l2", "beta", "free_bits"}

# Metrics where higher is better; everything else is minimised ------
_MAXIMISE = {"val_r2"}

_VALID_METRICS = {"val_recon_mse", "val_nll", "val_elbo", "val_kld", "val_r2"}


def _coerce_bool(value: Any) -> bool:
    """Robustly turn a CSV-roundtripped value into a Python bool.

    Handles:
      * numpy.bool_ (which is NOT a subclass of Python bool, so the previous
        ``isinstance(v, bool)`` check silently dropped it into the string
        branch and then the comparison ``np.bool_(True) == "True"`` evaluated
        to False, flipping every True to False).
      * Python bool.
      * Strings ``"True"/"False"`` (case-insensitive) and ``"1"/"0"``.
      * Numeric 0 / 1.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    raise ValueError(f"Cannot coerce {value!r} (type={type(value).__name__}) to bool")


def _coerce_value(key: str, value: Any) -> Any:
    """Type-coerce a single hyperparam value to the type ``train.py`` expects."""
    if pd.isna(value):
        raise ValueError(f"Hyperparameter {key!r} is NaN in the tune results.")
    if key in _BOOL_ARGS:
        return _coerce_bool(value)
    if key in _INT_ARGS:
        return int(value)
    if key in _FLOAT_ARGS:
        return float(value)
    # strings: norm, nonlin
    return str(value)


def best_config(csv_file: str | Path, metric: str = "val_recon_mse") -> Dict[str, Any]:
    """Read tuning results and return the best-performing hyper-params.

    The returned dict only contains keys in :data:`_TRAIN_ARGS` and is
    type-coerced (so e.g. ``variational`` is a real Python ``bool``, not a
    ``numpy.bool_`` masquerading as a string-shaped object).
    """
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Invalid metric '{metric}'. Must be one of {sorted(_VALID_METRICS)}."
        )

    csv_file = Path(csv_file)
    if not csv_file.is_file():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {csv_file}.")

    # Drop trials with NaN target metric so idxmin/idxmax doesn't pick them.
    df = df.dropna(subset=[metric])
    if df.empty:
        raise ValueError(
            f"No trials with a non-NaN '{metric}' in {csv_file}; "
            "did the tune step produce any successful trials?"
        )

    idx = df[metric].idxmax() if metric in _MAXIMISE else df[metric].idxmin()
    row = df.loc[idx]

    cfg: Dict[str, Any] = {}
    for k in _TRAIN_ARGS:
        if k not in row.index:
            continue
        try:
            cfg[k] = _coerce_value(k, row[k])
        except ValueError as exc:
            raise ValueError(
                f"Failed to parse hyperparameter {k!r} from {csv_file}: {exc}"
            ) from exc

    return cfg


def config_to_cli(cfg: Dict[str, Any]) -> str:
    """Convert a hyper-param dict into a CLI argument string."""

    def _fmt(key, val):
        if isinstance(val, bool):
            return f"--{key} {str(val).lower()}"
        if isinstance(val, float):
            # Use repr to preserve full precision when handing the value back
            # to argparse (str() truncates).
            return f"--{key} {repr(val)}"
        return f"--{key} {val}"

    return " ".join(_fmt(k, v) for k, v in cfg.items())
