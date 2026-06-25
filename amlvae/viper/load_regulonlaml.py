"""Load TCGA LAML regulon (regulonlaml) for pyviper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyviper import Interactome

REGULON_CACHE = "regulonlaml_human_symbol.parquet.gzip"


def load_regulonlaml(repo_root: Path | str | None = None) -> Interactome:
    """Load pre-built TCGA AML interactome (human symbols) from repo cache."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    cache_path = Path(repo_root) / "data" / "regulons" / REGULON_CACHE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Build once with workflows/scripts/build_regulonlaml_cache.py"
        )
    return Interactome("regulonlaml", pd.read_parquet(cache_path))
