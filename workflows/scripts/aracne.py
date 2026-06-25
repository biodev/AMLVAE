#!/usr/bin/env python3
"""Build MDS cohort ARACNe regulon from bulk RNA-seq.

Workflow: load expression -> TF list -> threshold -> bootstraps -> consolidate
-> Spearman MoR + likelihood -> save parquet.

Output: data/regulons/regulon_mds_aracne_human_symbol.parquet.gzip
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyviper

from amlvae.aracne import (
    load_aracne_regulon,
    network_to_regulon,
    run_aracne_pipeline,
    save_regulon,
)
from amlvae.aracne.build_regulon import REGULON_CACHE
from amlvae.aracne.run_aracne import DEFAULT_JAR

DEFAULT_EXPR_PATH = (
    "/home/groups/NGSdev/projects/evansmds/mds_data/20241219_WTS_Data_Proj805.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--expr-path", type=Path, default=Path(DEFAULT_EXPR_PATH))
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument(
        "--aracne-out",
        type=Path,
        default=None,
        help="ARACNe working directory (default: <repo>/data/aracne/mds_cohort)",
    )
    parser.add_argument(
        "--regulon-path",
        type=Path,
        default=None,
        help="Output regulon parquet (default: <repo>/data/regulons/<REGULON_CACHE>)",
    )
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--xmx", type=str, default="50G")
    parser.add_argument("--pvalue", type=float, default=1e-8)
    parser.add_argument(
        "--top-k",
        type=int,
        default=7000,
        help="Top variable genes (+ all regulators); use 0 for full transcriptome",
    )
    parser.add_argument("--min-sample-frac", type=float, default=0.05)
    parser.add_argument("--include-cotfs", action="store_true", default=True)
    parser.add_argument("--no-include-cotfs", dest="include_cotfs", action="store_false")
    parser.add_argument("--force-rerun", action="store_true", default=False)
    parser.add_argument("--skip-reload-check", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba"))

    repo_root = args.repo_root.resolve()
    aracne_out = args.aracne_out or (repo_root / "data" / "aracne" / "mds_cohort")
    regulon_path = args.regulon_path or (repo_root / "data" / "regulons" / REGULON_CACHE)
    top_k = args.top_k if args.top_k > 0 else None

    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    print(result.stderr.split("\n")[0])
    assert args.jar.exists(), f"Missing ARACNe JAR: {args.jar}"
    print(f"ARACNe JAR: {args.jar}")

    counts = (
        pd.read_csv(args.expr_path, sep="\t")
        .pivot(index="array_id", columns="gene_id", values="counts")
        .fillna(0)
    )
    log2_expr = np.log2(counts + 1)

    min_samples = max(1, int(args.min_sample_frac * log2_expr.shape[0]))
    expressed = (counts > 0).sum(axis=0) >= min_samples
    log2_expr = log2_expr.loc[:, expressed]
    print(f"{log2_expr.shape[1]} genes x {log2_expr.shape[0]} samples (after expression filter)")

    tfs = set(pyviper.load.TFs("human"))
    if args.include_cotfs:
        tfs |= set(pyviper.load.coTFs("human"))

    gene_set = {str(g) for g in log2_expr.columns}
    regulators = sorted(tfs & gene_set)
    print(f"{len(regulators)} regulators (of {len(tfs)} TFs/coTFs in pyviper)")

    if top_k is not None:
        top_genes = set(log2_expr.var(axis=0).nlargest(top_k).index.astype(str))
        keep = top_genes | set(regulators)
        log2_expr = log2_expr.loc[:, [g for g in log2_expr.columns if str(g) in keep]]
        print(
            f"TOP_K={top_k}: {len(top_genes)} variable genes + regulators -> "
            f"{log2_expr.shape[1]} genes"
        )

    expr_gxs = log2_expr.T
    print(f"{expr_gxs.shape[0]} genes x {expr_gxs.shape[1]} samples")

    network_path = run_aracne_pipeline(
        expr_gxs,
        regulators,
        aracne_out,
        jar=args.jar,
        n_bootstraps=args.n_bootstraps,
        pvalue=args.pvalue,
        threads=args.threads,
        xmx=args.xmx,
        force=args.force_rerun,
    )
    print(f"Consolidated network: {network_path}")

    net_table, interactome = network_to_regulon(network_path, expr_gxs, name="regulon_mds")
    save_regulon(net_table, regulon_path)

    print(f"Regulators: {len(interactome.get_reg_names())}")
    print(f"Edges: {len(net_table):,}")
    print(f"Median targets/regulator: {net_table.groupby('regulator').size().median():.0f}")
    print(f"Wrote {regulon_path}")

    if not args.skip_reload_check:
        loaded = load_aracne_regulon(repo_root)
        print(f"Loaded {len(loaded.get_reg_names())} regulators, {len(loaded.net_table):,} edges")


if __name__ == "__main__":
    main()
