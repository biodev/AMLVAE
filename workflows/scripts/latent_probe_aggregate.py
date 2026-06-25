"""Aggregate per-fold latent_probe CSVs and paired t-test VAE vs PCA on test_score."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_METRIC_HIGHER_BETTER = frozenset({"auroc", "accuracy", "f1_macro", "r2"})
_METRIC_LOWER_BETTER = frozenset({"rmse", "mae"})


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Per-fold latent_probe.csv paths (e.g. fold_0/.../latent_probe.csv)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output CSV path (e.g. .../latent_probe/paired_tests.csv)",
    )
    return p.parse_args()


def fold_from_path(path: str) -> str:
    m = re.search(r"fold_(\d+)", str(path))
    if not m:
        raise ValueError(f"Cannot parse fold index from path: {path!r}")
    return m.group(1)


def _metric_direction(metric: str) -> str:
    m = metric.strip().lower()
    if m in _METRIC_HIGHER_BETTER:
        return "higher"
    if m in _METRIC_LOWER_BETTER:
        return "lower"
    raise ValueError(f"Unknown metric {metric!r}; add to higher/lower sets in script.")


def load_all(paths: list[str]) -> pd.DataFrame:
    chunks = []
    for fp in paths:
        p = Path(fp)
        if not p.is_file():
            warnings.warn(f"Missing file, skipping: {fp}")
            continue
        df = pd.read_csv(fp)
        if df.empty:
            warnings.warn(f"Empty CSV, skipping: {fp}")
            continue
        df = df.copy()
        df["fold"] = fold_from_path(fp)
        chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def paired_advantage(vae_score: np.ndarray, pca_score: np.ndarray, metric: str) -> np.ndarray:
    if _metric_direction(metric) == "higher":
        return vae_score - pca_score
    return pca_score - vae_score


def main():
    args = get_args()
    df = load_all(args.inputs)
    if df.empty:
        raise SystemExit("No latent_probe rows loaded; check --inputs paths.")

    need = {"representation", "target", "test_score", "metric", "type", "model"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"latent_probe CSV missing columns: {sorted(miss)}")

    df = df[df["representation"].isin(["vae", "pca"])].copy()
    v = (
        df[df["representation"] == "vae"]
        .drop_duplicates(subset=["fold", "target"])
        .set_index(["fold", "target"])[["test_score", "metric", "type", "model"]]
        .rename(columns={"test_score": "vae_score"})
    )
    p = (
        df[df["representation"] == "pca"]
        .drop_duplicates(subset=["fold", "target"])
        .set_index(["fold", "target"])[["test_score"]]
        .rename(columns={"test_score": "pca_score"})
    )
    paired = v.join(p, how="inner").reset_index()

    only_v = v.index.difference(p.index)
    only_p = p.index.difference(v.index)
    if len(only_v):
        warnings.warn(f"{len(only_v)} fold/target rows have VAE only (no paired PCA); dropped.")
    if len(only_p):
        warnings.warn(f"{len(only_p)} fold/target rows have PCA only (no paired VAE); dropped.")

    rows = []
    for target, g in paired.groupby("target", sort=False):
        if g["metric"].str.lower().nunique() > 1:
            warnings.warn(f"Target {target!r} has multiple metrics across folds; using first row metric.")
        metric = str(g["metric"].iloc[0]).strip().lower()
        typ = str(g["type"].iloc[0])
        mdl = str(g["model"].iloc[0])
        try:
            _metric_direction(metric)
        except ValueError as e:
            warnings.warn(f"Skipping target {target!r}: {e}")
            continue

        vs = g["vae_score"].astype(float).values
        ps = g["pca_score"].astype(float).values
        mask = np.isfinite(vs) & np.isfinite(ps)
        vs, ps = vs[mask], ps[mask]
        n = int(vs.shape[0])
        if n < 2:
            warnings.warn(
                f"Skipping target {target!r}: need >=2 paired folds for t-test (got {n})."
            )
            continue

        adv = paired_advantage(vs, ps, metric)
        tt = stats.ttest_1samp(adv, 0.0, nan_policy="omit")
        t_stat = float(tt.statistic) if np.isfinite(tt.statistic) else float("nan")
        p_two = float(tt.pvalue) if np.isfinite(tt.pvalue) else float("nan")

        rows.append(
            {
                "target": target,
                "type": typ,
                "metric": metric,
                "model": mdl,
                "n_folds_paired": n,
                "mean_test_vae": float(np.mean(vs)),
                "mean_test_pca": float(np.mean(ps)),
                "mean_advantage_vae": float(np.mean(adv)),
                "std_advantage_vae": float(np.std(adv, ddof=1)) if n > 1 else float("nan"),
                "t_statistic": t_stat,
                "pvalue_two_sided": p_two,
                "significant_0.05": bool(np.isfinite(p_two) and p_two < 0.05),
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("No targets with enough paired folds for t-tests.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print()
    print("latent_probe aggregate: paired t-tests (test_score), positive advantage => VAE better")
    print(out_df.to_string(index=False))
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
