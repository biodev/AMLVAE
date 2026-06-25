"""Heatmap of consensus matrix with final cluster ordering and clinical annotations."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--consensus_csv", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--clin_path", required=True)
    p.add_argument("--clin_id_column", default="MLL ID")
    p.add_argument("--annotation_cols", default="[]", help="JSON list of clinical column names")
    p.add_argument("--out_png", required=True)
    p.add_argument("--figsize", default="14,12")
    return p.parse_args()


def load_clin(path: str, id_col: str) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        clin = pd.read_excel(path, sheet_name=0)
    else:
        clin = pd.read_csv(path)
    if id_col not in clin.columns:
        raise KeyError(f"{id_col} not in clinical columns")
    return clin.drop_duplicates(subset=[id_col], keep="first").set_index(id_col)


def stripe_rgb(colors: list) -> np.ndarray:
    """RGB array shape (1, n, 3) for imshow."""
    arr = np.array(colors)[:, :3]
    return arr[None, :, :]


def main():
    args = get_args()
    ann_cols = json.loads(args.annotation_cols)
    if not isinstance(ann_cols, list):
        raise ValueError("annotation_cols must be a JSON list")

    w, h = [float(x) for x in args.figsize.split(",")]

    C = pd.read_csv(args.consensus_csv, index_col=0)
    lab = pd.read_csv(args.labels_csv)
    lab = lab.set_index("sample_id")
    ids = C.index.astype(str)
    lab = lab.reindex(ids)
    if lab["cluster"].isna().any():
        raise ValueError("labels missing for some consensus matrix ids")

    order = np.argsort(lab["cluster"].values * 1_000_000 + np.arange(len(lab)))
    ord_ids = ids.values[order]
    C_ord = C.loc[ord_ids, ord_ids].values.astype(float)

    clin = load_clin(args.clin_path, args.clin_id_column)

    strips = []
    strip_names = []

    clusters_sorted = lab.loc[ord_ids]["cluster"].values
    n_c = int(np.max(clusters_sorted)) + 1 if len(clusters_sorted) else 1
    cmap_c = sns.color_palette("tab10", max(10, n_c))
    strips.append([cmap_c[int(c) % len(cmap_c)] for c in clusters_sorted])
    strip_names.append("cluster")

    for colname in ann_cols:
        if colname not in clin.columns:
            continue
        vals = clin.reindex(ord_ids.astype(str))[colname]
        codes, uniques = pd.factorize(vals.astype(str), sort=True)
        pal = sns.color_palette("husl", max(len(uniques), 1))
        strips.append(
            [pal[c % len(pal)] if c >= 0 else (0.85, 0.85, 0.85) for c in codes]
        )
        strip_names.append(colname[:24])

    n_strip = len(strips)
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(
        n_strip + 1,
        1,
        height_ratios=[0.35] * n_strip + [1],
        hspace=0.06,
    )

    for i, (name, colors) in enumerate(zip(strip_names, strips)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(stripe_rgb(colors), aspect="auto", interpolation="nearest")
        ax.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
        ax.set_yticks([0])
        ax.set_yticklabels([""])
        ax.set_xticks([])

    ax_hm = fig.add_subplot(gs[n_strip, 0])
    sns.heatmap(
        C_ord,
        ax=ax_hm,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "consensus"},
    )
    ax_hm.set_title("Consensus matrix (ordered by final cluster)")

    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
