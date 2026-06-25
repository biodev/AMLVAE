"""Per-fold consensus clustering: build kNN graph, Louvain, save co-assignment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import pairwise_distances


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dist_csv", default="", help="precomputed distance CSV (if distance_source dist_csv)")
    p.add_argument("--z_csv", default="", help="latent z CSV (if distance_source z_csv)")
    p.add_argument("--distance_source", choices=("dist_csv", "z_csv"), required=True)
    p.add_argument("--pairwise_metric", default="euclidean")
    p.add_argument("--k_nn", type=int, default=15)
    p.add_argument("--graph_union", type=str, default="true", help="true=union kNN, false=mutual")
    p.add_argument(
        "--edge_weight",
        choices=("jaccard", "rbf", "inv_dist"),
        default="inv_dist",
    )
    p.add_argument("--rbf_sigma_scale", type=float, default=1.0, help="sigma = scale * median(D)")
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--out_meta", default="", help="optional JSON path with n_communities")
    return p.parse_args()


def _parse_bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes")


def load_distance_matrix(args) -> tuple[np.ndarray, list[str]]:
    if args.distance_source == "dist_csv":
        df = pd.read_csv(args.dist_csv, index_col=0)
        ids = df.index.astype(str).tolist()
        D = df.values.astype(np.float64)
        np.fill_diagonal(D, 0.0)
        return D, ids
    z = pd.read_csv(args.z_csv, index_col=0)
    ids = z.index.astype(str).tolist()
    Z = z.values.astype(np.float64)
    D = pairwise_distances(Z, metric=args.pairwise_metric)
    return D, ids


def knn_adjacency(D: np.ndarray, k_nn: int, graph_union: bool) -> np.ndarray:
    n = D.shape[0]
    nn_idx = np.argsort(D, axis=1)[:, 1 : k_nn + 1]
    directed = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    directed[rows, nn_idx] = True
    if graph_union:
        undir = directed | directed.T
    else:
        undir = directed & directed.T
    np.fill_diagonal(undir, False)
    return undir


def edge_weights(D: np.ndarray, undir: np.ndarray, mode: str, rbf_sigma_scale: float) -> dict[tuple[int, int], float]:
    n = D.shape[0]
    wdict: dict[tuple[int, int], float] = {}
    nz = D[np.triu_indices(n, 1)]
    nz = nz[nz > 0]
    med = float(np.median(nz)) if len(nz) else 1.0
    sigma = rbf_sigma_scale * med

    rows, cols = np.where(undir)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i >= j:
            continue
        dij = float(D[i, j])
        if mode == "inv_dist":
            w = 1.0 / (1.0 + dij)
        elif mode == "rbf":
            w = float(np.exp(-(dij**2) / (2.0 * sigma**2 + 1e-12)))
        else:
            ni = set(np.where(undir[i])[0])
            nj = set(np.where(undir[j])[0])
            inter = len(ni & nj)
            union = len(ni | nj)
            w = inter / max(1, union)
        wdict[(i, j)] = w
        wdict[(j, i)] = w
    return wdict


def louvain_labels(G: nx.Graph, seed: int | None) -> np.ndarray:
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=1.0)
    labels = np.empty(G.number_of_nodes(), dtype=np.int32)
    for ci, nodes in enumerate(comms):
        for u in nodes:
            labels[u] = ci
    return labels


def connectivity_from_labels(labels: np.ndarray) -> np.ndarray:
    return (labels[:, None] == labels[None, :]).astype(np.float32)


def main():
    args = get_args()
    graph_union = _parse_bool(args.graph_union)

    if args.distance_source == "dist_csv" and not args.dist_csv:
        raise ValueError("dist_csv required when distance_source=dist_csv")
    if args.distance_source == "z_csv" and not args.z_csv:
        raise ValueError("z_csv required when distance_source=z_csv")

    D, ids = load_distance_matrix(args)
    n = len(ids)
    undir = knn_adjacency(D, args.k_nn, graph_union)
    wmap = edge_weights(D, undir, args.edge_weight, args.rbf_sigma_scale)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), w in wmap.items():
        if i < j:
            G.add_edge(i, j, weight=float(w))

    labs = louvain_labels(G, args.random_state)
    M = connectivity_from_labels(labs)
    n_comm = int(len(np.unique(labs)))

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        connectivity=M,
        sample_ids=np.array(ids, dtype=object),
    )

    if args.out_meta:
        Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
        meta = {"n_communities": n_comm, "n_samples": n}
        Path(args.out_meta).write_text(json.dumps(meta), encoding="utf-8")

    print(f"wrote {args.out_npz} communities={n_comm}")


if __name__ == "__main__":
    main()
