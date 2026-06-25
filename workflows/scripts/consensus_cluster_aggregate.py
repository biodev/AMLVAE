"""Aggregate per-fold connectivity matrices → consensus C and final cluster labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="connectivity NPZ files (one per fold)")
    p.add_argument(
        "--partition_method",
        choices=("louvain", "elbow_agglomerative"),
        default="louvain",
    )
    p.add_argument("--consensus_k_nn", type=int, default=15)
    p.add_argument("--consensus_graph_union", type=str, default="true")
    p.add_argument(
        "--consensus_edge_weight",
        choices=("identity", "inv_dist"),
        default="identity",
        help="edge weight from C (identity) or from inv(1+dist) on consensus kNN",
    )
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--k_min", type=int, default=3)
    p.add_argument("--k_max", type=int, default=12)
    p.add_argument("--linkage", default="average", help="average|complete|single for elbow mode")
    p.add_argument(
        "--elbow_metric",
        choices=("silhouette", "within_disagreement"),
        default="silhouette",
    )
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def _parse_bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes")


def load_fold(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    M = data["connectivity"].astype(np.float64)
    ids = data["sample_ids"].astype(str)
    return M, ids


def _permute_connectivity(M: np.ndarray, ids: list[str], target_order: list[str]) -> np.ndarray:
    """Reorder rows/columns so sample ids match target_order (must be the same set as ids)."""
    if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("connectivity must be square")
    if len(ids) != M.shape[0]:
        raise ValueError("sample_ids length does not match connectivity shape")
    s_ids, s_tgt = set(ids), set(target_order)
    if s_ids != s_tgt:
        only_a = sorted(s_ids - s_tgt)[:20]
        only_b = sorted(s_tgt - s_ids)[:20]
        raise ValueError(
            f"sample_id set mismatch: {len(only_a)} only in matrix / {len(only_b)} only in target "
            f"(examples: …{only_a!r} vs …{only_b!r})"
        )
    if len(ids) != len(s_ids):
        raise ValueError("duplicate sample_ids in fold connectivity")
    pos = {s: i for i, s in enumerate(ids)}
    perm = np.array([pos[s] for s in target_order], dtype=np.intp)
    return M[perm][:, perm]


def knn_adjacency_from_dist(D: np.ndarray, k_nn: int, graph_union: bool) -> np.ndarray:
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


def louvain_labels(G: nx.Graph, seed: int | None) -> np.ndarray:
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=1.0)
    labels = np.empty(G.number_of_nodes(), dtype=np.int32)
    for ci, nodes in enumerate(comms):
        for u in nodes:
            labels[u] = ci
    return labels


def within_cluster_disagreement(C: np.ndarray, labels: np.ndarray) -> float:
    n = C.shape[0]
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                total += 1.0 - C[i, j]
                pairs += 1
    return total / max(1, pairs)


def main():
    args = get_args()
    union = _parse_bool(args.consensus_graph_union)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Ms = []
    canonical_ids: list[str] | None = None
    for fp in args.inputs:
        M, ids = load_fold(fp)
        ids_list = ids.tolist()
        if canonical_ids is None:
            canonical_ids = sorted(ids_list)
        s_here, s_can = set(ids_list), set(canonical_ids)
        if s_here != s_can:
            only_here = sorted(s_here - s_can)[:15]
            only_can = sorted(s_can - s_here)[:15]
            raise ValueError(
                f"sample_id set mismatch in {fp}: extra in file {only_here!r}, "
                f"missing vs canonical {only_can!r}"
            )
        if len(ids_list) != len(s_here):
            raise ValueError(f"duplicate sample_ids in {fp}")
        M = _permute_connectivity(M, ids_list, canonical_ids)
        Ms.append(M)

    C = np.mean(np.stack(Ms, axis=0), axis=0).astype(np.float64)
    np.fill_diagonal(C, 1.0)

    id_index = canonical_ids
    df_c = pd.DataFrame(C, index=id_index, columns=id_index)
    df_c.to_csv(out_dir / "consensus_matrix.csv")

    n = C.shape[0]
    D_cons = 1.0 - C
    np.fill_diagonal(D_cons, 0.0)

    if args.partition_method == "louvain":
        undir = knn_adjacency_from_dist(D_cons, args.consensus_k_nn, union)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        ri, ci = np.where(undir)
        for i, j in zip(ri.tolist(), ci.tolist()):
            if i >= j:
                continue
            if args.consensus_edge_weight == "identity":
                w = max(0.0, float(C[i, j]))
            else:
                w = 1.0 / (1.0 + float(D_cons[i, j]))
            G.add_edge(i, j, weight=w)
        labs = louvain_labels(G, args.random_state)
        k_star = int(len(np.unique(labs)))
        Path(out_dir / "k_choice.json").write_text(
            json.dumps({"method": "louvain", "n_communities": k_star}),
            encoding="utf-8",
        )
    else:
        condensed = squareform(D_cons, checks=False)
        Z = linkage(condensed, method=args.linkage)
        rows = []
        best_k, best_score = None, -np.inf
        for k in range(args.k_min, args.k_max + 1):
            lab = fcluster(Z, k, criterion="maxclust").astype(np.int32) - 1
            if args.elbow_metric == "silhouette":
                if len(np.unique(lab)) < 2 or len(np.unique(lab)) >= n:
                    score = -1.0
                else:
                    score = float(silhouette_score(D_cons, lab, metric="precomputed"))
            else:
                score = -within_cluster_disagreement(C, lab)
            rows.append({"k": k, "score": score})
            if score > best_score:
                best_score = score
                best_k = k

        pd.DataFrame(rows).to_csv(out_dir / "k_sweep.csv", index=False)
        labs = fcluster(Z, best_k, criterion="maxclust").astype(np.int32) - 1
        Path(out_dir / "k_choice.json").write_text(
            json.dumps(
                {
                    "method": "elbow_agglomerative",
                    "k": int(best_k),
                    "metric": args.elbow_metric,
                    "best_score": float(best_score),
                }
            ),
            encoding="utf-8",
        )

    out_lab = pd.DataFrame({"sample_id": id_index, "cluster": labs})
    out_lab.to_csv(out_dir / "final_labels.csv", index=False)
    print(f"wrote {out_dir}/consensus_matrix.csv and final_labels.csv")


if __name__ == "__main__":
    main()
