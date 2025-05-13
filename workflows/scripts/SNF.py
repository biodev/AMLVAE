'''
similarity network fusion (SNF)

B Wang, A Mezlini, F Demir, M Fiume, T Zu, M Brudno, B Haibe-Kains, A Goldenberg (2014) Similarity Network Fusion: a fast and effective method to aggregate multiple data types on a genome wide scale. Nature Methods. Online. Jan 26, 2014  
'''


import pandas as pd 
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix
import argparse 
import networkx as nx 
from matplotlib import pyplot as plt
import pickle as pkl

def get_args(): 

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--run_dir', type=str, default='../../runs/',
                            help='path to run dir')
    argparser.add_argument('--out', type=str, default='../output/',
                            help='path to output dir')
    argparser.add_argument('--dataset', type=str, default='aml',
                            help='dataset name')
    argparser.add_argument('--num_folds', type=int, default=5,
                            help='number of folds')
    argparser.add_argument('--k', type=int, default=20,
                            help='number of nearest neighbours')
    argparser.add_argument('--mu', type=float, default=0.5,
                            help='scaling factor for adaptive Gaussian kernel bandwidth')
    argparser.add_argument('--T', type=int, default=20,
                            help='number of cross-network diffusion iterations')
    argparser.add_argument('--edge_thr_q', type=float, default=0.95,
                            help='post-fusion pruning threshold quantile for edge weights')
    argparser.add_argument('--seed', type=int, default=42,
                            help='random seed for reproducibility')

    return argparser.parse_args()


def _affinity_matrix_union_knn(X: np.ndarray, k: int = 20, mu: float = 0.5):
    """
    Union‑kNN adaptive‑bandwidth Gaussian affinity (Wang et al., 2014 SNF).
    """
    D2 = pairwise_distances(X, metric="euclidean", squared=True)
    n  = D2.shape[0]

    idx_sorted = np.argsort(D2, axis=1)
    sigmas = mu * np.sqrt(
        np.mean(D2[np.arange(n)[:, None], idx_sorted[:, 1 : k + 1]], axis=1) + 1e-12
    )
    sigma_mat = sigmas[:, None] * sigmas[None, :]
    W = np.exp(-D2 / (sigma_mat + 1e-12))

    # -------- union (not mutual) k‑NN --------
    mask = np.zeros_like(W, dtype=bool)
    for i in range(n):
        mask[i, idx_sorted[i, 1 : k + 1]] = True
    mask = mask | mask.T     # union
    W[~mask] = 0.0
    np.fill_diagonal(W, 0.0)

    # symmetric normalisation:  P = D^{-1/2} W D^{-1/2}
    deg = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
    P = D_inv_sqrt @ W @ D_inv_sqrt
    return P


def similarity_network_fusion(
    df_list: list[pd.DataFrame],
    k: int = 20,
    mu: float = 0.5,
    T: int = 20,
    edge_thr_q: float = 0.95,
):
    """
    Fuse multiple patient‑by‑feature embeddings into a single similarity graph
    using the Similarity Network Fusion (SNF) algorithm.

    Parameters
    ----------
    k : int, default = 20
        Number of nearest neighbours retained when building the initial
        affinity matrix for each view (fold or data type).

        * **Interpretation** – Controls graph sparsity and local context size.
          Each sample keeps edges to its `k` closest peers (union‑kNN rule).
        * **Typical range** – 10 ≤ k ≤ 30 for n ≈ 100 – 10 000 patients.
          Larger k makes the graph denser and can capture long‑range
          structure, but increases memory and may blur local clusters.
        * **Tip** – If the embeddings vary greatly in density across views,
          pick the *largest* k that still yields connected graphs in *all*
          views, or set k proportional to `log(n)`.

    mu : float, default = 0.5
        Scaling factor for the adaptive Gaussian kernel bandwidth σᵢ used to
        convert Euclidean distances to similarities.

        * **Computation** – For each sample *i*,  
          σᵢ = μ × mean(distance to its `k` nearest neighbours).
        * **Effect** – Smaller μ tightens kernels (sharper local similarity),
          emphasising only the closest neighbours. Larger μ smooths distances
          and keeps more medium‑range similarities.
        * **Typical values** – 0.3 – 0.8.  The algorithm is fairly robust as
          long as μ is not ‹ 0.1 (too sharp) or › 1.0 (too flat).

    T : int, default = 20
        Number of cross‑network diffusion iterations.

        * **Mechanism** – At each step, every view exchanges information with
          the average of the others (`P_v ← P_v ⋅ P̄_¬v ⋅ P_vᵀ`) and then
          re‑normalises.  Repeating `T` times yields convergence toward a
          consensus network that retains view‑specific nuances.
        * **Guideline** – 10 ≤ T ≤ 30 almost always suffices; extra iterations
          add compute cost but give diminishing changes after the first ~10.

    edge_thr : float, default = 0.05
        Post‑fusion pruning threshold quantile for edge weights in the final graph.

        * **Purpose** – Removes tiny similarities that are numerically non‑zero
          but negligible, producing a sparse COO representation friendly to
          graph libraries (e.g. PyTorch Geometric).
        * **Choosing a value** –  
          – Keep `edge_thr = 0.0` for a fully weighted graph.  
          – Set to 1e‑3 – 1e‑2 if memory is a concern or you want ≤ k edges per
            node.  Inspect `edge_weight` histogram to pick a sensible cut‑off.

    Returns
    -------
    edge_index : ndarray, shape (2, E)
        Row‑major COO index of the fused graph, where `edge_index[0]`
        contains source nodes and `edge_index[1]` contains target nodes.
    edge_weight : ndarray, shape (E,)
        Symmetric, non‑negative similarity weights corresponding to the
        edges in `edge_index`.

    Notes
    -----
    • All DataFrames in `df_list` must have identical row order (patients).  
    • The implementation follows Wang *et al.* 2014 (Nature Methods) exactly:
      union‑kNN sparsification, adaptive bandwidth Gaussian kernel,
      symmetric normalisation, and iterative cross‑view diffusion.
    """
    Ps = [_affinity_matrix_union_knn(df.values.astype(float), k, mu) for df in df_list]
    V  = len(Ps)
    n  = Ps[0].shape[0]

    for tt in range(T):
        print(f'SNF iteration {tt + 1}/{T}...', end='\r')
        new_Ps = []
        for v in range(V):
            P_bar = sum(Ps[u] for u in range(V) if u != v) / (V - 1)
            P_new = Ps[v] @ P_bar @ Ps[v].T
            # re‑normalise symmetrically
            deg = P_new.sum(axis=1)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
            P_new = D_inv_sqrt @ P_new @ D_inv_sqrt
            np.fill_diagonal(P_new, 0.0)
            new_Ps.append(P_new)
        Ps = new_Ps
    print()

    W_fused = sum(Ps) / V
    W_fused = (W_fused + W_fused.T) / 2

    edge_thr = np.quantile(W_fused.ravel(), edge_thr_q).item() 
    print(f'edge weight threshold [quantile: {edge_thr_q:.2f}]: {edge_thr}')

    W_fused[W_fused < edge_thr] = 0.0

    coo = coo_matrix(W_fused)
    edge_index  = np.vstack((coo.row, coo.col)).astype(np.int64)
    edge_weight = coo.data.astype(np.float32)
    return edge_index, edge_weight


def load(args): 

    '''
    directory structure: 

    runs/
        ├── <run_id> 
            ├── fold_0 
                ├── distance 
                    ├── <dataset>_dist.csv 
            ├── fold_1 
                ... 
            ... 
    '''
    dists = [] 
    for fold in range(args.num_folds): 
        dists.append( pd.read_csv(f'{args.run_dir}/fold_{fold}/distance/{args.dataset}_dist.csv', index_col=0) ) 

     # order the dataframes rows/cols by same id order 
    id_order = dists[0].index

    dists2 = []
    for df in dists: 
        df = df.reindex(index=id_order, columns=id_order)
        dists2.append(df) 
    dists = dists2

     # check that all dataframes have the same order 
    ids_x = dists[0].index
    ids_y = dists[0].columns 
    assert (ids_x == ids_y).all(), 'DataFrames have different order of rows and columns'
    for df in dists[1:]: 
        assert (df.index == ids_x).all(), 'DataFrames have different order of rows'
        assert (df.columns == ids_y).all(), 'DataFrames have different order of columns'

    return dists, id_order


if __name__ == '__main__': 

    print()
    print('---------------------------------------------')
    print('VAE: similarity network fusion (SNF)')
    print('---------------------------------------------')
    print() 
    print('arguments:')
    args = get_args()
    print(args)
    print('---------------------------------------------')

    dists, ids = load(args)

    edge_index, edge_weight = similarity_network_fusion(
        dists, k=args.k, mu=args.mu, T=args.T, edge_thr_q=args.edge_thr_q
    )

    # save edge_index and edge_weight
    np.save(f'{args.out}/{args.dataset}_edge_index.npy', edge_index) 
    np.save(f'{args.out}/{args.dataset}_edge_weight.npy', edge_weight)
    np.save(f'{args.out}/{args.dataset}_node_ids.npy', ids)

    # make nx graph and save a plot to file 
    G = nx.Graph() 
    G.add_nodes_from(range(dists[0].shape[0]), label=ids)

    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i], weight=edge_weight[i])

    # network metrics 
    print()
    print('----------------------------------------------')
    print('SNF graph statistics:')
    print('\tNumber of nodes:', G.number_of_nodes())
    print('\tNumber of isolates:', len(list(nx.isolates(G))))
    print('\tNumber of edges:', G.number_of_edges())
    print('\tDensity:', nx.density(G))
    print('\tAverage clustering coefficient:', nx.average_clustering(G))
    print('\tAverage degree:', np.mean([d for n, d in G.degree()]))
    try: 
        print('\tAverage shortest path length:', nx.average_shortest_path_length(G))
    except: 
        print('\tAverage shortest path length: not connected')
    try: 
        print('\tDiameter:', nx.diameter(G))
    except:
        print('\tDiameter: not connected')

    print('\tAssortativity coefficient:', nx.degree_assortativity_coefficient(G))
    print('----------------------------------------------')
    print()

    plt.figure() 
    plt.title(f'SNF graph for {args.dataset}')
    plt.axis('off')
    pos = nx.spring_layout(G, seed=args.seed)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.5)
    nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.5)
    plt.savefig(f'{args.out}/{args.dataset}_snf_graph.png', dpi=300, bbox_inches='tight') 
    plt.close() 

    # save nx graph to file
    pkl.dump(G, open(f'{args.out}/{args.dataset}_snf_nxgraph.pkl', 'wb'))

    print() 
    print('SNF complete.')
    print('---------------------------------------------')







    


    