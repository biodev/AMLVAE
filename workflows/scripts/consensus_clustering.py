"""
Consensus Clustering using Louvain Community Detection on KNN Graphs

This script performs graph-based clustering on latent representations using 
Louvain community detection on k-nearest neighbor graphs.

IMPORTANT CONSIDERATIONS:
- KNN graph construction is sensitive to the choice of k (neighborhood size)
  and resolution parameter. Small k values create sparser graphs while large k
  creates denser, more connected graphs.
- The Louvain algorithm with different resolution parameters can produce 
  different numbers of clusters. Low resolution -> fewer clusters.
- Silhouette scores can be misleading in high-dimensional spaces and when
  clusters have irregular shapes.
- This method assumes Euclidean distance is appropriate for the latent space.
  For other geometries, consider alternative distance metrics.

DRAWBACKS:
- Results are sensitive to random seed in Louvain algorithm
- Silhouette score computation scales O(n^2) with number of samples
- May produce different cluster numbers with different parameter choices
- Does not provide uncertainty estimates for cluster assignments
"""

import pandas as pd
import numpy as np
import argparse
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def get_args():
    """Parse command line arguments."""
    
    argparser = argparse.ArgumentParser(
        description='Consensus clustering using Louvain on KNN graphs'
    )
    
    argparser.add_argument(
        '--latent_csv', 
        type=str, 
        required=True,
        help='Path to latent space CSV file (index = sample IDs)'
    )
    argparser.add_argument(
        '--out', 
        type=str, 
        required=True,
        help='Output path for cluster assignments CSV file'
    )
    argparser.add_argument(
        '--out_metrics', 
        type=str, 
        required=True,
        help='Output path for cluster metrics CSV file'
    )
    argparser.add_argument(
        '--k', 
        type=int, 
        required=True,
        help='Number of neighbors for KNN graph'
    )
    argparser.add_argument(
        '--resolution', 
        type=float, 
        required=True,
        help='Resolution parameter for Louvain clustering'
    )
    argparser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    return argparser.parse_args()


def load_latent_space(latent_csv):
    """
    Load latent space data.
    
    Parameters
    ----------
    latent_csv : str
        Path to latent space CSV file
        
    Returns
    -------
    z_df : pd.DataFrame
        Latent space features (samples x features)
    """
    z_df = pd.read_csv(latent_csv, index_col=0)
    print(f"Loaded {len(z_df)} samples with {z_df.shape[1]} latent dimensions")
    return z_df


def build_knn_graph(z_matrix, k):
    """
    Build k-nearest neighbor graph from latent space.
    
    Parameters
    ----------
    z_matrix : np.ndarray
        Latent space matrix (samples x features)
    k : int
        Number of nearest neighbors
        
    Returns
    -------
    G : networkx.Graph
        KNN graph
    """
    # Build KNN adjacency matrix
    adj_matrix = kneighbors_graph(
        z_matrix, 
        n_neighbors=k, 
        mode='connectivity',
        include_self=False,
        metric='euclidean'
    )
    
    # Convert to undirected graph (symmetric)
    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix.data = np.ones_like(adj_matrix.data)  # Binary edges
    
    # Convert to NetworkX graph
    G = nx.from_scipy_sparse_array(adj_matrix)
    
    return G


def cluster_louvain(G, resolution, seed):
    """
    Perform Louvain clustering on graph.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    resolution : float
        Resolution parameter for Louvain algorithm
    seed : int
        Random seed
        
    Returns
    -------
    clusters : np.ndarray
        Cluster assignments (0-indexed)
    """
    # Louvain clustering
    communities = nx.community.louvain_communities(
        G, 
        resolution=resolution, 
        seed=seed
    )
    
    # Convert to array
    n_nodes = G.number_of_nodes()
    clusters = np.zeros(n_nodes, dtype=int)
    
    for cluster_id, community in enumerate(communities):
        for node in community:
            clusters[node] = cluster_id
    
    return clusters


def evaluate_clustering(z_matrix, clusters):
    """
    Evaluate clustering using silhouette score.
    
    Parameters
    ----------
    z_matrix : np.ndarray
        Latent space matrix
    clusters : np.ndarray
        Cluster assignments
        
    Returns
    -------
    metrics : dict
        Dictionary with evaluation metrics
    """
    n_clusters = len(np.unique(clusters))
    
    metrics = {'n_clusters': n_clusters}
    
    # Silhouette score (only if >1 cluster)
    if n_clusters > 1:
        sil_score = silhouette_score(z_matrix, clusters, metric='euclidean')
        metrics['avg_silhouette'] = sil_score
    else:
        metrics['avg_silhouette'] = np.nan
    
    return metrics


def perform_clustering(z_df, k, resolution, seed):
    """
    Perform clustering with specified parameters.
    
    Parameters
    ----------
    z_df : pd.DataFrame
        Latent space features
    k : int
        Number of neighbors for KNN graph
    resolution : float
        Resolution parameter for Louvain
    seed : int
        Random seed
        
    Returns
    -------
    cluster_df : pd.DataFrame
        DataFrame with cluster assignments and sample IDs
    metrics : dict
        Dictionary with clustering metrics
    """
    z_matrix = z_df.values
    
    print(f"\nPerforming clustering with k={k}, resolution={resolution}...")
    
    # Build graph and cluster
    print("  Building KNN graph...")
    G = build_knn_graph(z_matrix, k)
    
    print("  Running Louvain clustering...")
    clusters = cluster_louvain(G, resolution, seed)
    
    # Evaluate
    print("  Evaluating clustering quality...")
    metrics = evaluate_clustering(z_matrix, clusters)
    
    print(f"  -> {metrics['n_clusters']} clusters identified")
    if metrics['n_clusters'] > 1:
        print(f"  -> Avg silhouette score: {metrics['avg_silhouette']:.3f}")
    
    # Create output dataframe
    cluster_df = pd.DataFrame({
        'sample_id': z_df.index,
        'cluster': clusters
    })
    
    return cluster_df, metrics


def main():
    """Main execution function."""
    
    print()
    print('---------------------------------------------')
    print('Consensus Clustering with Louvain on KNN')
    print('---------------------------------------------')
    print()
    
    args = get_args()
    print('Arguments:')
    for arg, value in vars(args).items():
        print(f'  {arg}: {value}')
    print('---------------------------------------------')
    
    # Load data
    z_df = load_latent_space(args.latent_csv)
    
    # Perform clustering
    cluster_df, metrics = perform_clustering(z_df, args.k, args.resolution, args.seed)
    
    # Save cluster assignments
    cluster_df.to_csv(args.out, index=False)
    print(f"\nCluster assignments saved to: {args.out}")
    
    # Save clustering metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df['k'] = args.k
    metrics_df['resolution'] = args.resolution
    metrics_df.to_csv(args.out_metrics, index=False)
    print(f"Cluster metrics saved to: {args.out_metrics}")
    
    print()
    print('---------------------------------------------')
    print('Clustering complete.')
    print('---------------------------------------------')
    print()


if __name__ == '__main__':
    main()
