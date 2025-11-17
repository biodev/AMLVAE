"""
Aggregate Cluster Assignments Across Folds to Create Consensus Matrix

This script takes cluster assignments from multiple folds and creates a 
consensus co-occurrence matrix that counts how many times each pair of 
samples appears in the same cluster across all folds.

The consensus matrix can be used for:
- Identifying stable cluster memberships
- Downstream consensus clustering algorithms
- Assessing robustness of cluster assignments
"""

import pandas as pd
import numpy as np
import argparse
import glob
import os


def get_args():
    """Parse command line arguments."""
    
    argparser = argparse.ArgumentParser(
        description='Aggregate cluster assignments to create consensus matrix'
    )
    
    argparser.add_argument(
        '--cluster_files', 
        type=str, 
        nargs='+',
        required=True,
        help='List of cluster assignment CSV files'
    )
    argparser.add_argument(
        '--out', 
        type=str, 
        required=True,
        help='Output path for consensus co-occurrence matrix CSV'
    )
    
    return argparser.parse_args()


def load_cluster_assignments(file_list):
    """
    Load cluster assignments from multiple files.
    
    Parameters
    ----------
    file_list : list of str
        List of paths to cluster assignment files
        
    Returns
    -------
    cluster_dfs : list of pd.DataFrame
        List of dataframes with cluster assignments
    """
    files = sorted(file_list)
    
    if len(files) == 0:
        raise ValueError(f"No files provided")
    
    print(f"Found {len(files)} cluster assignment files")
    
    cluster_dfs = []
    for f in files:
        if not os.path.exists(f):
            raise ValueError(f"File does not exist: {f}")
        df = pd.read_csv(f)
        if 'sample_id' not in df.columns or 'cluster' not in df.columns:
            raise ValueError(f"File {f} must contain 'sample_id' and 'cluster' columns")
        cluster_dfs.append(df)
    
    return cluster_dfs


def create_cooccurrence_matrix(cluster_df):
    """
    Create co-occurrence matrix for a single fold.
    
    For each fold, creates a sample x sample matrix where entry (i,j) = 1
    if samples i and j are in the same cluster, 0 otherwise.
    
    Parameters
    ----------
    cluster_df : pd.DataFrame
        DataFrame with 'sample_id' and 'cluster' columns
        
    Returns
    -------
    cooc_matrix : pd.DataFrame
        Sample x sample co-occurrence matrix (binary)
    """
    # Get unique samples and clusters
    samples = cluster_df['sample_id'].values
    clusters = cluster_df['cluster'].values
    
    n_samples = len(samples)
    
    # Create binary co-occurrence matrix
    cooc = np.zeros((n_samples, n_samples), dtype=int)
    
    # For each cluster, set all pairs in that cluster to 1
    for cluster_id in np.unique(clusters):
        # Get indices of samples in this cluster
        in_cluster = np.where(clusters == cluster_id)[0]
        
        # Set all pairs in this cluster to 1
        for i in in_cluster:
            for j in in_cluster:
                cooc[i, j] = 1
    
    # Convert to DataFrame with sample IDs as index/columns
    cooc_df = pd.DataFrame(cooc, index=samples, columns=samples)
    
    return cooc_df


def aggregate_consensus(cluster_dfs):
    """
    Aggregate co-occurrence matrices across all folds.
    
    Parameters
    ----------
    cluster_dfs : list of pd.DataFrame
        List of cluster assignment dataframes
        
    Returns
    -------
    consensus_matrix : pd.DataFrame
        Sample x sample consensus matrix (counts of co-occurrences)
    """
    print("\nCreating co-occurrence matrices for each fold...")
    
    # Create co-occurrence matrix for each fold
    cooc_matrices = []
    all_sample_sets = []
    
    for i, df in enumerate(cluster_dfs):
        print(f"  Processing fold {i+1}/{len(cluster_dfs)}...")
        cooc = create_cooccurrence_matrix(df)
        cooc_matrices.append(cooc)
        all_sample_sets.append(set(cooc.index))
    
    # Check that all matrices have the same samples (regardless of order)
    first_samples = all_sample_sets[0]
    for i, sample_set in enumerate(all_sample_sets[1:], 1):
        if first_samples != sample_set:
            missing_in_fold = first_samples - sample_set
            extra_in_fold = sample_set - first_samples
            error_msg = f"Fold {i} has different samples than fold 0.\n"
            if missing_in_fold:
                error_msg += f"  Missing {len(missing_in_fold)} samples in fold {i}\n"
            if extra_in_fold:
                error_msg += f"  Extra {len(extra_in_fold)} samples in fold {i}\n"
            raise ValueError(error_msg)
    
    print("\nAggregating consensus matrix...")
    
    # Get common sample order (sorted for consistency)
    common_samples = sorted(first_samples)
    print(f"  Found {len(common_samples)} common samples across all folds")
    
    # Reindex all matrices to the same sample order
    print("  Reindexing matrices to common sample order...")
    cooc_matrices_reindexed = []
    for cooc in cooc_matrices:
        cooc_reindexed = cooc.reindex(index=common_samples, columns=common_samples, fill_value=0)
        cooc_matrices_reindexed.append(cooc_reindexed)
    
    # Sum all co-occurrence matrices
    consensus = cooc_matrices_reindexed[0].copy()
    for cooc in cooc_matrices_reindexed[1:]:
        consensus = consensus + cooc
    
    print(f"  Consensus matrix shape: {consensus.shape}")
    print(f"  Min co-occurrence: {consensus.values.min()}")
    print(f"  Max co-occurrence: {consensus.values.max()}")
    print(f"  Mean co-occurrence: {consensus.values.mean():.2f}")
    
    return consensus


def main():
    """Main execution function."""
    
    print()
    print('---------------------------------------------')
    print('Aggregate Consensus Clustering')
    print('---------------------------------------------')
    print()
    
    args = get_args()
    print('Arguments:')
    for arg, value in vars(args).items():
        print(f'  {arg}: {value}')
    print('---------------------------------------------')
    
    # Load cluster assignments
    cluster_dfs = load_cluster_assignments(args.cluster_files)
    
    # Create consensus matrix
    consensus_matrix = aggregate_consensus(cluster_dfs)
    
    # Save consensus matrix
    print(f"\nSaving consensus matrix to: {args.out}")
    consensus_matrix.to_csv(args.out, index=True)
    
    print()
    print('---------------------------------------------')
    print('Aggregation complete.')
    print('---------------------------------------------')
    print()


if __name__ == '__main__':
    main()

