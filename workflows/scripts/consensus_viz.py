"""
Consensus-based Clinical Feature Visualization

This script visualizes clinical features using UMAP embeddings derived from
a consensus co-occurrence matrix. The co-occurrence matrix (from consensus
clustering across folds) is used to construct a k-nearest neighbor graph,
which is then used as a precomputed KNN graph for UMAP.

IMPORTANT CONSIDERATIONS:
- Uses co-occurrence values directly to construct k-nearest neighbor graph
- For each sample, selects k samples with highest co-occurrence values as neighbors
- Higher co-occurrence values indicate more stable co-clustering relationships
- UMAP with precomputed graphs can be sensitive to graph connectivity
- The resulting embedding reflects clustering stability rather than raw feature similarity
- Co-occurrence values are converted to distances (higher co-occurrence = closer)

DRAWBACKS:
- UMAP behavior with precomputed graphs differs from standard distance-based UMAP
- Does not account for within-cluster vs between-cluster structure
- May produce artifacts if co-occurrence distribution is very skewed
- Graph connectivity depends on n_neighbors parameter
"""

import pandas as pd
import argparse
import umap
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from scipy.sparse import csr_matrix


def get_args():
    """Parse command line arguments."""
    
    argparser = argparse.ArgumentParser(
        description='Visualize clinical features using consensus-based UMAP'
    )
    
    argparser.add_argument(
        '--consensus_matrix_path', 
        type=str, 
        required=True,
        help='Path to consensus co-occurrence matrix CSV'
    )
    argparser.add_argument(
        '--clin_path', 
        type=str, 
        required=True,
        help='Path to clinical data file (.csv or .xlsx)'
    )
    argparser.add_argument(
        '--out', 
        type=str, 
        required=True,
        help='Output directory for plots'
    )
    argparser.add_argument(
        '--prefix', 
        type=str, 
        default='',
        help='Prefix for output files (e.g., "vae_" or "pca_")'
    )
    argparser.add_argument(
        '--n_neighbors', 
        type=int, 
        default=15,
        help='Number of nearest neighbors to use for KNN graph construction'
    )
    argparser.add_argument(
        '--min_dist', 
        type=float, 
        default=0.1,
        help='Minimum distance parameter for UMAP'
    )
    argparser.add_argument(
        '--spread', 
        type=float, 
        default=2.0,
        help='UMAP spread parameter - controls overall scale (1.0=default, higher=more spread)'
    )
    argparser.add_argument(
        '--clin_vars', 
        type=str, 
        default='',
        help='Clinical variables to plot (separated by <::>)'
    )
    argparser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    argparser.add_argument(
        '--id_col',
        type=str,
        default='id',
        help='Column name for sample IDs in clinical data'
    )
    
    args = argparser.parse_args()
    args.clin_vars = args.clin_vars.split('<::>')
    
    return args


def load_data(args):
    """
    Load consensus matrix and clinical data.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    consensus_matrix : pd.DataFrame
        Sample x sample co-occurrence matrix
    clin : pd.DataFrame
        Clinical data
    id_col : str
        Column name for sample IDs
    """
    print('Loading consensus co-occurrence matrix...')
    consensus_matrix = pd.read_csv(args.consensus_matrix_path, index_col=0)
    print(f'  Loaded {consensus_matrix.shape[0]} samples')
    print(f'  Co-occurrence range: {consensus_matrix.values.min()} to {consensus_matrix.values.max()}')
    
    print('Loading clinical data...')
    if args.clin_path.endswith('.xlsx'):
        clin = pd.read_excel(args.clin_path, sheet_name=0)
        # Try common ID column names
        if 'MLL ID' in clin.columns:
            id_col = 'MLL ID'
        elif args.id_col in clin.columns:
            id_col = args.id_col
        else:
            raise ValueError(f"Could not find ID column. Available columns: {clin.columns.tolist()}")
    elif args.clin_path.endswith('.csv'):
        clin = pd.read_csv(args.clin_path)
        id_col = args.id_col
        if id_col not in clin.columns:
            raise ValueError(f"ID column '{id_col}' not found in clinical data. Available: {clin.columns.tolist()}")
    else:
        raise ValueError('Unsupported clinical data format. Use .xlsx or .csv.')
    
    print(f'  Loaded clinical data for {len(clin)} samples')
    print(f'  Using ID column: {id_col}')
    
    return consensus_matrix, clin, id_col


def create_distance_matrix_from_cooccurrence(consensus_matrix):
    """
    Create full distance matrix from consensus co-occurrence matrix.
    
    Converts co-occurrence counts to distances for UMAP. Higher co-occurrence
    values indicate samples that consistently cluster together, so they get
    smaller distances.
    
    Uses exponential transformation to increase separation between low and high
    co-occurrence values, which helps UMAP create more spread out embeddings.
    
    Parameters
    ----------
    consensus_matrix : pd.DataFrame
        Sample x sample co-occurrence counts
        
    Returns
    -------
    dist_matrix : np.ndarray
        Full distance matrix, shape (n_samples, n_samples)
    """
    print(f'\nCreating distance matrix from co-occurrence matrix...')
    
    n_samples = len(consensus_matrix)
    cooc = consensus_matrix.values.astype(float)
    
    # Get statistics
    max_cooc = cooc.max()
    min_cooc = cooc[cooc > 0].min() if np.any(cooc > 0) else 0
    
    print(f'  Co-occurrence range: {min_cooc:.0f} to {max_cooc:.0f}')
    print(f'  Matrix shape: {cooc.shape}')
    
    # Convert co-occurrence to distance using exponential transformation
    # This increases separation between different co-occurrence levels
    # Strategy: 
    # 1. Normalize co-occurrence to [0, 1]
    # 2. Apply exponential: distance = exp(scale * (1 - normalized_cooc)) - 1
    # 3. This gives larger distances for low co-occurrence
    
    # Normalize co-occurrence to [0, 1]
    cooc_norm = cooc / (max_cooc + 1e-10)
    
    # Apply exponential transformation with scale factor
    # Higher scale = more separation
    scale = 3.0
    dist_matrix = np.exp(scale * (1.0 - cooc_norm)) - 1.0
    
    # Normalize to reasonable range for UMAP (0 to ~10)
    dist_matrix = dist_matrix / dist_matrix.max() * 10.0
    
    # Set diagonal to exactly zero
    np.fill_diagonal(dist_matrix, 0.0)
    
    # Ensure symmetry
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    
    print(f'  Distance matrix range: {dist_matrix[dist_matrix > 0].min():.4f} to {dist_matrix.max():.4f}')
    print(f'  Mean distance: {dist_matrix[~np.eye(n_samples, dtype=bool)].mean():.4f}')
    print(f'  Using exponential transformation (scale={scale}) for better separation')
    
    return dist_matrix


def run_umap_with_distance_matrix(dist_matrix, n_neighbors, min_dist, spread, seed):
    """
    Run UMAP with precomputed full distance matrix.
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        Full pairwise distance matrix, shape (n_samples, n_samples)
    n_neighbors : int
        Number of neighbors for UMAP graph construction
    min_dist : float
        UMAP min_dist parameter - controls tightness of embedding
        - Lower values (0.0-0.1): Tighter clusters
        - Higher values (0.3-0.99): More spread out, better for visualization
    spread : float
        UMAP spread parameter - controls overall scale of embedding
        - Default: 1.0
        - Higher values (2.0-5.0): More spread out
    seed : int
        Random seed
        
    Returns
    -------
    embedding : np.ndarray
        2D UMAP embedding
    """
    print('\nRunning UMAP with precomputed distance matrix...')
    print(f'  n_neighbors: {n_neighbors}')
    print(f'  min_dist: {min_dist}')
    print(f'  spread: {spread}')
    
    # Create UMAP with precomputed metric
    # Using full distance matrix allows UMAP to properly respect min_dist
    # spread parameter controls overall scale of embedded points (higher = more spread)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=2,
        metric='precomputed',
        random_state=seed,
        verbose=True
    )
    
    # Fit using full distance matrix
    embedding = reducer.fit_transform(dist_matrix)
    
    print(f'  UMAP embedding shape: {embedding.shape}')
    print(f'  Embedding range: x=[{embedding[:, 0].min():.2f}, {embedding[:, 0].max():.2f}], '
          f'y=[{embedding[:, 1].min():.2f}, {embedding[:, 1].max():.2f}]')
    
    return embedding


def create_visualizations(embedding, sample_ids, clin, id_col, clin_vars, out_dir, prefix=''):
    """
    Create scatter plots colored by clinical variables.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D UMAP embedding
    sample_ids : list or pd.Index
        Sample IDs corresponding to rows in embedding
    clin : pd.DataFrame
        Clinical data
    id_col : str
        ID column name in clinical data
    clin_vars : list
        Clinical variables to plot
    out_dir : str
        Output directory
    prefix : str
        Prefix for output files (e.g., 'vae_' or 'pca_')
    """
    print('\nCreating visualizations...')
    
    # Create dataframe with UMAP coordinates
    u = pd.DataFrame(embedding, columns=['u1', 'u2'])
    u[id_col] = sample_ids
    
    # Merge with clinical data
    u = u.merge(clin, on=id_col, how='left')
    print(f'  Merged with clinical data: {len(u)} samples')
    
    # Create plots
    for clin_var in clin_vars:
        if not clin_var or clin_var.strip() == '':
            continue
            
        print(f'  Plotting UMAP for {clin_var}...')
        
        # Check if variable exists
        if clin_var not in u.columns:
            print(f'    Variable {clin_var} not found in data. Skipping...')
            continue
        
        try:
            # Try to convert to numeric if possible
            # This handles columns that should be numeric but are stored as strings
            is_numeric = False
            
            if u[clin_var].dtype == 'object':
                # Try conversion on a copy to see if it works
                converted = pd.to_numeric(u[clin_var], errors='coerce')
                n_original = u[clin_var].notna().sum()
                n_converted = converted.notna().sum()
                
                # Only use numeric conversion if we don't lose too many values
                # (allows for some missing data, but not wholesale conversion to NaN)
                if n_converted > 0 and n_converted >= 0.5 * n_original:
                    u[clin_var] = converted
                    is_numeric = True
                    print(f'    Converted {clin_var} to numeric ({n_converted}/{n_original} values)')
                else:
                    is_numeric = False
                    print(f'    Keeping {clin_var} as categorical')
            else:
                # Already numeric
                is_numeric = pd.api.types.is_numeric_dtype(u[clin_var])
            
            plt.figure(figsize=(8, 8))
            
            if is_numeric:
                # For numeric variables, use continuous color scale
                # Remove NaN values for plotting
                plot_data = u.dropna(subset=[clin_var])
                scatter = plt.scatter(
                    plot_data['u1'], 
                    plot_data['u2'], 
                    c=plot_data[clin_var],
                    cmap='viridis',
                    alpha=0.7,
                    s=50
                )
                plt.colorbar(scatter, label=clin_var)
            else:
                # For categorical variables, use discrete colors
                sbn.scatterplot(data=u, x='u1', y='u2', hue=clin_var, alpha=0.7, s=50)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            plt.title(f'Consensus-based UMAP ({prefix.rstrip("_")}): {clin_var}')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/{prefix}consensus_umap_{clin_var}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'    Error plotting {clin_var}: {e}. Skipping...')
            plt.close()


def main():
    """Main execution function."""
    
    print()
    print('---------------------------------------------')
    print('Consensus-based Clinical Feature Visualization')
    print('---------------------------------------------')
    print()
    
    args = get_args()
    print('Arguments:')
    for arg, value in vars(args).items():
        print(f'  {arg}: {value}')
    print('---------------------------------------------')
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data
    consensus_matrix, clin, id_col = load_data(args)
    
    # Create full distance matrix from co-occurrence matrix
    dist_matrix = create_distance_matrix_from_cooccurrence(consensus_matrix)
    
    # Run UMAP with full distance matrix
    embedding = run_umap_with_distance_matrix(dist_matrix, args.n_neighbors, args.min_dist, args.spread, args.seed)
    
    # Create visualizations
    create_visualizations(
        embedding, 
        consensus_matrix.index, 
        clin, 
        id_col, 
        args.clin_vars, 
        args.out,
        args.prefix
    )
    
    # Mark complete
    completion_file = f'{args.out}/{args.prefix}consensus_viz_complete.txt'
    with open(completion_file, 'w') as f:
        f.write('complete')
    print(f'\nCompletion marker written to: {completion_file}')
    
    print()
    print('---------------------------------------------')
    print('Visualization complete.')
    print('---------------------------------------------')
    print()


if __name__ == '__main__':
    main()

