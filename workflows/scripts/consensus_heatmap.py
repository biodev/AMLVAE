"""
Consensus Co-occurrence Heatmap with Clinical Annotations

Creates a clustered heatmap of the consensus co-occurrence matrix with
hierarchical clustering and clinical annotations displayed as color bars.

IMPORTANT CONSIDERATIONS:
- Hierarchical clustering is applied to both rows and columns of the consensus matrix
- Clustering uses the co-occurrence as a similarity metric (converted to distance)
- Clinical annotations are displayed as colored bars (row colors)
- Handles both categorical and continuous clinical variables
- Missing values are shown in gray
- Large matrices may be slow to render and require careful figure sizing

DRAWBACKS:
- May be difficult to read with very large sample sizes (>500 samples)
- Categorical variables with many levels may have hard-to-distinguish colors
- Hierarchical clustering can be sensitive to linkage method choice
- Memory intensive for large matrices
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')


def get_args():
    """Parse command line arguments."""
    
    argparser = argparse.ArgumentParser(
        description='Create consensus heatmap with clinical annotations'
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
        help='Output path for heatmap PNG'
    )
    argparser.add_argument(
        '--prefix', 
        type=str, 
        default='',
        help='Prefix for output files (e.g., "vae_" or "pca_")'
    )
    argparser.add_argument(
        '--clin_vars', 
        type=str, 
        default='',
        help='Clinical variables for annotations (separated by <::>)'
    )
    argparser.add_argument(
        '--id_col',
        type=str,
        default='id',
        help='Column name for sample IDs in clinical data'
    )
    argparser.add_argument(
        '--linkage_method',
        type=str,
        default='average',
        help='Linkage method for hierarchical clustering (average, complete, single, ward)'
    )
    argparser.add_argument(
        '--figsize',
        type=str,
        default='20,18',
        help='Figure size as width,height (e.g., "20,18")'
    )
    
    args = argparser.parse_args()
    args.clin_vars = [v.strip() for v in args.clin_vars.split('<::>') if v.strip()]
    args.figsize = tuple(map(int, args.figsize.split(',')))

    if args.id_col == 'MLL_ID':
        args.id_col = 'MLL ID'
        
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
    print(f'  Loaded {len(consensus_matrix)} samples')
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
    
    clin = clin.set_index(id_col)
    
    print(f'  Loaded clinical data for {len(clin)} samples')
    print(f'  Using ID column: {id_col}')
    
    # Filter to common samples
    common_ids = consensus_matrix.index.intersection(clin.index)
    consensus_matrix = consensus_matrix.loc[common_ids, common_ids]
    clin = clin.loc[common_ids]
    
    print(f'  {len(common_ids)} samples with both consensus and clinical data')
    
    return consensus_matrix, clin


def prepare_annotations(clin, clin_vars):
    """
    Prepare clinical annotations for heatmap display.
    
    Parameters
    ----------
    clin : pd.DataFrame
        Clinical data
    clin_vars : list
        List of clinical variables to include
        
    Returns
    -------
    row_colors : pd.DataFrame
        DataFrame of colors for each annotation
    color_maps : dict
        Mapping of variable names to color legends
    """
    print('\nPreparing clinical annotations...')
    
    row_colors_dict = {}
    color_maps = {}
    
    for var in clin_vars:
        if var not in clin.columns:
            print(f'  Warning: {var} not found in clinical data, skipping')
            continue
        
        print(f'  Processing {var}...')
        
        # Try to convert to numeric
        if clin[var].dtype == 'object':
            converted = pd.to_numeric(clin[var], errors='coerce')
            n_original = clin[var].notna().sum()
            n_converted = converted.notna().sum()
            
            if n_converted > 0 and n_converted >= 0.5 * n_original:
                # Numeric variable
                var_data = converted
                is_numeric = True
            else:
                # Categorical variable
                var_data = clin[var]
                is_numeric = False
        else:
            var_data = clin[var]
            is_numeric = pd.api.types.is_numeric_dtype(var_data)
        
        if is_numeric:
            # For numeric: use continuous colormap
            # Normalize to [0, 1] for color mapping
            var_clean = var_data.dropna()
            if len(var_clean) == 0:
                print(f'    All values missing for {var}, skipping')
                continue
            
            vmin, vmax = var_clean.min(), var_clean.max()
            if vmin == vmax:
                # All same value
                colors = ['#808080'] * len(var_data)  # Gray
            else:
                # Normalize and map to colors
                norm_values = (var_data - vmin) / (vmax - vmin)
                cmap = plt.cm.viridis
                colors = []
                for val in norm_values:
                    if pd.isna(val):
                        colors.append('#D3D3D3')  # Light gray for missing
                    else:
                        colors.append(mcolors.rgb2hex(cmap(val)))
            
            row_colors_dict[var] = colors
            color_maps[var] = {'type': 'continuous', 'vmin': vmin, 'vmax': vmax, 'cmap': 'viridis'}
            print(f'    Continuous variable: range [{vmin:.2f}, {vmax:.2f}]')
        
        else:
            # For categorical: assign discrete colors
            categories = var_data.dropna().unique()
            n_cats = len(categories)
            
            if n_cats == 0:
                print(f'    All values missing for {var}, skipping')
                continue
            
            # Use a good categorical palette
            if n_cats <= 10:
                palette = sns.color_palette('tab10', n_cats)
            elif n_cats <= 20:
                palette = sns.color_palette('tab20', n_cats)
            else:
                palette = sns.color_palette('husl', n_cats)
            
            cat_colors = dict(zip(categories, [mcolors.rgb2hex(c) for c in palette]))
            
            colors = []
            for val in var_data:
                if pd.isna(val):
                    colors.append('#D3D3D3')  # Light gray for missing
                else:
                    colors.append(cat_colors.get(val, '#D3D3D3'))
            
            row_colors_dict[var] = colors
            color_maps[var] = {'type': 'categorical', 'mapping': cat_colors}
            print(f'    Categorical variable: {n_cats} categories')
    
    if len(row_colors_dict) == 0:
        print('  Warning: No valid annotations found')
        return None, {}
    
    row_colors = pd.DataFrame(row_colors_dict, index=clin.index)
    
    print(f'  Prepared {len(row_colors.columns)} annotations')
    
    return row_colors, color_maps


def create_clustered_heatmap(consensus_matrix, row_colors, color_maps, args):
    """
    Create clustered heatmap of consensus matrix with annotations.
    
    Parameters
    ----------
    consensus_matrix : pd.DataFrame
        Sample x sample co-occurrence matrix
    row_colors : pd.DataFrame
        DataFrame of colors for annotations
    color_maps : dict
        Color mapping information for legends
    args : argparse.Namespace
        Command line arguments
    """
    print('\nCreating clustered heatmap...')
    
    # Convert co-occurrence to distance for clustering
    # distance = max_cooc - cooc (higher co-occurrence = smaller distance)
    max_cooc = consensus_matrix.values.max()
    dist_matrix = max_cooc - consensus_matrix.values
    np.fill_diagonal(dist_matrix, 0)
    
    # Ensure distance matrix is symmetric
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    
    # Convert to condensed distance matrix for linkage
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # Perform hierarchical clustering
    print(f'  Performing hierarchical clustering (method={args.linkage_method})...')
    linkage_matrix = linkage(condensed_dist, method=args.linkage_method)
    
    # Create clustermap
    print(f'  Creating figure (size={args.figsize})...')
    
    # Set up the plot with proper spacing for annotations
    if row_colors is not None:
        n_annotations = len(row_colors.columns)
        # Colorbar in lower left
        cbar_pos = (0.02, 0.1, 0.03, 0.15)  # x, y, width, height
    else:
        n_annotations = 0
        cbar_pos = (0.02, 0.1, 0.03, 0.15)
    
    g = sns.clustermap(
        consensus_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        row_colors=None,  # Don't show row colors (left side)
        col_colors=row_colors,  # Only show column colors (top)
        cmap='YlOrRd',  # Good for co-occurrence (white=low, red=high)
        figsize=args.figsize,
        cbar_pos=cbar_pos,
        cbar_kws={'label': 'Co-occurrence count'},
        xticklabels=False,  # Don't show individual sample labels
        yticklabels=False,
        linewidths=0,
        rasterized=True,  # Faster rendering for large matrices
        dendrogram_ratio=(0.001, 0.001),  # Minimal dendrograms (row, col)
        colors_ratio=0.03  # Adjust annotation bar height
    )
    
    # Hide dendrogram axes by removing them from display
    if hasattr(g, 'ax_row_dendrogram'):
        g.ax_row_dendrogram.set_visible(False)
    if hasattr(g, 'ax_col_dendrogram'):
        g.ax_col_dendrogram.set_visible(False)
    
    # Add title
    title = f'Consensus Co-occurrence Matrix'
    if args.prefix:
        title += f' ({args.prefix.rstrip("_").upper()})'
    g.fig.suptitle(title, fontsize=16, y=0.98)
    
    # Create custom legends for annotations
    if row_colors is not None and len(color_maps) > 0:
        print('  Creating annotation legends...')
        
        # Position for legends (right side of plot)
        legend_x = 0.92
        legend_y_start = 0.95
        legend_y_step = 0.15  # Space between legends
        
        for idx, (var_name, color_info) in enumerate(color_maps.items()):
            legend_y = legend_y_start - (idx * legend_y_step)
            
            if color_info['type'] == 'categorical':
                # Create categorical legend
                mapping = color_info['mapping']
                handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=8, label=cat)
                          for cat, color in mapping.items()]
                
                # Add missing data indicator
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='#D3D3D3', markersize=8, 
                                         label='Missing'))
                
                legend = g.fig.legend(handles=handles, title=var_name,
                                     bbox_to_anchor=(legend_x, legend_y),
                                     loc='upper left', frameon=True, fontsize=8)
                legend.get_title().set_fontsize(9)
                legend.get_title().set_fontweight('bold')
            
            else:
                # Create continuous colorbar legend
                # This is more complex, we'll add a small colorbar
                from matplotlib.colorbar import ColorbarBase
                from matplotlib.colors import Normalize
                
                ax_cbar = g.fig.add_axes([legend_x, legend_y - 0.1, 0.02, 0.08])
                norm = Normalize(vmin=color_info['vmin'], vmax=color_info['vmax'])
                cbar = ColorbarBase(ax_cbar, cmap=color_info['cmap'], 
                                   norm=norm, orientation='vertical')
                cbar.set_label(var_name, fontsize=9, fontweight='bold')
                cbar.ax.tick_params(labelsize=7)
    
    # Save figure
    output_path = f"{args.out}/{args.prefix}consensus_heatmap.png"
    print(f'  Saving to {output_path}...')
    g.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'  Heatmap saved successfully')


def main():
    """Main execution function."""
    
    print()
    print('---------------------------------------------')
    print('Consensus Co-occurrence Heatmap')
    print('---------------------------------------------')
    print()
    
    args = get_args()
    print('Arguments:')
    for arg, value in vars(args).items():
        if arg != 'clin_vars':  # Skip long list
            print(f'  {arg}: {value}')
    print(f'  clin_vars: {len(args.clin_vars)} variables')
    print('---------------------------------------------')
    
    # Load data
    consensus_matrix, clin = load_data(args)
    
    # Prepare annotations
    row_colors, color_maps = prepare_annotations(clin, args.clin_vars)
    
    # Create heatmap
    create_clustered_heatmap(consensus_matrix, row_colors, color_maps, args)
    
    # Mark complete
    completion_file = f'{args.out}/{args.prefix}consensus_heatmap_complete.txt'
    with open(completion_file, 'w') as f:
        f.write('complete')
    print(f'\nCompletion marker written to: {completion_file}')
    
    print()
    print('---------------------------------------------')
    print('Heatmap creation complete.')
    print('---------------------------------------------')
    print()


if __name__ == '__main__':
    main()

