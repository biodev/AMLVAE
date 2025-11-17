# Consensus Clustering Workflow - Final Implementation

## Workflow Overview

The consensus clustering has been integrated into the AMLCV workflow with proper file naming conventions for VAE and PCA outputs.

## Data Flow

```
evaluate (eval.py)
    ↓ produces
    ├── aml_vae_z.csv  (VAE latent embeddings)
    └── aml_pca_z.csv  (PCA latent embeddings)
    ↓
cluster_vae / cluster_pca (consensus_clustering.py)
    ↓ produces (per fold)
    ├── vae_cluster_assignments.csv
    ├── vae_cluster_metrics.csv
    ├── pca_cluster_assignments.csv
    └── pca_cluster_metrics.csv
    ↓
aggregate_vae_consensus / aggregate_pca_consensus (aggregate_consensus.py)
    ↓ produces (across all folds)
    ├── vae_consensus_matrix.csv  (sample × sample co-occurrence)
    └── pca_consensus_matrix.csv  (sample × sample co-occurrence)
    ↓
visualize_vae_consensus / visualize_pca_consensus (consensus_viz.py)
    ↓ produces
    ├── consensus_umap_<clinical_var>.png  (UMAP plots colored by clinical features)
    └── consensus_viz_complete.txt
```

## File Naming Convention

### Per-fold Outputs

| Method | Cluster Assignments | Cluster Metrics |
|--------|---------------------|-----------------|
| VAE    | `vae_cluster_assignments.csv` | `vae_cluster_metrics.csv` |
| PCA    | `pca_cluster_assignments.csv` | `pca_cluster_metrics.csv` |

### Consensus Outputs

| Method | Consensus Matrix |
|--------|------------------|
| VAE    | `vae_consensus_matrix.csv` |
| PCA    | `pca_consensus_matrix.csv` |

## Configuration Parameters

In `config.yaml`:

```yaml
scripts:
  consensus_clustering: "../scripts/consensus_clustering.py"
  aggregate_consensus: "../scripts/aggregate_consensus.py"
  consensus_viz: "../scripts/consensus_viz.py"

clustering:
  k: 10                      # KNN neighbors
  resolution: 0.75           # Louvain resolution
  seed: 976                  # Base seed (+ fold number)

consensus_viz:
  clin_path: "<path_to_clinical_data>"
  n_neighbors: 15            # number of neighbors for KNN graph
  min_dist: 0.1              # UMAP min_dist parameter
  id_col: "id"               # sample ID column name
  clin_vars: [...]           # list of clinical variables to visualize
  seed: 0
```

## Snakemake Rules

1. **`cluster_vae`**: Clusters VAE latent space per fold
   - Input: `aml_vae_z.csv` from eval rule
   - Output: `vae_cluster_assignments.csv`, `vae_cluster_metrics.csv`

2. **`cluster_pca`**: Clusters PCA latent space per fold
   - Input: `aml_pca_z.csv` from eval rule
   - Output: `pca_cluster_assignments.csv`, `pca_cluster_metrics.csv`

3. **`aggregate_vae_consensus`**: Creates VAE consensus matrix
   - Input: All fold `vae_cluster_assignments.csv` files (passed as list)
   - Output: `vae_consensus_matrix.csv`

4. **`aggregate_pca_consensus`**: Creates PCA consensus matrix
   - Input: All fold `pca_cluster_assignments.csv` files (passed as list)
   - Output: `pca_consensus_matrix.csv`

5. **`visualize_vae_consensus`**: Visualizes VAE consensus with clinical features
   - Input: `vae_consensus_matrix.csv`
   - Output: UMAP plots colored by clinical variables

6. **`visualize_pca_consensus`**: Visualizes PCA consensus with clinical features
   - Input: `pca_consensus_matrix.csv`
   - Output: UMAP plots colored by clinical variables

## Running the Workflow

### Full pipeline
```bash
cd workflows/AMLCV
snakemake --cores 8 --use-conda
```

### Only clustering steps
```bash
snakemake --cores 8 --use-conda \
    ../folds/<run_id>/consensus/vae_consensus_matrix.csv \
    ../folds/<run_id>/consensus/pca_consensus_matrix.csv
```

### Dry run to check
```bash
snakemake -n
```

## Output Interpretation

### Cluster Assignments CSV
```csv
sample_id,cluster
001454b2-aff9-4659-85a6-73fb8092589a,0
00231f4e-4e13-4f89-9c34-2f6e3c8f9a1b,1
...
```

### Cluster Metrics CSV
```csv
n_clusters,avg_silhouette,k,resolution
3,0.245,10,0.75
```

### Consensus Matrix CSV
```csv
,001454b2-aff9-4659-85a6-73fb8092589a,00231f4e-4e13-4f89-9c34-2f6e3c8f9a1b,...
001454b2-aff9-4659-85a6-73fb8092589a,10,8,...
00231f4e-4e13-4f89-9c34-2f6e3c8f9a1b,8,10,...
...
```

Values range from 0-10 (for 10-fold CV), indicating how many folds the sample pair was co-clustered.

## Key Implementation Details

1. **No hardcoded filenames**: All paths constructed from config
2. **Specific output names**: Script takes `--out` and `--out_metrics` as full paths
3. **Fold-specific seeds**: `seed = base_seed + fold_number` for reproducibility
4. **No survival analysis**: Simplified compared to R version
5. **No grid search**: Uses fixed parameters from config
6. **Proper dependency chain**: eval → cluster → aggregate → visualize

## Consensus Visualization Approach

The consensus visualization uses a novel approach to visualize clinical features based on clustering stability:

### Method

1. **Create KNN from Co-occurrence**: Build k-nearest neighbor graph directly from co-occurrence values
   - For each sample, select k samples with highest co-occurrence values
   - Convert co-occurrence to distance: distance = max_cooc - cooc + small_epsilon
   - No thresholding needed - preserves relative strength of relationships

2. **UMAP Embedding**: Run UMAP with precomputed KNN graph
   - UMAP uses `metric='precomputed'` mode
   - Embedding reflects clustering stability rather than raw feature distance
   - Samples that consistently co-cluster appear close in the UMAP space

3. **Visualize Clinical Features**: Color UMAP by clinical variables
   - Same visualization approach as standard `clin_viz.py`
   - Interpretation: Clinical patterns should align with stable cluster structure

### Advantages

- Focuses on robust, reproducible relationships (stable across folds)
- Reduces noise from single-fold clustering artifacts
- Integrates information from entire cross-validation
- Network-based embedding preserves local neighborhood structure

### Parameter Selection

**n_neighbors** controls the graph connectivity:
- **Small (5-10)**: Focus on strongest co-clustering relationships
- **Medium (15-20)**: Balanced view (recommended default)
- **Large (25-50)**: Include weaker relationships for more global structure

The method automatically selects samples with highest co-occurrence values, so no manual thresholding is needed.

