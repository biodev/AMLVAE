# Consensus Clustering Implementation

This document describes the consensus clustering workflow added to the AMLCV pipeline.

## Overview

The consensus clustering workflow performs Louvain community detection on k-nearest neighbor (KNN) graphs built from latent space representations. The workflow operates on both VAE and PCA latent embeddings across all cross-validation folds, then aggregates results into consensus co-occurrence matrices.

## Scripts

### 1. `consensus_clustering.py`

Performs graph-based clustering on latent representations using Louvain community detection.

**Key Features:**
- Builds KNN graph from latent space using Euclidean distance
- Runs Louvain community detection with configurable resolution parameter
- Computes silhouette scores for cluster quality assessment
- Seeds are fold-specific for reproducibility

**Usage:**
```bash
python consensus_clustering.py \
    --latent_csv <path_to_latent_space.csv> \
    --out <output_cluster_assignments.csv> \
    --out_metrics <output_cluster_metrics.csv> \
    --k <num_neighbors> \
    --resolution <louvain_resolution> \
    --seed <random_seed>
```

**Outputs:**
- Cluster assignments CSV: Sample IDs with cluster assignments
- Cluster metrics CSV: Number of clusters and silhouette score

### 2. `aggregate_consensus.py`

Aggregates cluster assignments across folds to create consensus co-occurrence matrix.

**Key Features:**
- Loads cluster assignments from all folds
- Creates binary co-occurrence matrices (1 if samples in same cluster, 0 otherwise)
- Sums across folds to create consensus matrix
- Consensus values range from 0 (never co-clustered) to K (always co-clustered)

**Usage:**
```bash
python aggregate_consensus.py \
    --cluster_files file1.csv file2.csv file3.csv ... \
    --out <output_consensus_matrix.csv>
```

**Outputs:**
- `consensus_matrix.csv`: Sample × sample matrix with co-occurrence counts

## Snakemake Workflow

### Rules Added

1. **`cluster_vae`**: Clusters VAE latent space for each fold
2. **`cluster_pca`**: Clusters PCA latent space for each fold
3. **`aggregate_vae_consensus`**: Aggregates VAE clusters into consensus matrix
4. **`aggregate_pca_consensus`**: Aggregates PCA clusters into consensus matrix

### Configuration Parameters

Added to `config.yaml`:

```yaml
clustering:
  k: 10                      # number of neighbors for KNN graph
  resolution: 0.75           # resolution parameter for Louvain algorithm
  seed: 976                  # base random seed (incremented by fold number)
```

### Output Structure

```
folds/<run_id>/
├── fold_0/
│   ├── eval/
│   │   ├── aml_vae_z.csv         # VAE latent space (from eval.py)
│   │   └── aml_pca_z.csv         # PCA latent space (from eval.py)
│   └── clustering/
│       ├── vae_cluster_assignments.csv
│       ├── vae_cluster_metrics.csv
│       ├── pca_cluster_assignments.csv
│       └── pca_cluster_metrics.csv
├── fold_1/
│   └── ...
├── ...
└── consensus/
    ├── vae_consensus_matrix.csv
    └── pca_consensus_matrix.csv
```

## Parameters

### K (Number of Neighbors)
- Controls graph sparsity
- Small k (5-10): Sparser graphs, more local structure
- Large k (50-100): Denser graphs, more global structure
- **Default: 10** (based on R code optimization)

### Resolution
- Controls cluster granularity in Louvain algorithm
- Low resolution (0.1-0.5): Fewer, larger clusters
- High resolution (0.75-1.0): More, smaller clusters
- **Default: 0.75** (based on R code optimization)

### Seed
- Base seed for reproducibility
- Incremented by fold number (seed + fold) for fold-specific variation
- **Default: 976** (matching R code)

## Important Considerations

1. **Distance Metric**: Uses Euclidean distance; may not be appropriate for all latent geometries
2. **Silhouette Scores**: Can be misleading in high dimensions or with irregular cluster shapes
3. **Computational Cost**: KNN construction and silhouette score computation scale O(n²)
4. **Stochasticity**: Louvain algorithm has randomness; results depend on seed
5. **Cluster Number**: Number of clusters not fixed; determined by resolution parameter

## Example Usage

Run the full workflow:
```bash
cd workflows/AMLCV
snakemake --cores 8 --use-conda
```

Run just clustering:
```bash
snakemake --cores 8 --use-conda \
    ../folds/<run_id>/consensus/vae_consensus_matrix.csv \
    ../folds/<run_id>/consensus/pca_consensus_matrix.csv
```

## Interpretation

The consensus co-occurrence matrix can be used to:

1. **Identify stable clusters**: High co-occurrence values indicate robust cluster membership
2. **Downstream clustering**: Use as similarity matrix for consensus clustering algorithms
3. **Visualization**: Heatmap of consensus matrix shows cluster structure stability
4. **Subset selection**: Filter to samples with high consensus (e.g., ≥ 8 out of 10 folds)

## Dependencies

- pandas
- numpy
- networkx
- scikit-learn

All dependencies are included in the existing `amlvae.yaml` conda environment.

