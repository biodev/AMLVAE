# AMLVAE: Data Documentation

This document describes the data sources, formats, preprocessing steps, and data management for the AMLVAE project.

## Table of Contents

1. [Data Sources](#data-sources)
2. [Data Formats](#data-formats)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Data Partitioning](#data-partitioning)
5. [Clinical Data](#clinical-data)
6. [Data Quality Considerations](#data-quality-considerations)

---

## Data Sources

### BeatAML Dataset

**Description**: The BeatAML (Beat AML Master Trial) project is a precision medicine initiative for Acute Myeloid Leukemia. It provides comprehensive molecular profiling and drug sensitivity data for hundreds of AML patients.

**Data Types**:
- **Gene Expression**: RNA-seq quantified as FPKM (Fragments Per Kilobase Million) and raw read counts
- **Clinical Metadata**: Patient demographics, survival outcomes, AML subtypes (FAB classification)
- **Genetic Features**: Gene fusions, mutations, cytogenetics
- **Drug Response**: Ex vivo sensitivity to hundreds of compounds (AUC, IC50)

**Files in Repository**:
- `data/aml_full_manuscript.csv` - Complete gene expression data
- `data/aml_train.csv` - Training set gene expression
- `data/aml_test.csv` - Test set gene expression
- `data/beataml_clinical_for_inputs.csv` - Clinical variables for visualization/analysis
- `data/beataml_probit_curve_fits_v4_distr.txt` - Drug response curves
- `data/beataml_wv_1to4_vg_cts.xlsx` - Wave 1-4 read counts

**Access**: BeatAML data is publicly available through:
- Viz platform: https://vizome.org/aml/
- Publication: Tyner et al. (2018), Nature

### Myelodysplastic Syndrome (MDS) Dataset

**Description**: Gene expression data from MDS patients, a related hematological disorder that can progress to AML.

**Workflow**:
- Configured in `workflows/MDS/config.yaml`
- Separate preprocessing and training pipelines
- Can be used for transfer learning or comparative analysis

**Purpose**: 
- Study relationship between MDS and AML
- Test generalization of models trained on AML
- Identify shared and distinct molecular features

---

## Data Formats

### Gene Expression Files

#### Long Format (Input)

Gene expression data typically starts in "long" or "tidy" format:

```
| sample_id | gene_name | fpkm_uq_unstranded | unstranded | gene_id |
|-----------|-----------|-------------------|------------|---------|
| 00-00001  | GAPDH     | 2543.21           | 156420     | ENSG... |
| 00-00001  | TP53      | 89.45             | 5421       | ENSG... |
| 00-00002  | GAPDH     | 2198.76           | 142315     | ENSG... |
| ...       | ...       | ...               | ...        | ...     |
```

**Columns**:
- `sample_id` / `id`: Unique patient/sample identifier
- `gene_name`: Human-readable gene symbol (e.g., GAPDH, TP53)
- `gene_id`: Ensembl gene ID (e.g., ENSG00000111640)
- `fpkm_uq_unstranded`: Upper-quartile normalized FPKM (main expression metric)
- `fpkm_unstranded`: Standard FPKM
- `unstranded`: Raw read counts

**Characteristics**:
- Each row = one gene in one sample
- Total rows = N_samples × N_genes (~20,000-60,000 genes per sample)
- Multiple expression quantifications available

#### Wide Format (Processed)

After preprocessing, data is pivoted to "wide" format (samples × genes matrix):

```
| sample_id | GAPDH  | TP53   | MYC    | ... | (1000+ genes) |
|-----------|--------|--------|--------|-----|---------------|
| 00-00001  | 2.31   | -0.45  | 1.23   | ... | ...           |
| 00-00002  | 1.98   | 0.12   | -0.67  | ... | ...           |
| ...       | ...    | ...    | ...    | ... | ...           |
```

**Characteristics**:
- Each row = one sample
- Each column = one gene
- Values are normalized (z-scores or [0,1] range)
- Only selected genes included (typically 1,000-2,500)

**Saved As**: `{dataset}_expr.csv` in processed data directory

**Format**:
```python
df = pd.read_csv('aml_expr.csv')
df = df.set_index(df.columns[0])  # First column is sample IDs
# Shape: (N_samples, N_genes)
```

### Partition Files

Data splits stored as PyTorch pickle files:

```python
partitions = torch.load('aml_partitions.pt')
# partitions = {
#     'train_ids': ['00-00001', '00-00003', ...],  # ~60-70% of samples
#     'val_ids': ['00-00005', '00-00008', ...],     # ~15-20% of samples
#     'test_ids': ['00-00012', '00-00015', ...]     # ~15-20% of samples
# }
```

**Purpose**:
- Ensure consistent train/val/test splits across experiments
- Enable reproducible model evaluation
- Prevent data leakage

**Generation**: Created by `partition.py` script

### Model Files

Trained models saved as PyTorch objects:

```python
model = torch.load('model.pt')
# model: VAE instance with learned parameters
```

**Contents**:
- Model architecture (encoder, decoder, mask_classifier)
- Learned parameters (weights, biases)
- Hyperparameters (latent_dim, hidden_dim, etc.)

**Usage**:
```python
# Encode new samples
z, _ = model.encode(X_new.to('cuda'))

# Reconstruct
x_hat = model.predict(X_new.to('cuda'))
```

### Latent Representations

Encoded samples saved as CSV:

```
| sample_id | z1     | z2     | z3     | ... | z32    |
|-----------|--------|--------|--------|-----|--------|
| 00-00001  | 0.45   | -1.23  | 0.78   | ... | -0.34  |
| 00-00002  | -0.67  | 0.89   | -0.12  | ... | 1.45   |
| ...       | ...    | ...    | ...    | ... | ...    |
```

**Generated By**: `eval.py` script  
**Saved As**: `{dataset}_z.csv` in output directory

**Purpose**:
- Downstream analysis (clustering, classification)
- Visualization (UMAP, t-SNE)
- Clinical association studies

---

## Preprocessing Pipeline

The preprocessing pipeline transforms raw gene expression data into model-ready tensors.

### Step 1: Load Data

```python
from amlvae.data.ExprProcessor import ExprProcessor

# Load long-format expression data
expr_long = pd.read_csv('data/aml_full_manuscript.csv')

# Initialize processor
processor = ExprProcessor(
    expr_long,
    target='fpkm_uq_unstranded',  # Expression metric to use
    counts_name='unstranded',      # Raw counts (for WGCNA)
    gene_col='gene_name',          # Gene identifier column
    sample_id_col='id'             # Sample identifier column
)
```

### Step 2: Select Genes

```python
# Choose method: 'variance', 'tcga', or 'wgcna'
processor.select_genes_(method='wgcna', top_n=2000)

# Selected genes stored in processor.selected_genes
print(f"Selected {len(processor.selected_genes)} genes")
```

**Method Comparison**:

| Method | Filters | Ranking Metric | Best For |
|--------|---------|----------------|----------|
| Variance | None | Raw variance | Quick exploration |
| TCGA | Noise & expression threshold | Coefficient of variation | Published protocols |
| WGCNA | CPM, MAD | Log-space variance | RNA-seq best practices |

### Step 3: Normalize

```python
# Choose method: 'zscore' or 'minmax'
processor.normalize_(method='zscore')

# Normalized data stored in processor.expr
# Transform parameters stored in processor.transform_params
```

**Transform Parameters** (for applying to new data):
```python
# Z-score: {'mu': gene_means, 'sd': gene_stds, 'method': 'zscore'}
# Min-max: {'min': gene_mins, 'max': gene_maxs, 'method': 'minmax'}
```

### Step 4: Extract Data

```python
# Get processed data as numpy arrays
X, sample_ids = processor.get_data()
# X: shape (N_samples, N_genes), normalized expression
# sample_ids: list of sample IDs (same order as rows in X)
```

### Step 5: Create DataFrame and Save

```python
# Create wide-format DataFrame
expr_df = pd.DataFrame(
    X,
    index=sample_ids,
    columns=processor.selected_genes
)

# Save for model training
expr_df.to_csv('processed/aml_expr.csv')
```

### Complete Example

See `workflows/scripts/proc.py` for a full preprocessing script that:
1. Loads raw expression data
2. Applies gene selection and normalization
3. Saves processed expression matrix
4. Saves gene list and transform parameters

**Usage**:
```bash
python workflows/scripts/proc.py \
    --data ../data/aml_full_manuscript.csv \
    --out ../processed/ \
    --gene_selection wgcna \
    --top_n 2000 \
    --norm zscore \
    --dataset_name aml
```

---

## Data Partitioning

### Creating Partitions

```python
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Load processed expression data
expr = pd.read_csv('processed/aml_expr.csv', index_col=0)
sample_ids = expr.index.tolist()

# Split into train/val/test (60%/20%/20%)
train_ids, temp_ids = train_test_split(
    sample_ids, test_size=0.4, random_state=42
)
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.5, random_state=42
)

# Save partitions
partitions = {
    'train_ids': train_ids,
    'val_ids': val_ids,
    'test_ids': test_ids
}
torch.save(partitions, 'processed/aml_partitions.pt')
```

**Script**: `workflows/scripts/partition.py`

### Loading Partitions

```python
partitions = torch.load('processed/aml_partitions.pt', weights_only=False)

# Load expression data
expr = pd.read_csv('processed/aml_expr.csv', index_col=0)

# Extract partitions
X_train = torch.tensor(expr.loc[partitions['train_ids']].values, dtype=torch.float32)
X_val = torch.tensor(expr.loc[partitions['val_ids']].values, dtype=torch.float32)
X_test = torch.tensor(expr.loc[partitions['test_ids']].values, dtype=torch.float32)
```

### Stratified Partitioning (Optional)

For imbalanced datasets or important clinical subgroups:

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Load clinical labels (e.g., AML subtype)
labels = clinical_df.loc[sample_ids, 'aml_subtype'].values

# Stratified split (ensures proportional representation)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(sss.split(sample_ids, labels))

# Continue splitting val/test...
```

---

## Clinical Data

### Clinical Variables File

**File**: `data/beataml_clinical_for_inputs.csv`

**Key Variables**:

| Variable | Type | Description |
|----------|------|-------------|
| `overallSurvival` | Continuous | Survival time in days |
| `vitalStatus` | Binary | 0 = alive, 1 = deceased |
| `consensusAMLFusions` | Categorical | AML-defining gene fusions (e.g., RUNX1-RUNX1T1) |
| `fabBlastMorphology` | Categorical | FAB classification (M0-M7) |
| `ageAtDiagnosis` | Continuous | Age in years |
| `sex` | Binary | M = male, F = female |
| `priorMalignancy` | Binary | Previous cancer diagnosis |
| `specimenType` | Categorical | Bone marrow, peripheral blood, etc. |

### Linking Clinical and Expression Data

```python
# Load clinical data
clinical = pd.read_csv('data/beataml_clinical_for_inputs.csv', index_col=0)

# Load latent representations
latent = pd.read_csv('output/aml_z.csv', index_col=0)

# Merge on sample ID
combined = latent.join(clinical, how='inner')

# Now you can analyze relationships
# e.g., correlation between latent dims and survival
from scipy.stats import spearmanr

for col in latent.columns:
    rho, pval = spearmanr(combined[col], combined['overallSurvival'])
    print(f"{col}: rho={rho:.3f}, p={pval:.3e}")
```

### Visualization with Clinical Data

```python
import umap
import matplotlib.pyplot as plt

# Compute UMAP embedding
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(latent.values)

# Plot colored by clinical variable
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=combined['overallSurvival'],
    cmap='viridis',
    s=10
)
plt.colorbar(label='Overall Survival (days)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Latent Space Colored by Survival')
plt.show()
```

**Script**: `workflows/scripts/clin_viz.py` automates this process

---

## Data Quality Considerations

### Gene Expression Quality Control

**Before Preprocessing**:

1. **Library Size**: Check total read counts per sample
   - Very low counts (< 1M) may indicate failed sequencing
   - Remove outlier samples

2. **Gene Detection**: Number of genes with non-zero counts
   - Typical: 15,000-20,000 detected genes
   - Very low detection suggests low-quality sample

3. **Batch Effects**: Check for systematic differences by:
   - Sequencing date
   - Sequencing platform
   - Processing batch
   - Use PCA to visualize potential batch structure

**Code Example**:
```python
# Library sizes
lib_sizes = expr_long.groupby('sample_id')['unstranded'].sum()
plt.hist(lib_sizes, bins=50)
plt.xlabel('Total Read Count')
plt.ylabel('Number of Samples')
plt.title('Library Size Distribution')

# Remove low-quality samples
good_samples = lib_sizes[lib_sizes > 1e6].index
expr_long = expr_long[expr_long['sample_id'].isin(good_samples)]
```

### Missing Data

**Gene Expression**: 
- FPKM = 0 typically means "not detected" rather than "missing"
- Log transform log₂(x + 1) handles zeros gracefully
- If truly missing (NaN), either:
  - Impute with gene-specific median
  - Exclude samples with too many missing values

**Clinical Data**:
- Missing clinical variables common (especially historic data)
- Options:
  - Exclude samples with missing outcome of interest
  - Impute with median/mode
  - Use multiple imputation for sensitivity analysis

### Data Leakage Prevention

**Critical Rules**:

1. **No Test Set Usage**: Test data must never influence:
   - Gene selection (compute variance on train only)
   - Normalization parameters (fit on train only)
   - Hyperparameter tuning (use validation set)

2. **Proper Validation**: 
   - Use validation set for early stopping, hyperparameter tuning
   - Test set only for final evaluation (once)

3. **Temporal Splits** (if applicable):
   - If data collected over time, split chronologically
   - Train on earlier samples, test on later samples
   - Simulates real-world deployment

### Recommended Workflow

1. **Exploratory Analysis**: Use full dataset for QC and visualization
2. **Partition**: Create train/val/test splits
3. **Lock Test Set**: Move to separate directory, don't touch until final evaluation
4. **Preprocessing**: Fit all transformations on train set only
5. **Model Development**: Use train + validation
6. **Final Evaluation**: Apply to test set once

---

## Data Privacy and Ethics

### Patient Privacy

- BeatAML data is de-identified (no names, dates of birth, etc.)
- Follow institutional IRB/ethics guidelines
- Don't attempt to re-identify patients
- Aggregate reporting only (no individual patient results)

### Data Sharing

- Processed data (normalized expression matrices) can be shared with collaborators
- Include data use agreements if required by data source
- Don't share models trained on restricted-access data without permission

### Responsible Use

- Gene expression patterns may correlate with sensitive attributes (race, ancestry)
- Be aware of potential biases in training data
- Validate findings across diverse cohorts when possible

---

## References

- Tyner et al. (2018). Functional genomic landscape of acute myeloid leukaemia. Nature 562: 526-531.
- GTEx Consortium (2020). The GTEx Consortium atlas of genetic regulatory effects across human tissues. Science 369: 1318-1330.
- Robinson et al. (2010). edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. Bioinformatics 26: 139-140.

---

**Last Updated**: October 2025



