# AMLVAE: Usage Guide

This document provides practical instructions for using AMLVAE, from installation to running experiments and analyzing results.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Workflow](#detailed-workflow)
4. [Command-Line Scripts](#command-line-scripts)
5. [Python API Usage](#python-api-usage)
6. [Configuration](#configuration)
7. [Common Tasks](#common-tasks)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, not required)
- 8+ GB RAM
- 10+ GB disk space

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd AMLVAE
```

### Step 2: Create Conda Environment

```bash
# Create environment from YAML
conda env create -f workflows/envs/amlvae.yaml

# Activate environment
conda activate amlvae
```

**Alternative: Manual Installation**

```bash
conda create -n amlvae python=3.9
conda activate amlvae

# Install PyTorch (visit pytorch.org for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn
pip install umap-learn ray[tune] pyyaml
```

### Step 3: Install AMLVAE Package

```bash
# Install in editable mode
pip install -e .
```

**Verify Installation**:

```python
import amlvae
from amlvae.models import VAE
from amlvae.data import ExprProcessor
print("AMLVAE installed successfully!")
```

---

## Quick Start

### 5-Minute Example

```bash
# 1. Navigate to AML workflow
cd workflows/AML

# 2. Preprocess data (assumes data in ../../data/)
python ../scripts/proc.py \
    --data ../../data/ \
    --out ../proc/ \
    --config config.yaml

# 3. Create train/val/test partitions
python ../scripts/partition.py \
    --proc ../proc/ \
    --dataset aml

# 4. Train model with default hyperparameters
python ../scripts/train.py \
    --data ../../data/ \
    --proc ../proc/ \
    --out ../output/ \
    --dataset_name aml \
    --n_latent 32 \
    --n_hidden 512 \
    --epochs 1000

# 5. Evaluate model
python ../scripts/eval.py \
    --proc ../proc/ \
    --out ../output/ \
    --model_path ../output/model.pt \
    --dataset aml

# 6. Visualize latent space with clinical data
python ../scripts/clin_viz.py \
    --config config.yaml
```

### Expected Output

```
workflows/
├── proc/
│   ├── aml_expr.csv          # Preprocessed expression (samples × genes)
│   ├── aml_partitions.pt     # Train/val/test split
│   └── aml_gene_list.txt     # Selected gene names
├── output/
│   ├── model.pt              # Trained VAE model
│   ├── aml_z.csv             # Latent representations
│   ├── eval.csv              # Reconstruction metrics
│   └── umap_visualization.png # UMAP plot
```

---

## Detailed Workflow

### Step 1: Data Preparation

**Input**: Raw gene expression CSV in long format

```
| id       | gene_name | fpkm_uq_unstranded | unstranded |
|----------|-----------|-------------------|------------|
| 00-00001 | GAPDH     | 2543.21           | 156420     |
| 00-00001 | TP53      | 89.45             | 5421       |
| ...      | ...       | ...               | ...        |
```

**Script**: `workflows/scripts/proc.py`

```bash
python workflows/scripts/proc.py \
    --data data/aml_full_manuscript.csv \
    --out workflows/proc/ \
    --target fpkm_uq_unstranded \
    --counts_name unstranded \
    --gene_selection wgcna \
    --top_n 2000 \
    --norm zscore \
    --dataset_name aml
```

**Parameters**:
- `--data`: Path to raw expression CSV
- `--out`: Output directory for processed data
- `--target`: Expression value column (fpkm_uq_unstranded, fpkm_unstranded, tpm)
- `--counts_name`: Raw count column (for WGCNA method)
- `--gene_selection`: Method (variance, tcga, wgcna)
- `--top_n`: Number of genes to select
- `--norm`: Normalization (zscore, minmax)
- `--dataset_name`: Prefix for output files

**Output**:
- `{dataset}_expr.csv`: Normalized expression matrix (samples × genes)
- `{dataset}_gene_list.txt`: Selected gene names
- `{dataset}_transform_params.pkl`: Normalization parameters

### Step 2: Data Partitioning

**Script**: `workflows/scripts/partition.py`

```bash
python workflows/scripts/partition.py \
    --proc workflows/proc/ \
    --dataset aml \
    --train_frac 0.6 \
    --val_frac 0.2 \
    --test_frac 0.2 \
    --seed 42
```

**Parameters**:
- `--proc`: Directory with processed data
- `--dataset`: Dataset name
- `--train_frac`: Training set fraction (default: 0.6)
- `--val_frac`: Validation set fraction (default: 0.2)
- `--test_frac`: Test set fraction (default: 0.2)
- `--seed`: Random seed for reproducibility

**Output**:
- `{dataset}_partitions.pt`: Dictionary with train/val/test sample IDs

### Step 3A: Train with Fixed Hyperparameters

**Script**: `workflows/scripts/train.py`

```bash
python workflows/scripts/train.py \
    --proc workflows/proc/ \
    --out workflows/output/ \
    --dataset_name aml \
    --epochs 1500 \
    --patience 100 \
    --n_hidden 512 \
    --n_latent 32 \
    --n_layers 2 \
    --norm layer \
    --variational true \
    --anneal true \
    --dropout 0.0 \
    --nonlin elu \
    --lr 1e-4 \
    --l2 0.0 \
    --beta 1.0 \
    --batch_size 256
```

**Key Parameters**:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--n_latent` | Latent dimension | 8, 12, 16, 24, 32, 64 |
| `--n_hidden` | Hidden layer width | 128, 256, 512, 1024 |
| `--n_layers` | Number of layers | 1, 2, 3, 4 |
| `--lr` | Learning rate | 1e-5 to 1e-3 |
| `--beta` | KL weight | 0.1 to 5.0 |
| `--batch_size` | Batch size | 64, 128, 256, 512 |
| `--dropout` | Dropout rate | 0.0, 0.1, 0.2 |
| `--anneal` | Use β-annealing | true/false |
| `--patience` | Early stop patience | 100-1500 epochs |

**Output**:
- `model.pt`: Trained PyTorch model

**Monitoring Training**:

```bash
# With verbose output
python workflows/scripts/train.py ... --verbose

# Output:
# epoch: 0, val mse: 1.2345, val r2: 0.45, kld: 8.23, beta: 0.01
# epoch: 1, val mse: 1.1234, val r2: 0.52, kld: 7.89, beta: 0.02
# ...
```

### Step 3B: Hyperparameter Tuning

**Script**: `workflows/scripts/tune.py`

```bash
python workflows/scripts/tune.py \
    --proc workflows/proc/ \
    --out workflows/tuning/ \
    --dataset_name aml \
    --target_metric r2 \
    --num_samples 100 \
    --epochs 1500 \
    --patience 150 \
    --gpus 4 \
    --cpus 20
```

**Parameters**:
- `--target_metric`: Metric to optimize (r2, mse, elbo)
- `--num_samples`: Number of configurations to try
- `--gpus`: Number of GPUs (trials run in parallel)
- `--cpus`: Number of CPU cores per trial
- `--epochs`: Max epochs per trial
- `--patience`: Early stopping patience

**Search Space** (defined in `tune.py`):

```python
config = {
    "n_hidden": tune.choice([128, 256, 512, 1024]),
    "n_latent": tune.choice([8, 12, 16, 24, 32, 64]),
    "n_layers": tune.choice([1, 2, 3, 4]),
    "lr": tune.loguniform(1e-5, 1e-3),
    "beta": tune.choice([0.1, 0.5, 1.0, 2.0, 5.0]),
    "dropout": tune.choice([0.0, 0.1, 0.2, 0.3]),
    "l2": tune.choice([0.0, 1e-5, 1e-4, 1e-3]),
    "batch_size": tune.choice([64, 128, 256, 512]),
    # ... other hyperparameters
}
```

**Output**:
- `best_config.json`: Best hyperparameter configuration
- `results.csv`: All trial results
- Ray Tune logs in `~/ray_results/`

**Using Best Config**:

```bash
# Extract best hyperparameters
python -c "
import json
with open('workflows/tuning/best_config.json') as f:
    config = json.load(f)
print(' '.join([f'--{k} {v}' for k, v in config.items()]))
"

# Retrain with best config
python workflows/scripts/train.py \
    --proc workflows/proc/ \
    --out workflows/output/ \
    --dataset_name aml \
    <paste best config here>
```

### Step 4: Evaluation

**Script**: `workflows/scripts/eval.py`

```bash
python workflows/scripts/eval.py \
    --proc workflows/proc/ \
    --out workflows/output/ \
    --model_path workflows/output/model.pt \
    --dataset aml
```

**Output**:

`eval.csv`:
```
| model | r2_train | r2_val | r2_test | mse_train | mse_val | mse_test | path |
|-------|----------|--------|---------|-----------|---------|----------|------|
| vae   | 0.85     | 0.78   | 0.76    | 0.123     | 0.145   | 0.152    | ... |
| pca   | 0.72     | 0.69   | 0.68    | 0.234     | 0.256   | 0.261    | ... |
```

`aml_z.csv`:
```
| sample_id | z1    | z2    | ... | z32   |
|-----------|-------|-------|-----|-------|
| 00-00001  | 0.45  | -1.23 | ... | -0.34 |
| ...       | ...   | ...   | ... | ...   |
```

### Step 5: Visualization & Clinical Analysis

**Script**: `workflows/scripts/clin_viz.py`

```bash
python workflows/scripts/clin_viz.py \
    --z_path workflows/output/aml_z.csv \
    --clin_path data/beataml_clinical_for_inputs.csv \
    --out workflows/output/ \
    --clin_vars overallSurvival consensusAMLFusions fabBlastMorphology \
    --n_neighbors 15 \
    --min_dist 0.1 \
    --metric euclidean
```

**Parameters**:
- `--z_path`: Path to latent representations
- `--clin_path`: Path to clinical data
- `--clin_vars`: Clinical variables to visualize (space-separated)
- `--n_neighbors`, `--min_dist`, `--metric`: UMAP parameters

**Output**:
- `umap_survival.png`: UMAP colored by survival time
- `umap_fusions.png`: UMAP colored by gene fusions
- `umap_fab.png`: UMAP colored by FAB classification

**Statistical Analysis** (`workflows/scripts/clin_eval.py`):

```bash
python workflows/scripts/clin_eval.py \
    --z_path workflows/output/aml_z.csv \
    --clin_path data/beataml_clinical_for_inputs.csv \
    --out workflows/output/ \
    --test spearman \
    --adjust_pval fdr_bh
```

**Output**:

`clinical_associations.csv`:
```
| latent_dim | clinical_var      | rho    | pval   | pval_adj |
|------------|-------------------|--------|--------|----------|
| z1         | overallSurvival   | 0.32   | 0.001  | 0.015    |
| z2         | ageAtDiagnosis    | -0.28  | 0.003  | 0.024    |
| ...        | ...               | ...    | ...    | ...      |
```

---

## Command-Line Scripts

### Complete Script Reference

| Script | Purpose | Key Inputs | Key Outputs |
|--------|---------|------------|-------------|
| `proc.py` | Preprocess expression data | Raw CSV | Normalized matrix |
| `partition.py` | Create train/val/test splits | Expression matrix | Partition file |
| `train.py` | Train VAE with fixed config | Partitioned data | Trained model |
| `tune.py` | Hyperparameter search | Partitioned data | Best config |
| `eval.py` | Evaluate reconstruction | Model + data | Metrics + latents |
| `clin_viz.py` | Visualize with clinical data | Latents + clinical | UMAP plots |
| `clin_eval.py` | Statistical associations | Latents + clinical | Correlation table |
| `distances.py` | Compute sample distances | Latents | Distance matrix |

---

## Python API Usage

### Example: Complete Pipeline in Python

```python
import pandas as pd
import torch
from amlvae.data import ExprProcessor
from amlvae.models import VAE
from amlvae.train import Trainer

# 1. Preprocess data
expr_long = pd.read_csv('data/aml_full_manuscript.csv')
processor = ExprProcessor(
    expr_long,
    target='fpkm_uq_unstranded',
    counts_name='unstranded'
)
processor.select_genes_(method='wgcna', top_n=2000)
processor.normalize_(method='zscore')
X, sample_ids = processor.get_data()

# Save processed data
expr_df = pd.DataFrame(X, index=sample_ids, columns=processor.selected_genes)
expr_df.to_csv('processed/aml_expr.csv')

# 2. Create partitions
from sklearn.model_selection import train_test_split

train_ids, temp_ids = train_test_split(sample_ids, test_size=0.4, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

partitions = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}
torch.save(partitions, 'processed/aml_partitions.pt')

# 3. Train model
trainer = Trainer(
    root='processed/',
    dataset_name='aml',
    epochs=1500,
    patience=100,
    verbose=True,
    return_best_model=True
)

config = {
    'n_hidden': 512,
    'n_layers': 2,
    'n_latent': 32,
    'norm': 'layer',
    'variational': True,
    'anneal': True,
    'dropout': 0.0,
    'nonlin': 'elu',
    'lr': 1e-4,
    'l2': 0.0,
    'beta': 1.0,
    'batch_size': 256,
    'masked_prob': 0.0
}

model = trainer(config)
torch.save(model, 'output/model.pt')

# 4. Encode samples
model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
    z, _ = model.encode(X_tensor)
    z = z.cpu().numpy()

# Save latent representations
z_df = pd.DataFrame(z, index=sample_ids, columns=[f'z{i+1}' for i in range(32)])
z_df.to_csv('output/aml_z.csv')

# 5. Evaluate reconstruction
from sklearn.metrics import r2_score
import torch.nn.functional as F

with torch.no_grad():
    xhat = model.predict(X_tensor).cpu()
    mse = F.mse_loss(X_tensor.cpu(), xhat).item()
    r2 = r2_score(X, xhat.numpy(), multioutput='variance_weighted')

print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")
```

### Example: Load Pretrained Model

```python
import torch
import pandas as pd

# Load model
model = torch.load('output/model.pt', map_location='cuda')
model.eval()

# Load new expression data
new_expr = pd.read_csv('new_samples.csv', index_col=0)

# Encode (assumes new_expr is already preprocessed with same genes)
with torch.no_grad():
    X = torch.tensor(new_expr.values, dtype=torch.float32).cuda()
    z, _ = model.encode(X)
    z = z.cpu().numpy()

# Reconstruct
with torch.no_grad():
    xhat = model.predict(X).cpu().numpy()

print("Latent shape:", z.shape)
print("Reconstruction shape:", xhat.shape)
```

### Example: Apply Preprocessing to New Data

```python
# Load original processor (with fitted transform parameters)
import pickle

with open('processed/aml_transform_params.pkl', 'rb') as f:
    transform_params = pickle.load(f)

# Load gene list
with open('processed/aml_gene_list.txt') as f:
    gene_list = [line.strip() for line in f]

# Process new data
new_expr_long = pd.read_csv('new_data.csv')
processor = ExprProcessor(new_expr_long)
processor.transform_params = transform_params
processor.selected_genes = gene_list

# Apply same transformation
X_new, sample_ids_new = processor.process_new(new_expr_long)
```

---

## Configuration

### YAML Configuration Files

**Location**: `workflows/{DATASET}/config.yaml`

**Example** (`workflows/AML/config.yaml`):

```yaml
# Unique identifier for this experiment
run_id: "aml_wgcna_zscore_latent32"

# Global settings
data_dir: "../../../data"
n_latent: 32

# Script paths (relative to config file)
scripts:
  proc: "../scripts/proc.py"
  train: "../scripts/train.py"
  tune: "../scripts/tune.py"
  eval: "../scripts/eval.py"

# Preprocessing parameters
proc:
  target_type: "fpkm_unstranded"
  gene_selection_method: "wgcna"
  num_top_genes: 2000
  norm_method: "zscore"
  dataset_name: "aml"
  gene_id_type: "gene_name"
  sample_id_type: "id"

# Tuning parameters
tune:
  target_metric: "r2"
  num_samples: 100
  patience: 1500
  gpus: 1
  cpus: 10
  epochs: 1500

# Training parameters
train:
  epochs: 1500
  patience: 1500

# Visualization parameters
clin_viz:
  z_path: "{EVAL_OUT}/{dataset_name}_z.csv"
  clin_path: "../../../data/beataml_clinical_for_inputs.csv"
  clin_vars: ["overallSurvival", "consensusAMLFusions", "fabBlastMorphology"]
  n_neighbors: 15
  min_dist: 0.1
  metric: "euclidean"
  seed: 42
```

**Loading Config in Scripts**:

```python
import yaml

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# Access values
n_genes = cfg['proc']['num_top_genes']
epochs = cfg['train']['epochs']
```

---

## Common Tasks

### Task 1: Compare Different Gene Selection Methods

```bash
# Variance-based
python proc.py --gene_selection variance --top_n 2000 --out proc_variance/
python train.py --proc proc_variance/ --out output_variance/

# TCGA
python proc.py --gene_selection tcga --top_n 2000 --out proc_tcga/
python train.py --proc proc_tcga/ --out output_tcga/

# WGCNA
python proc.py --gene_selection wgcna --top_n 2000 --out proc_wgcna/
python train.py --proc proc_wgcna/ --out output_wgcna/

# Compare
python eval.py --model_path output_variance/model.pt --out eval_variance/
python eval.py --model_path output_tcga/model.pt --out eval_tcga/
python eval.py --model_path output_wgcna/model.pt --out eval_wgcna/

# Aggregate results
cat eval_*/eval.csv > all_eval.csv
```

### Task 2: Test Different Latent Dimensions

```bash
for latent in 8 12 16 24 32 64; do
    python train.py \
        --proc workflows/proc/ \
        --out workflows/output_L${latent}/ \
        --n_latent ${latent} \
        --dataset_name aml
        
    python eval.py \
        --proc workflows/proc/ \
        --model_path workflows/output_L${latent}/model.pt \
        --out workflows/output_L${latent}/ \
        --dataset aml
done

# Compare
for latent in 8 12 16 24 32 64; do
    echo "Latent dim: ${latent}"
    cat workflows/output_L${latent}/eval.csv | grep vae
done
```

### Task 3: Transfer Learning (AML → MDS)

```bash
# 1. Train on AML
python train.py --proc workflows/proc/ --dataset_name aml --out output_aml/

# 2. Load AML model and fine-tune on MDS
python train.py \
    --proc workflows/proc/ \
    --dataset_name mds \
    --out output_mds_transfer/ \
    --pretrained_model output_aml/model.pt \
    --epochs 500 \
    --lr 1e-5  # Lower learning rate for fine-tuning

# Note: Requires implementing --pretrained_model flag in train.py
```

### Task 4: Extract Gene Importance

```python
import torch
import numpy as np

# Load model
model = torch.load('output/model.pt')

# Get decoder weights (first layer)
W_decoder = model.decoder.mlp[0].weight.data.cpu().numpy()
# Shape: (hidden_dim, latent_dim)

# For each latent dimension, find most important genes
gene_names = [...] # Load from gene_list.txt

for i in range(model.latent_dim):
    # Get weights for this latent dimension
    weights = W_decoder[:, i]
    
    # Find top genes
    top_indices = np.argsort(np.abs(weights))[-10:]
    top_genes = [gene_names[j] for j in top_indices]
    
    print(f"Latent dim {i+1}: {', '.join(top_genes)}")
```

---

## Troubleshooting

### Issue: Out of Memory (OOM) Error

**Symptoms**: CUDA out of memory during training

**Solutions**:
1. Reduce batch size: `--batch_size 64` (or lower)
2. Reduce hidden dimension: `--n_hidden 256`
3. Reduce number of layers: `--n_layers 1`
4. Clear CUDA cache: Add to training script:
   ```python
   torch.cuda.empty_cache()
   ```

### Issue: Model Not Learning (High Loss)

**Symptoms**: Validation loss doesn't decrease, poor reconstruction

**Diagnostics**:
```python
# Check data scale
print(X_train.mean(), X_train.std())  # Should be ~0 and ~1 if z-scored

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.norm())  # Should not be 0 or very large
```

**Solutions**:
1. Check data preprocessing (normalization applied correctly?)
2. Increase learning rate: `--lr 1e-3`
3. Reduce β: `--beta 0.1`
4. Increase epochs/patience
5. Try different initialization (run with different seeds)

### Issue: Posterior Collapse

**Symptoms**: KL divergence near 0, model ignores latent space

**Diagnostics**:
```python
# Check latent variance
z_var = z.var(axis=0)
print(z_var)  # Should not all be ~1.0 (indicating prior, not posterior)
```

**Solutions**:
1. Use β-annealing: `--anneal true`
2. Reduce β: `--beta 0.5` or `--beta 0.1`
3. Add masking: `--masked_prob 0.3`
4. Increase latent dimension: `--n_latent 64`

### Issue: Slow Training

**Solutions**:
1. Ensure GPU is being used:
   ```python
   print(torch.cuda.is_available())  # Should be True
   print(next(model.parameters()).device)  # Should be 'cuda:0'
   ```
2. Increase batch size: `--batch_size 512`
3. Reduce patience: `--patience 100`
4. Use fewer tuning trials: `--num_samples 20`

### Issue: Poor Generalization (Train ≫ Test)

**Symptoms**: High train R² but low test R²

**Solutions**:
1. Increase regularization:
   - Higher β: `--beta 2.0`
   - Add dropout: `--dropout 0.2`
   - Add L2: `--l2 1e-4`
2. More data augmentation: `--masked_prob 0.3`
3. Reduce model capacity: `--n_hidden 256`, `--n_layers 1`
4. Ensure test set is representative (check for batch effects)

---

## Best Practices

1. **Always Use a Held-Out Test Set**: Never touch test data during development
2. **Set Random Seeds**: For reproducibility (`--seed 42`)
3. **Monitor Both Train and Validation**: Watch for overfitting
4. **Start Simple**: Begin with small models, increase complexity if needed
5. **Tune Systematically**: Architecture → Regularization → Optimization
6. **Validate Biologically**: Check if latent space captures known biology
7. **Document Experiments**: Keep track of configurations and results

---

## Further Resources

- See [METHODOLOGY.md](METHODOLOGY.md) for theoretical background
- See [API.md](API.md) for detailed API reference
- See [examples/notebooks/](../notebooks/) for interactive tutorials
- Visit project repository for latest updates

---

**Last Updated**: October 2025



