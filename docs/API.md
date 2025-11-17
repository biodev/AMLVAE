# AMLVAE: API Reference

This document provides detailed API documentation for all public classes, methods, and functions in the AMLVAE package.

## Table of Contents

1. [Data Processing](#data-processing)
2. [Models](#models)
3. [Training](#training)
4. [Utilities](#utilities)

---

## Data Processing

### `amlvae.data.ExprProcessor`

**Class for processing gene expression data from long to wide format with gene selection and normalization.**

#### `__init__(expr_long, target, counts_name, gene_col, sample_id_col)`

Initialize the expression processor.

**Parameters:**
- `expr_long` (pd.DataFrame): Long-format expression DataFrame with columns for sample ID, gene ID, expression values, and counts
- `target` (str, default='fpkm_uq_unstranded'): Column name for expression values (e.g., 'fpkm_uq_unstranded', 'tpm')
- `counts_name` (str, default='unstranded'): Column name for raw read counts (required for WGCNA method)
- `gene_col` (str, default='gene_name'): Column name for gene identifiers
- `sample_id_col` (str, default='id'): Column name for sample identifiers

**Attributes:**
- `raw_expr` (pd.DataFrame): Pivoted expression matrix (samples × genes), shape (N_samples, N_genes)
- `sample_ids` (list): List of sample IDs
- `selected_genes` (list): List of selected gene names (after calling `select_genes_()`)
- `expr` (pd.DataFrame): Normalized expression matrix (after calling `normalize_()`)
- `transform_params` (dict): Normalization parameters for applying to new data

**Example:**

```python
from amlvae.data import ExprProcessor
import pandas as pd

# Load long-format data
expr_long = pd.read_csv('data/aml_full_manuscript.csv')

# Initialize processor
processor = ExprProcessor(
    expr_long,
    target='fpkm_uq_unstranded',
    counts_name='unstranded',
    gene_col='gene_name',
    sample_id_col='id'
)

print(f"Raw expression shape: {processor.raw_expr.shape}")
```

#### `select_genes_(method, top_n)`

Select top N genes based on specified method.

**Parameters:**
- `method` (str): Gene selection method
  - `'variance'`: Select genes by raw variance
  - `'tcga'`: TCGA protocol (noise filtering + CV ranking)
  - `'wgcna'`: WGCNA/edgeR protocol (CPM filtering + MAD + variance)
- `top_n` (int, default=1000): Number of genes to select

**Side Effects:**
- Sets `self.selected_genes` to list of selected gene names

**Example:**

```python
# Select genes using WGCNA protocol
processor.select_genes_(method='wgcna', top_n=2000)

print(f"Selected {len(processor.selected_genes)} genes")
print(f"Top 10 genes: {processor.selected_genes[:10]}")
```

**Raises:**
- `ValueError`: If method is not one of {'variance', 'tcga', 'wgcna'}
- `RuntimeError`: If no genes pass filtering criteria (for WGCNA/TCGA methods)

#### `normalize_(method)`

Normalize expression values.

**Parameters:**
- `method` (str): Normalization method
  - `'zscore'`: Log₂ transform + z-score normalization per gene
  - `'minmax'`: Log₂ transform + min-max scaling per gene to [0,1]

**Side Effects:**
- Sets `self.expr` to normalized expression DataFrame
- Sets `self.transform_params` to dict with normalization parameters

**Example:**

```python
# Z-score normalization
processor.normalize_(method='zscore')

print(f"Normalized expression shape: {processor.expr.shape}")
print(f"Mean: {processor.expr.mean().mean():.4f}")  # Should be ~0
print(f"Std: {processor.expr.std().mean():.4f}")    # Should be ~1
```

**Raises:**
- `ValueError`: If method is not one of {'zscore', 'minmax'}

#### `get_data()`

Extract processed data as numpy array.

**Returns:**
- `X` (np.ndarray): Normalized expression matrix, shape (N_samples, N_genes)
- `sample_ids` (list): List of sample IDs corresponding to rows in X

**Example:**

```python
X, sample_ids = processor.get_data()

print(f"X shape: {X.shape}")
print(f"X dtype: {X.dtype}")
print(f"Number of samples: {len(sample_ids)}")
```

**Raises:**
- `ValueError`: If `normalize_()` has not been called
- `ValueError`: If `select_genes_()` has not been called

#### `process_new(expr_long)`

Apply fitted transformations to new expression data.

**Parameters:**
- `expr_long` (pd.DataFrame): New long-format expression data (same format as training data)

**Returns:**
- `X` (np.ndarray): Normalized expression matrix, shape (N_new_samples, N_genes)
- `sample_ids` (list): List of sample IDs

**Example:**

```python
# Process new data with same transformations
new_expr_long = pd.read_csv('data/new_samples.csv')
X_new, sample_ids_new = processor.process_new(new_expr_long)

print(f"New data shape: {X_new.shape}")
```

**Notes:**
- Uses same gene selection and normalization parameters as fitted on training data
- New data must contain the same genes (in `gene_col`) as training data
- For min-max normalization, values are clipped to [0,1] after transformation

---

### Module-Level Functions

#### `select_genes_wgcna_protocol(expr, counts, top_n, min_count, min_total_count, min_prop)`

Select genes using WGCNA/edgeR protocol.

**Parameters:**
- `expr` (pd.DataFrame): Expression matrix (samples × genes)
- `counts` (pd.DataFrame): Raw count matrix (samples × genes)
- `top_n` (int, default=1000): Number of genes to return
- `min_count` (int, default=10): Minimum count threshold for CPM filter
- `min_total_count` (int, default=15): Minimum total count across all samples
- `min_prop` (float, default=0.66): Proportion of samples that must pass CPM threshold

**Returns:**
- `genes` (list): Gene names ordered by decreasing variance

**Example:**

```python
from amlvae.data.ExprProcessor import select_genes_wgcna_protocol

genes = select_genes_wgcna_protocol(
    expr=expr_df,
    counts=counts_df,
    top_n=2000,
    min_count=10,
    min_total_count=15,
    min_prop=0.66
)
```

#### `select_genes_tcga(expr, noise_threshold, median_threshold, top_n)`

Select genes using TCGA protocol.

**Parameters:**
- `expr` (pd.DataFrame): Expression matrix (samples × genes)
- `noise_threshold` (float, default=0.2): Genes with expression ≤ this value in ≥75% of samples are removed
- `median_threshold` (float, default=10.0): Minimum median expression
- `top_n` (int, default=1000): Number of genes to return

**Returns:**
- `genes` (list): Gene names ordered by decreasing coefficient of variation

#### `select_genes_by_variance(expr, top_n)`

Select genes by raw variance.

**Parameters:**
- `expr` (pd.DataFrame): Expression matrix (samples × genes)
- `top_n` (int, default=1000): Number of genes to return

**Returns:**
- `genes` (list): Gene names ordered by decreasing variance

#### `normalize_zscore(expr)`

Z-score normalize expression data.

**Parameters:**
- `expr` (pd.DataFrame): Expression matrix (samples × genes)

**Returns:**
- `normed` (pd.DataFrame): Normalized expression
- `params` (dict): {'mu': gene_means, 'sd': gene_stds, 'method': 'zscore'}

#### `normalize_minmax(expr)`

Min-max normalize expression data to [0,1].

**Parameters:**
- `expr` (pd.DataFrame): Expression matrix (samples × genes)

**Returns:**
- `normed` (pd.DataFrame): Normalized expression
- `params` (dict): {'min': gene_mins, 'max': gene_maxs, 'method': 'minmax'}

---

## Models

### `amlvae.models.VAE`

**Variational Autoencoder for gene expression data.**

#### `__init__(input_dim, hidden_dim, n_layers, latent_dim, conditions, norm, nonlin, variational, dropout)`

Initialize VAE model.

**Parameters:**
- `input_dim` (int): Number of input features (genes)
- `hidden_dim` (int): Width of hidden layers
- `n_layers` (int): Number of hidden layers in encoder/decoder
- `latent_dim` (int): Dimension of latent space
- `conditions` (dict, default={}): Dictionary of {name: dimension} for adversarial tasks (deprecated)
- `norm` (str, default='layer'): Normalization type ('layer', 'batch', or None)
- `nonlin` (str, default='elu'): Activation function ('elu', 'relu', 'leakyrelu')
- `variational` (bool, default=True): If True, sample from latent distribution; if False, use mean only
- `dropout` (float, default=0.0): Dropout probability

**Attributes:**
- `encoder` (MLP): Encoder network
- `decoder` (MLP): Decoder network
- `mask_classifier` (MLP): Mask prediction network (for VIME)
- `latent_dim` (int): Latent space dimension
- `variational` (bool): Whether to use stochastic sampling

**Example:**

```python
from amlvae.models import VAE

model = VAE(
    input_dim=2000,      # 2000 genes
    hidden_dim=512,      # 512 hidden units
    n_layers=2,          # 2 hidden layers
    latent_dim=32,       # 32 latent dimensions
    norm='layer',
    nonlin='elu',
    variational=True,
    dropout=0.0
)

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
```

#### `forward(x)`

Forward pass through VAE.

**Parameters:**
- `x` (torch.Tensor): Input expression, shape (batch_size, input_dim)

**Returns:**
- `output` (dict): Dictionary containing:
  - `'xhat'` (torch.Tensor): Reconstructed expression, shape (batch_size, input_dim)
  - `'mu'` (torch.Tensor): Latent mean, shape (batch_size, latent_dim)
  - `'logvar'` (torch.Tensor): Latent log-variance, shape (batch_size, latent_dim)
  - `'mask_hat'` (torch.Tensor): Predicted mask, shape (batch_size, input_dim)

**Example:**

```python
import torch

x = torch.randn(128, 2000)  # Batch of 128 samples
output = model(x)

print(f"Reconstruction shape: {output['xhat'].shape}")
print(f"Latent mean shape: {output['mu'].shape}")
```

#### `encode(x)`

Encode input to latent distribution parameters.

**Parameters:**
- `x` (torch.Tensor): Input expression, shape (batch_size, input_dim)

**Returns:**
- `mu` (torch.Tensor): Latent mean, shape (batch_size, latent_dim)
- `logvar` (torch.Tensor): Latent log-variance, shape (batch_size, latent_dim)

**Example:**

```python
mu, logvar = model.encode(x)

# Compute standard deviation
std = torch.exp(0.5 * logvar)
print(f"Latent mean: {mu[0][:5]}")
print(f"Latent std: {std[0][:5]}")
```

#### `reparameterize(mu, logvar)`

Sample from latent distribution using reparameterization trick.

**Parameters:**
- `mu` (torch.Tensor): Latent mean, shape (batch_size, latent_dim)
- `logvar` (torch.Tensor): Latent log-variance, shape (batch_size, latent_dim)

**Returns:**
- `z` (torch.Tensor): Sampled latent vector, shape (batch_size, latent_dim)

**Example:**

```python
z = model.reparameterize(mu, logvar)
print(f"Sampled latent shape: {z.shape}")
```

#### `decode(z)`

Decode latent vector to reconstructed expression.

**Parameters:**
- `z` (torch.Tensor): Latent vector, shape (batch_size, latent_dim)

**Returns:**
- `xhat` (torch.Tensor): Reconstructed expression, shape (batch_size, input_dim)

**Example:**

```python
xhat = model.decode(z)
print(f"Reconstruction shape: {xhat.shape}")
```

#### `predict(x)`

Deterministic reconstruction (uses mean encoding, no sampling).

**Parameters:**
- `x` (torch.Tensor): Input expression, shape (batch_size, input_dim)

**Returns:**
- `xhat` (torch.Tensor): Reconstructed expression, shape (batch_size, input_dim)

**Example:**

```python
# For evaluation, use predict() for deterministic results
model.eval()
with torch.no_grad():
    xhat = model.predict(x)
```

#### `loss(x, xhat, mu, logvar, beta, mask, mask_hat)` (staticmethod)

Compute VAE loss.

**Parameters:**
- `x` (torch.Tensor): Input expression, shape (batch_size, input_dim)
- `xhat` (torch.Tensor): Reconstructed expression, shape (batch_size, input_dim)
- `mu` (torch.Tensor): Latent mean, shape (batch_size, latent_dim)
- `logvar` (torch.Tensor): Latent log-variance, shape (batch_size, latent_dim)
- `beta` (float, default=1.0): Weight for KL divergence term
- `mask` (torch.Tensor, optional): True mask for VIME, shape (batch_size, input_dim)
- `mask_hat` (torch.Tensor, optional): Predicted mask, shape (batch_size, input_dim)

**Returns:**
- `total_loss` (torch.Tensor): Total loss (scalar)
- `recon_loss` (torch.Tensor): Reconstruction loss (MSE)
- `kld` (torch.Tensor): KL divergence
- `mask_loss` (torch.Tensor or float): Mask prediction loss (0.0 if mask is None)

**Example:**

```python
output = model(x)
total_loss, recon_loss, kld, mask_loss = VAE.loss(
    x,
    beta=1.0,
    **output
)

print(f"Total: {total_loss.item():.4f}")
print(f"Reconstruction: {recon_loss.item():.4f}")
print(f"KL: {kld.item():.4f}")
```

---

### `amlvae.models.MLP`

**Multi-layer perceptron building block.**

#### `__init__(in_channels, hidden_channels, out_channels, layers, dropout, nonlin, out, norm, bias)`

Initialize MLP.

**Parameters:**
- `in_channels` (int): Input dimension
- `hidden_channels` (int): Hidden layer width
- `out_channels` (int): Output dimension
- `layers` (int, default=2): Number of hidden layers
- `dropout` (float, default=0.0): Dropout probability
- `nonlin` (torch.nn.Module, default=torch.nn.ELU): Activation function class
- `out` (torch.nn.Module, optional): Output activation (e.g., torch.nn.Sigmoid)
- `norm` (torch.nn.Module, default=torch.nn.LayerNorm): Normalization class
- `bias` (bool, default=True): Use bias in linear layers

**Example:**

```python
from amlvae.models import MLP
import torch.nn as nn

mlp = MLP(
    in_channels=2000,
    hidden_channels=512,
    out_channels=64,
    layers=3,
    dropout=0.1,
    nonlin=nn.ELU,
    norm=nn.LayerNorm
)

x = torch.randn(128, 2000)
output = mlp(x)
print(f"Output shape: {output.shape}")  # (128, 64)
```

#### `forward(x)`

Forward pass through MLP.

**Parameters:**
- `x` (torch.Tensor): Input, shape (batch_size, in_channels)

**Returns:**
- `output` (torch.Tensor): Output, shape (batch_size, out_channels)

---

## Training

### `amlvae.train.Trainer`

**Training orchestrator with data loading, training loops, and evaluation.**

#### `__init__(root, dataset_name, checkpoint, log_every, epochs, verbose, patience, return_best_model)`

Initialize trainer.

**Parameters:**
- `root` (str): Path to directory with processed data (contains `{dataset}_expr.csv` and `{dataset}_partitions.pt`)
- `dataset_name` (str, default='aml'): Dataset name prefix
- `checkpoint` (bool, default=False): Enable Ray Tune checkpointing
- `log_every` (int, default=250): Checkpoint frequency (epochs)
- `epochs` (int, default=500): Maximum training epochs
- `verbose` (bool, default=False): Print training progress
- `patience` (int, default=100): Early stopping patience (epochs without improvement)
- `return_best_model` (bool, default=False): Return trained model (True) or None (False)

**Attributes:**
- `X_train` (torch.Tensor): Training data, shape (N_train, N_genes)
- `X_val` (torch.Tensor): Validation data, shape (N_val, N_genes)
- `X_test` (torch.Tensor): Test data, shape (N_test, N_genes)

**Example:**

```python
from amlvae.train import Trainer

trainer = Trainer(
    root='workflows/proc/',
    dataset_name='aml',
    checkpoint=False,
    epochs=1500,
    verbose=True,
    patience=100,
    return_best_model=True
)

print(f"Training samples: {len(trainer.X_train)}")
print(f"Validation samples: {len(trainer.X_val)}")
print(f"Test samples: {len(trainer.X_test)}")
```

#### `__call__(config)`

Train VAE with given configuration.

**Parameters:**
- `config` (dict): Configuration dictionary with keys:
  - `'n_hidden'` (int): Hidden layer width
  - `'n_layers'` (int): Number of layers
  - `'n_latent'` (int): Latent dimension
  - `'norm'` (str): Normalization type
  - `'variational'` (bool): Use variational sampling
  - `'anneal'` (bool): Use β-annealing
  - `'dropout'` (float): Dropout rate
  - `'nonlin'` (str): Activation function
  - `'lr'` (float): Learning rate
  - `'l2'` (float): L2 regularization
  - `'beta'` (float): KL weight (target if annealing)
  - `'batch_size'` (int): Batch size
  - `'masked_prob'` (float): Masking probability (VIME)

**Returns:**
- `model` (VAE or None): Trained model if `return_best_model=True`, else None

**Example:**

```python
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

# Save trained model
torch.save(model, 'output/model.pt')
```

#### `train_epoch(model, optim, batch_size, device, beta, masked_prob)`

Train for one epoch.

**Parameters:**
- `model` (VAE): Model to train
- `optim` (torch.optim.Optimizer): Optimizer
- `batch_size` (int): Batch size
- `device` (str): Device ('cuda' or 'cpu')
- `beta` (float): KL weight for this epoch
- `masked_prob` (float, default=0.0): Masking probability

**Side Effects:**
- Updates model parameters via backpropagation

#### `eval(model, device, partition)`

Evaluate model on specified partition.

**Parameters:**
- `model` (VAE): Model to evaluate
- `device` (str): Device ('cuda' or 'cpu')
- `partition` (str): Data partition ('train', 'val', or 'test')

**Returns:**
- `mse` (float): Mean squared error
- `r2` (float): R² score
- `elbo` (float): Evidence lower bound (negative)
- `kld` (float): KL divergence

**Example:**

```python
# Evaluate on test set
mse, r2, elbo, kld = trainer.eval(model, 'cuda', partition='test')

print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test ELBO: {elbo:.4f}")
print(f"Test KLD: {kld:.4f}")
```

#### `mask(x, prob)`

Apply masking for VIME self-supervision.

**Parameters:**
- `x` (torch.Tensor): Input tensor, shape (batch_size, input_dim)
- `prob` (float): Masking probability (0.0 = no masking)

**Returns:**
- `x_masked` (torch.Tensor): Masked input, shape (batch_size, input_dim)
- `mask` (torch.Tensor): Binary mask, shape (batch_size, input_dim)

**Example:**

```python
x_masked, mask = trainer.mask(x, prob=0.3)

# 30% of features are masked (replaced with permuted values)
print(f"Fraction masked: {mask.mean().item():.2f}")
```

---

## Utilities

### `amlvae.models.utils`

#### `get_nonlin(name)`

Get activation function by name.

**Parameters:**
- `name` (str): Activation name ('elu', 'relu', 'leakyrelu', 'tanh')

**Returns:**
- `nonlin` (torch.nn.Module): Activation function class

**Example:**

```python
from amlvae.models.utils import get_nonlin

elu = get_nonlin('elu')
activation = elu()  # Instantiate

x = torch.tensor([-1.0, 0.0, 1.0])
print(activation(x))  # tensor([-0.6321,  0.0000,  1.0000])
```

#### `get_norm(name)`

Get normalization layer by name.

**Parameters:**
- `name` (str): Normalization name ('layer', 'batch', or None)

**Returns:**
- `norm` (torch.nn.Module or None): Normalization class

**Example:**

```python
from amlvae.models.utils import get_norm

layer_norm = get_norm('layer')
norm = layer_norm(512)  # Normalize over 512 features

x = torch.randn(128, 512)
x_normed = norm(x)
print(f"Mean: {x_normed.mean(dim=1).mean():.4f}")  # ~0
print(f"Std: {x_normed.std(dim=1).mean():.4f}")    # ~1
```

---

## Type Definitions

### Common Types

```python
from typing import Tuple, List, Dict, Optional
import torch
import pandas as pd
import numpy as np

# Expression data
ExpressionMatrix = pd.DataFrame  # Shape: (N_samples, N_genes)
ExpressionArray = np.ndarray     # Shape: (N_samples, N_genes)
ExpressionTensor = torch.Tensor  # Shape: (batch_size, N_genes)

# Sample identifiers
SampleIDs = List[str]

# Gene identifiers
GeneNames = List[str]

# Model outputs
VAEOutput = Dict[str, torch.Tensor]  # {'xhat', 'mu', 'logvar', 'mask_hat'}

# Configuration
Config = Dict[str, any]  # Hyperparameter dictionary

# Normalization parameters
TransformParams = Dict[str, any]  # {'mu', 'sd', 'method'} or {'min', 'max', 'method'}
```

---

## Error Handling

### Common Exceptions

#### `ValueError`

Raised when:
- Invalid method name for gene selection or normalization
- Data processing steps called out of order (e.g., `get_data()` before `normalize_()`)
- Configuration missing required keys

**Example:**

```python
try:
    processor.select_genes_(method='invalid_method')
except ValueError as e:
    print(f"Error: {e}")
    # Output: Unknown method 'invalid_method'
```

#### `RuntimeError`

Raised when:
- No genes pass filtering criteria (WGCNA/TCGA methods)
- All genes have zero variance after filtering

**Example:**

```python
try:
    genes = select_genes_wgcna_protocol(expr, counts, top_n=10000)
except RuntimeError as e:
    print(f"Error: {e}")
    # Output: No genes passed the edgeR low-count filter.
```

#### `torch.cuda.OutOfMemoryError`

Raised when GPU memory is exhausted.

**Solution:**

```python
try:
    model(x)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Reduce batch size or model size
    print("Out of memory! Try smaller batch size.")
```

---

## Version Compatibility

### PyTorch Versions

AMLVAE is tested with:
- PyTorch 1.12+
- PyTorch 2.0+ (recommended)

**Note**: Ray Tune checkpoint API changed between versions. The code handles both:

```python
try:
    from ray.tune import Checkpoint
except:
    from ray.train import Checkpoint
```

### Python Versions

- Python 3.8+
- Python 3.9-3.10 (recommended)

### Dependencies

See `workflows/envs/amlvae.yaml` for complete dependency list with versions.

---

## Examples

### Complete Workflow Example

```python
import pandas as pd
import torch
import numpy as np
from amlvae.data import ExprProcessor
from amlvae.models import VAE
from amlvae.train import Trainer

# 1. Load and preprocess data
expr_long = pd.read_csv('data/aml_full_manuscript.csv')

processor = ExprProcessor(
    expr_long,
    target='fpkm_uq_unstranded',
    counts_name='unstranded'
)
processor.select_genes_(method='wgcna', top_n=2000)
processor.normalize_(method='zscore')
X, sample_ids = processor.get_data()

# 2. Save processed data
expr_df = pd.DataFrame(
    X,
    index=sample_ids,
    columns=processor.selected_genes
)
expr_df.to_csv('processed/aml_expr.csv')

# 3. Create partitions
from sklearn.model_selection import train_test_split

train_ids, temp_ids = train_test_split(sample_ids, test_size=0.4, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

partitions = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}
torch.save(partitions, 'processed/aml_partitions.pt')

# 4. Train model
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

# 5. Encode samples
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    z, _ = model.encode(X_tensor)
    z = z.cpu().numpy()

# 6. Save latent representations
z_df = pd.DataFrame(
    z,
    index=sample_ids,
    columns=[f'z{i+1}' for i in range(model.latent_dim)]
)
z_df.to_csv('output/aml_z.csv')

print("Workflow complete!")
print(f"Latent representations shape: {z.shape}")
```

---

**Last Updated**: October 2025  
**API Version**: 0.0







