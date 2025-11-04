# AMLVAE: Architecture Documentation

This document provides detailed technical specifications of the AMLVAE model architecture, implementation details, and design decisions.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Model Components](#model-components)
3. [Implementation Details](#implementation-details)
4. [Code Organization](#code-organization)
5. [Computational Requirements](#computational-requirements)
6. [Design Decisions](#design-decisions)

---

## System Architecture

### Overview

AMLVAE follows a modular architecture separating data processing, model definition, training, and evaluation:

```
┌─────────────────────────────────────────────────────────┐
│                    Raw Data                             │
│  (Gene Expression CSV, Clinical Data)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Data Preprocessing                         │
│  - Gene Selection (Variance/TCGA/WGCNA)                 │
│  - Normalization (Z-score/Min-Max)                      │
│  - Train/Val/Test Partitioning                          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Model Training                             │
│  - VAE Architecture (Encoder/Decoder)                   │
│  - Optimization (Adam + Early Stopping)                 │
│  - Hyperparameter Tuning (Ray Tune)                     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Evaluation                                 │
│  - Reconstruction Metrics (MSE, R²)                     │
│  - Latent Space Extraction                              │
│  - Clinical Association Analysis                        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│            Downstream Applications                      │
│  - Visualization (UMAP, t-SNE)                          │
│  - Clustering & Classification                          │
│  - Drug Response Prediction                             │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```python
# 1. Load and preprocess
expr_long = pd.read_csv('aml_full.csv')
processor = ExprProcessor(expr_long)
processor.select_genes_(method='wgcna', top_n=2000)
processor.normalize_(method='zscore')
X, sample_ids = processor.get_data()

# 2. Partition
partitions = create_partitions(sample_ids)
X_train, X_val, X_test = load_partitions(X, partitions)

# 3. Train
model = VAE(input_dim=2000, latent_dim=32, ...)
trainer = Trainer(root='processed/', ...)
trained_model = trainer(config)

# 4. Evaluate
z_train, _ = model.encode(X_train)
z_test, _ = model.encode(X_test)
evaluate_reconstruction(model, X_test)

# 5. Visualize
embedding = umap.fit_transform(z_test)
plot_with_clinical(embedding, clinical_data)
```

---

## Model Components

### 1. MLP Module (`amlvae/models/MLP.py`)

**Purpose**: Flexible multi-layer perceptron used as building block for encoder/decoder

**Architecture**:
```python
class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,      # Input dimension
        hidden_channels: int,  # Hidden layer width
        out_channels: int,     # Output dimension
        layers: int = 2,       # Number of hidden layers
        dropout: float = 0.0,  # Dropout probability
        nonlin = nn.ELU,       # Activation function class
        norm = nn.LayerNorm,   # Normalization class
        out = None,            # Output activation (e.g., nn.Sigmoid)
        bias: bool = True      # Use bias in linear layers
    )
```

**Layer Structure**:
```
Input (in_channels)
  → Linear(in, hidden) + Norm(hidden) + Activation + Dropout
  → [Linear(hidden, hidden) + Norm + Activation + Dropout] × (layers - 1)
  → Linear(hidden, out)
  → [Optional Output Activation]
```

**Example Instantiation**:
```python
encoder = MLP(
    in_channels=2000,        # 2000 genes
    hidden_channels=512,     # 512 hidden units
    out_channels=64,         # 2 × 32 (μ and log σ² for 32 latent dims)
    layers=2,                # 2 hidden layers
    nonlin=nn.ELU,
    norm=nn.LayerNorm,
    dropout=0.0
)
```

**Forward Pass**:
```python
x = torch.randn(128, 2000)  # Batch of 128 samples, 2000 genes
h = encoder(x)               # h.shape = (128, 64)
```

**Flexibility**:
- Different normalization strategies (LayerNorm, BatchNorm, None)
- Configurable activation functions (ELU, ReLU, LeakyReLU)
- Optional output transformations (Sigmoid for mask prediction)

### 2. VAE Module (`amlvae/models/VAE.py`)

**Purpose**: Complete VAE implementation with encoder, decoder, and optional components

#### Initialization

```python
class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,           # Number of genes
        hidden_dim: int,          # Hidden layer width
        n_layers: int,            # Number of layers in encoder/decoder
        latent_dim: int,          # Latent space dimension
        conditions: dict = {},    # Adversarial tasks (deprecated)
        norm: str = 'layer',      # Normalization type
        nonlin: str = 'elu',      # Activation function
        variational: bool = True, # Use stochastic sampling
        dropout: float = 0.0      # Dropout rate
    )
```

**Components Created**:

1. **Encoder** (MLP):
   - Input: `input_dim` genes
   - Output: `2 * latent_dim` (concatenated μ and log σ²)

2. **Decoder** (MLP):
   - Input: `latent_dim`
   - Output: `input_dim` genes

3. **Mask Classifier** (MLP):
   - Input: `latent_dim`
   - Output: `input_dim` (probability each gene was masked)
   - Used for VIME self-supervision

#### Forward Pass

```python
def forward(self, x):
    # Encode to latent distribution parameters
    mu, logvar = self.encode(x)
    
    # Sample latent vector (if variational=True)
    if self.variational:
        z = self.reparameterize(mu, logvar)
    else:
        z = mu
    
    # Decode to reconstruction
    xhat = self.decode(z)
    
    # Predict mask (for VIME)
    mask_hat = self.mask_classifier(z).sigmoid()
    
    return {
        'xhat': xhat,          # Reconstructed expression
        'mu': mu,              # Latent mean
        'logvar': logvar,      # Latent log-variance
        'mask_hat': mask_hat   # Predicted mask
    }
```

**Detailed Methods**:

**Encoding**:
```python
def encode(self, x):
    """Maps input to latent distribution parameters"""
    h = self.encoder(x)              # h.shape = (batch, 2*latent_dim)
    mu, logvar = h.chunk(2, dim=-1)  # Split into μ and log σ²
    return mu, logvar
```

**Reparameterization**:
```python
def reparameterize(self, mu, logvar):
    """Sample from N(μ, σ²) using reparameterization trick"""
    std = torch.exp(0.5 * logvar)    # σ = exp(0.5 * log σ²)
    eps = torch.randn_like(std)      # ε ~ N(0, I)
    z = mu + eps * std               # z = μ + σ * ε
    return z
```

**Decoding**:
```python
def decode(self, z):
    """Maps latent vector to reconstructed expression"""
    return self.decoder(z)
```

**Prediction** (for evaluation):
```python
def predict(self, x):
    """Deterministic reconstruction using mean encoding"""
    mu, logvar = self.encode(x)
    xhat = self.decode(mu)  # Use μ directly (no sampling)
    return xhat
```

#### Loss Function

```python
@staticmethod
def loss(x, xhat, mu, logvar, beta=1.0, mask=None, mask_hat=None):
    """
    Compute VAE loss = reconstruction + β * KL + mask loss
    
    Args:
        x: Input gene expression (batch, genes)
        xhat: Reconstructed gene expression (batch, genes)
        mu: Latent mean (batch, latent_dim)
        logvar: Latent log-variance (batch, latent_dim)
        beta: Weight for KL divergence term
        mask: True mask for VIME (batch, genes)
        mask_hat: Predicted mask (batch, genes)
    
    Returns:
        total_loss, recon_loss, kld, mask_loss
    """
    # 1. Reconstruction loss (MSE)
    recon_loss = F.mse_loss(xhat, x, reduction='sum') / x.size(0)
    
    # 2. KL divergence (closed form for Gaussians)
    # KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log σ² - μ² - σ²)
    std = torch.exp(0.5 * logvar)
    P = torch.distributions.Normal(mu, std)
    Q = torch.distributions.Normal(0, 1)
    kld = torch.distributions.kl_divergence(P, Q).sum() / x.size(0)
    
    # 3. Mask prediction loss (VIME)
    if mask is not None:
        mask_loss = F.binary_cross_entropy(
            mask_hat, mask, reduction='mean'
        )
    else:
        mask_loss = 0.0
    
    # Total loss
    total_loss = recon_loss + beta * kld + mask_loss
    
    return total_loss, recon_loss, kld, mask_loss
```

**Loss Components**:

1. **Reconstruction Loss**: 
   - \( L_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2 \)
   - Measures how well the model reconstructs input

2. **KL Divergence**:
   - \( L_{\text{KL}} = \frac{1}{N} \sum_{i=1}^{N} D_{KL}(q(z|x_i) \| p(z)) \)
   - Regularizes latent space to follow standard normal prior

3. **Mask Loss** (optional):
   - \( L_{\text{mask}} = \text{BCE}(\text{mask}, \hat{\text{mask}}) \)
   - Self-supervised task to predict masked genes

### 3. Trainer Module (`amlvae/train/Trainer.py`)

**Purpose**: Handles data loading, training loops, validation, and early stopping

#### Initialization

```python
class Trainer:
    def __init__(
        self,
        root: str,                     # Path to processed data
        dataset_name: str = 'aml',     # Dataset prefix
        checkpoint: bool = False,      # Enable Ray Tune checkpointing
        log_every: int = 250,          # Logging frequency
        epochs: int = 500,             # Maximum epochs
        verbose: bool = False,         # Print progress
        patience: int = 100,           # Early stopping patience
        return_best_model: bool = False  # Return trained model or None
    )
```

**Data Loading**:
```python
# Load expression data
data = pd.read_csv(f'{root}/{dataset_name}_expr.csv')
data = data.set_index(data.columns[0])

# Load partitions
partitions = torch.load(f'{root}/{dataset_name}_partitions.pt')

# Create tensors
self.X_train = torch.tensor(data.loc[partitions['train_ids']].values)
self.X_val = torch.tensor(data.loc[partitions['val_ids']].values)
self.X_test = torch.tensor(data.loc[partitions['test_ids']].values)
```

#### Training Loop

```python
def __call__(self, config):
    """
    Train VAE with given configuration
    
    Args:
        config: Dict with hyperparameters (lr, batch_size, beta, etc.)
    
    Returns:
        Trained model (if return_best_model=True)
    """
    # 1. Initialize model
    model = VAE(
        input_dim=self.X_train.size(1),
        hidden_dim=config['n_hidden'],
        n_layers=config['n_layers'],
        latent_dim=config['n_latent'],
        # ... other config params
    ).to(device)
    
    # 2. Initialize optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['l2']
    )
    
    # 3. Training loop
    best_elbo = float('inf')
    patience_count = 0
    best_model = None
    
    for epoch in range(self.epochs):
        # Anneal beta (if enabled)
        beta = compute_beta(epoch, config)
        
        # Train one epoch
        self.train_epoch(model, optim, config['batch_size'], device, beta)
        
        # Validate
        mse, r2, elbo, kld = self.eval(model, device, partition='val')
        
        # Track best model
        if elbo < best_elbo:
            best_elbo = elbo
            best_model = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        
        # Early stopping
        if patience_count > self.patience:
            break
        
        # Log progress (if verbose)
        if self.verbose:
            print(f'Epoch {epoch}: MSE={mse:.4f}, R²={r2:.2f}, KLD={kld:.2f}')
    
    # Load best model
    model.load_state_dict(best_model)
    
    if self.return_best_model:
        return model
```

**Key Methods**:

**Single Epoch Training**:
```python
def train_epoch(self, model, optim, batch_size, device, beta):
    model.train()
    for ixs in torch.split(torch.randperm(len(self.X_train)), batch_size):
        # Get batch
        x = self.X_train[ixs].to(device)
        
        # Optional masking (VIME)
        if masked_prob > 0:
            x_in, mask = self.mask(x, masked_prob)
        else:
            x_in = x
            mask = None
        
        # Forward pass
        optim.zero_grad()
        out = model(x_in)
        loss, _, _, _ = model.loss(x, beta=beta, mask=mask, **out)
        
        # Backward pass
        loss.backward()
        optim.step()
```

**Evaluation**:
```python
def eval(self, model, device, partition='val'):
    # Select data partition
    if partition == 'val':
        X = self.X_val
    elif partition == 'test':
        X = self.X_test
    else:
        X = self.X_train
    
    # Compute metrics
    model.eval()
    with torch.no_grad():
        out = model(X.to(device))
        loss, mse, kld, _ = model.loss(X.to(device), beta=0, **out)
        r2 = r2_score(X.cpu().numpy(), out['xhat'].cpu().numpy())
    
    return mse.item(), r2, loss.item(), kld.item()
```

### 4. ExprProcessor Module (`amlvae/data/ExprProcessor.py`)

**Purpose**: Handle gene expression preprocessing (selection, normalization)

See [DATA.md](DATA.md) for detailed documentation of preprocessing methods.

---

## Implementation Details

### Tensor Shapes Throughout Pipeline

```python
# Input: Long-form DataFrame
expr_long.shape  # (N_samples × N_genes, columns)

# After pivoting: Wide-form DataFrame
expr_wide.shape  # (N_samples, N_genes)

# After gene selection
X_selected.shape  # (N_samples, top_n_genes)

# Training batch
batch.shape  # (batch_size, top_n_genes)

# Encoder output
mu.shape     # (batch_size, latent_dim)
logvar.shape # (batch_size, latent_dim)

# Latent sample
z.shape      # (batch_size, latent_dim)

# Decoder output (reconstruction)
xhat.shape   # (batch_size, top_n_genes)

# Mask classifier output
mask_hat.shape  # (batch_size, top_n_genes)
```

### GPU vs CPU Execution

```python
# Check device availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move model to device
model = VAE(...).to(device)

# Move data to device during training/inference
x = x.to(device)
out = model(x)

# Move results back to CPU for numpy operations
z_cpu = z.cpu().numpy()
```

**Memory Considerations**:
- Large models (2000 genes, 1024 hidden units) may require 4-16 GB GPU memory
- Use smaller batch sizes if OOM (Out of Memory) errors occur
- Consider gradient accumulation for effectively larger batches

### Numerical Stability

**Log-Variance Instead of Variance**:
```python
# Unstable: σ can become very small or large
std = encoder_output  # Could be 1e-10 or 1e10

# Stable: log(σ²) can be any real number
logvar = encoder_output
std = torch.exp(0.5 * logvar)  # Always positive
```

**Log-Transform of Expression**:
```python
# Avoid log(0)
expr_log = np.log2(expr + 1)  # +1 pseudo-count
```

**Small Epsilon in Normalization**:
```python
# Avoid division by zero
normed = (x - mu) / (sd + 1e-8)
```

### Gradient Flow

```python
# Reparameterization allows gradients through μ and σ
z = mu + sigma * eps  # ∂z/∂μ = 1, ∂z/∂σ = eps

# Without reparameterization:
z = sample_from(Normal(mu, sigma))  # No gradient!
```

---

## Code Organization

### Package Structure

```
amlvae/
├── __init__.py              # Package initialization
├── data/
│   ├── __init__.py
│   ├── ExprProcessor.py     # Gene expression preprocessing
│   ├── ClinProcessor.py     # Clinical data processing
│   └── MutProcessor.py      # Mutation data processing
├── models/
│   ├── __init__.py
│   ├── MLP.py               # Multi-layer perceptron
│   ├── VAE.py               # Variational autoencoder
│   └── utils.py             # Model utilities (activations, norms)
├── train/
│   ├── __init__.py
│   └── Trainer.py           # Training loops and evaluation
└── utils/
    ├── __init__.py
    └── tune_parsing.py      # Ray Tune utilities
```

### Workflows Structure

```
workflows/
├── AML/
│   └── config.yaml          # AML-specific configuration
├── MDS/
│   └── config.yaml          # MDS-specific configuration
└── scripts/
    ├── proc.py              # Preprocessing script
    ├── partition.py         # Data partitioning
    ├── train.py             # Training script
    ├── tune.py              # Hyperparameter tuning
    ├── eval.py              # Evaluation script
    ├── clin_viz.py          # Clinical visualization
    ├── clin_eval.py         # Clinical association analysis
    ├── distances.py         # Distance metric computation
    ├── SNF.py               # Similarity network fusion
    └── graph_eval.py        # Graph-based evaluation
```

### Configuration Management

**YAML Configuration** (`workflows/AML/config.yaml`):

```yaml
run_id: "aml_wgcna_zscore_latent32"
data_dir: "../../../data"
n_latent: 32

proc:
  target_type: "fpkm_unstranded"
  gene_selection_method: "wgcna"
  num_top_genes: 2000
  norm_method: "zscore"
  dataset_name: "aml"

tune:
  target_metric: "r2"
  num_samples: 100
  epochs: 1500
  patience: 1500

train:
  epochs: 1500
  patience: 1500
```

**Loading Configuration**:
```python
import yaml

with open('workflows/AML/config.yaml') as f:
    config = yaml.safe_load(f)

# Access nested values
gene_method = config['proc']['gene_selection_method']
```

---

## Computational Requirements

### Training Time Estimates

**Single Model Training** (on NVIDIA V100):

| Configuration | Time per Epoch | Total Time (1500 epochs) |
|---------------|----------------|--------------------------|
| Small (L=12, H=256) | ~2 seconds | ~50 minutes |
| Medium (L=32, H=512) | ~5 seconds | ~2 hours |
| Large (L=64, H=1024) | ~15 seconds | ~6 hours |

**Hyperparameter Tuning**:
- 100 trials × 2 hours = 200 GPU-hours (~8 GPU-days)
- Use Ray Tune with multiple GPUs for parallelization
- Early stopping reduces effective time

### Memory Requirements

| Component | Memory Usage |
|-----------|-------------|
| Model parameters (H=512, L=32, G=2000) | ~10 MB |
| Batch (256 samples × 2000 genes × 4 bytes) | ~2 MB |
| Gradients | ~10 MB |
| Optimizer state (Adam) | ~20 MB |
| **Total per model** | ~50 MB |

**Batch Size Limits** (16 GB GPU):
- Can train very large batches (4096+) with standard configurations
- Memory bottleneck is usually training data (if all loaded to GPU)
- Solution: Keep data on CPU, transfer batches as needed

### Disk Space

- Raw expression data: ~100 MB (long format)
- Processed data: ~10 MB (wide format, selected genes)
- Trained model: ~10 MB
- Latent representations: ~1 MB
- Tuning results: ~1 GB (100 models + metadata)

---

## Design Decisions

### Why VAE Over Standard Autoencoder?

1. **Probabilistic Latent Space**: Provides uncertainty estimates
2. **Regularization**: KL term prevents overfitting
3. **Smooth Interpolation**: Can generate intermediate samples
4. **Generative Capability**: Can sample new data (though not primary goal)

### Why MLP Over Other Architectures?

**Alternatives Considered**:
- **Convolutional Networks**: Assume spatial structure (genes aren't spatially arranged)
- **Transformers**: Expensive for high-dimensional data, no clear ordering of genes
- **Graph Networks**: Require gene-gene relationship graph (not always available)

**MLP Advantages**:
- Simple, interpretable
- Flexible (easily configure depth/width)
- Fast training and inference
- Sufficient for capturing gene-gene interactions through hidden layers

### Why Layer Normalization?

**Alternatives**:
- **Batch Normalization**: Sensitive to batch size, requires running statistics
- **No Normalization**: Can suffer from internal covariate shift

**Layer Norm Advantages**:
- Independent of batch size (works with small batches)
- No running statistics (simpler implementation)
- Normalizes across features (genes), preserves sample-to-sample variation

### Why ELU Activation?

**Alternatives**:
- **ReLU**: Can cause dead neurons (zero gradient for negative inputs)
- **LeakyReLU**: Better than ReLU but still non-smooth
- **Tanh**: Saturates for large inputs

**ELU Advantages**:
- Smooth everywhere (better gradients)
- Allows negative values (important for z-scored data)
- Empirically performs well for VAEs

### Why Cosine Annealing for β?

**Alternatives**:
- **Linear Annealing**: Simpler but abrupt changes
- **No Annealing**: Can cause posterior collapse
- **Exponential**: Too fast or too slow

**Cosine Advantages**:
- Smooth increase (gradual transition)
- Well-studied in deep learning (cosine learning rate schedules)
- Reaches target β smoothly

---

## Extending the Architecture

### Adding New Components

**Example: Add Drug Response Decoder**

```python
class VAE_DrugResponse(VAE):
    def __init__(self, *args, n_drugs=100, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add drug response decoder
        self.drug_decoder = MLP(
            in_channels=self.latent_dim,
            hidden_channels=self.hidden_dim,
            out_channels=n_drugs,
            layers=self.n_layers,
            norm=self.norm_layer,
            nonlin=self.nonlin
        )
    
    def forward(self, x):
        out = super().forward(x)
        
        # Predict drug response from latent
        drug_response = self.drug_decoder(out['mu'])
        out['drug_response'] = drug_response
        
        return out
```

**Usage**:
```python
model = VAE_DrugResponse(
    input_dim=2000,
    latent_dim=32,
    n_drugs=100,
    ...
)

out = model(x)
# out contains: xhat, mu, logvar, mask_hat, drug_response
```

### Custom Loss Functions

```python
def custom_loss(x, xhat, mu, logvar, y_drug, yhat_drug, beta=1.0):
    # Standard VAE loss
    recon_loss = F.mse_loss(xhat, x)
    kld = kl_divergence(mu, logvar)
    
    # Drug response loss
    drug_loss = F.mse_loss(yhat_drug, y_drug)
    
    # Combined loss
    total = recon_loss + beta * kld + 0.5 * drug_loss
    return total
```

---

**Last Updated**: October 2025

