# AMLVAE: Methodology

This document provides a detailed description of the methodology underlying AMLVAE, including theoretical foundations, implementation details, and design rationale.

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Gene Expression Preprocessing](#gene-expression-preprocessing)
3. [Model Architecture](#model-architecture)
4. [Training Procedure](#training-procedure)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Theoretical Background

### Variational Autoencoders (VAEs)

Variational Autoencoders are probabilistic generative models that learn a low-dimensional latent representation of high-dimensional data. Unlike standard autoencoders, VAEs explicitly model the latent space as a probability distribution.

#### Mathematical Formulation

Given gene expression data \( x \in \mathbb{R}^G \) (where \( G \) is the number of genes), the VAE learns:

1. **Encoder**: \( q_\phi(z|x) \) - Approximates the posterior distribution of latent variables \( z \in \mathbb{R}^L \)
2. **Decoder**: \( p_\theta(x|z) \) - Models the likelihood of data given latent variables

The encoder outputs parameters of a Gaussian distribution:

\[
q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x))
\]

where \( \mu_\phi \) and \( \sigma_\phi \) are neural networks parameterized by \( \phi \).

#### Loss Function

The VAE optimizes the Evidence Lower Bound (ELBO):

\[
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) \| p(z))
\]

This consists of two terms:

1. **Reconstruction Loss**: \( -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] \approx \|x - \hat{x}\|^2 \)
   - Encourages accurate reconstruction of input data
   - Implemented as Mean Squared Error (MSE) between input and reconstructed expression

2. **KL Divergence**: \( D_{KL}(q_\phi(z|x) \| p(z)) \)
   - Regularizes the latent space to follow prior \( p(z) = \mathcal{N}(0, I) \)
   - Prevents overfitting and encourages smooth, structured latent space
   - Has closed-form solution for Gaussian distributions

3. **β-VAE Weighting**: The parameter \( \beta \) controls the trade-off
   - \( \beta = 1 \): Standard VAE (ELBO)
   - \( \beta > 1 \): Stronger regularization, more disentangled representations
   - \( \beta < 1 \): Prioritizes reconstruction over regularization

#### Reparameterization Trick

To enable backpropagation through stochastic sampling, we use:

\[
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

This separates deterministic (μ, σ) and stochastic (ε) components, allowing gradients to flow through μ and σ.

### β-VAE and Disentanglement

The β-VAE framework (Higgins et al., 2017) introduces adjustable weighting of the KL term to encourage disentangled latent representations, where each latent dimension captures independent factors of variation.

**Benefits for Gene Expression**:
- Each latent dimension may correspond to distinct biological processes
- Reduces correlation between latent dimensions
- Improves interpretability of learned features

**Trade-off**: Higher β can reduce reconstruction quality but improves latent space structure.

### VIME: Value Imputation and Mask Estimation

AMLVAE optionally incorporates ideas from VIME (Yoon et al., 2020), a self-supervised learning approach:

1. **Masking**: Randomly mask a fraction of input genes with values from other samples
2. **Dual Prediction**: Train the model to:
   - Reconstruct the original (unmasked) expression
   - Predict which genes were masked (binary classification)

**Benefits**:
- Encourages robust representations that aren't overly dependent on individual genes
- Acts as a regularizer similar to dropout
- Improves generalization to noisy or incomplete data

**Implementation**: 
- A separate mask classifier head predicts the binary mask
- Loss combines reconstruction, KL divergence, and binary cross-entropy for mask prediction

---

## Gene Expression Preprocessing

High-quality preprocessing is critical for VAE performance. AMLVAE supports multiple preprocessing pipelines tailored to different data characteristics.

### 1. Gene Selection

#### Why Gene Selection?

RNA-seq typically quantifies ~20,000-60,000 genes, but:
- Many genes have low/no expression (uninformative)
- Many genes have constant expression (no variation to model)
- High dimensionality increases computational cost and overfitting risk
- Most biological variation captured by top 1,000-5,000 genes

#### Method A: Variance-Based Selection

**Procedure**:
1. Calculate variance of each gene across samples
2. Rank genes by variance (descending)
3. Select top N genes (typically N = 1,000-2,500)

**Advantages**: Simple, fast, no assumptions about data distribution

**Disadvantages**: 
- Doesn't account for mean-variance relationship
- May select noisy genes with high variance but low biological signal

**Implementation**: `select_genes_by_variance()` in `ExprProcessor.py`

#### Method B: TCGA Protocol

**Procedure** (based on Hoadley et al., 2018):
1. **Noise Filter**: Remove genes with RPKM ≤ 0.2 in ≥ 75% of samples
2. **Expression Filter**: Keep genes with median RPKM ≥ 10
3. **Variability Ranking**: Rank by coefficient of variation (CV = σ/μ)
4. **Selection**: Take top N genes (or top 25% as in original TCGA)

**Advantages**: 
- Filters out lowly expressed genes (likely noise)
- CV normalizes for mean-variance relationship
- Validated in large TCGA pan-cancer studies

**Disadvantages**: 
- Fixed thresholds may not suit all datasets
- Can remove lowly expressed but biologically important genes

**Implementation**: `select_genes_tcga()` in `ExprProcessor.py`

#### Method C: WGCNA Protocol

**Procedure** (reproduces edgeR + WGCNA workflow):
1. **CPM Filtering**: 
   - Calculate library sizes (total reads per sample)
   - Set CPM threshold: k = min_count × 10⁶ / min(library sizes)
   - Keep genes with CPM ≥ k in ≥ 66% of samples
   - Require total read count ≥ min_total_count
2. **Log Transform**: log₂(FPKM + 1)
3. **MAD Filter**: Remove genes with median absolute deviation (MAD) = 0
4. **Variance Ranking**: Rank by variance in log-space
5. **Selection**: Top N genes

**Advantages**: 
- Biologically motivated (edgeR is standard in RNA-seq)
- Removes flat genes (MAD filter)
- Accounts for library size differences

**Disadvantages**: 
- More complex, harder to tune
- Requires raw count data (not just FPKM)

**Implementation**: `select_genes_wgcna_protocol()` in `ExprProcessor.py`

### 2. Normalization

After gene selection, expression values are normalized to ensure:
- Similar scales across genes (prevents dominance by highly expressed genes)
- Numerical stability during training
- Comparable results across datasets

#### Method A: Z-Score Normalization

**Procedure**:
1. Log-transform: \( y = \log_2(x + 1) \)
2. Standardize per gene: \( z_g = \frac{y_g - \mu_g}{\sigma_g} \)

Where:
- \( x \): Raw FPKM/TPM values
- \( y \): Log-transformed values
- \( z_g \): Z-score for gene g
- \( \mu_g, \sigma_g \): Mean and standard deviation of gene g across samples

**Advantages**:
- Centers data at zero (suits VAE's Gaussian prior)
- Each gene has comparable scale
- Standard practice in genomics

**Disadvantages**:
- Sensitive to outliers
- Changes interpretation of values

**Implementation**: `normalize_zscore()` in `ExprProcessor.py`

#### Method B: Min-Max Normalization

**Procedure**:
1. Log-transform: \( y = \log_2(x + 1) \)
2. Scale to [0,1] per gene: \( z_g = \frac{y_g - \min(y_g)}{\max(y_g) - \min(y_g)} \)

**Advantages**:
- Bounded range [0,1] (can aid optimization)
- Preserves shape of distribution

**Disadvantages**:
- Very sensitive to outliers (single extreme value affects all samples)
- Loses information about relative expression levels between genes

**Implementation**: `normalize_minmax()` in `ExprProcessor.py`

**Recommendation**: Z-score normalization is preferred for most applications.

### 3. Data Partitioning

**Procedure**:
- Split data into training (typically 60-70%), validation (15-20%), and test (15-20%) sets
- Stratified splitting if clinical outcomes are available (ensures balanced representation)
- Random seed for reproducibility

**Purpose**:
- **Training**: Learn model parameters
- **Validation**: Monitor overfitting, tune hyperparameters, early stopping
- **Test**: Final evaluation of model performance (never used during training/tuning)

**Implementation**: `partition.py` script creates `{dataset}_partitions.pt` file storing sample IDs for each partition

---

## Model Architecture

### Encoder

The encoder maps gene expression \( x \in \mathbb{R}^G \) to latent distribution parameters:

```
Input (G genes) 
  → Linear(G, H) + Norm + Activation + Dropout
  → Linear(H, H) + Norm + Activation + Dropout  [repeat n_layers - 1 times]
  → Linear(H, 2L)  [outputs μ and log(σ²) for L latent dimensions]
  → Split into μ (L dims) and logvar (L dims)
```

**Components**:
- **Linear Layers**: Fully connected with bias
- **Normalization**: Layer normalization (default) or batch normalization
  - Layer norm preferred for variable batch sizes
- **Activation**: ELU (default), ReLU, or LeakyReLU
  - ELU allows negative values (important for z-scored data)
- **Dropout**: Optional regularization (typically 0 or low values)

**Output**: 
- μ: Mean of latent distribution
- logvar: Log-variance (for numerical stability)

### Latent Sampling

Sample latent vector using reparameterization trick:

```python
z = μ + exp(0.5 * logvar) * ε,  where ε ~ N(0, I)
```

**Training**: Use sampled z (introduces stochasticity for regularization)  
**Inference**: Use μ directly (deterministic, stable representation)

### Decoder

The decoder reconstructs gene expression from latent variables \( z \in \mathbb{R}^L \):

```
Latent (L dims)
  → Linear(L, H) + Norm + Activation + Dropout
  → Linear(H, H) + Norm + Activation + Dropout  [repeat n_layers - 1 times]
  → Linear(H, G)  [outputs reconstructed expression for G genes]
```

**Symmetric Design**: Same number of layers and hidden units as encoder (but reversed)

**Output**: \( \hat{x} \in \mathbb{R}^G \) - Reconstructed gene expression (same scale as input)

### Mask Classifier (Optional, for VIME)

Predicts which genes were masked during self-supervised training:

```
Latent (L dims)
  → Linear(L, H) + Norm + Activation + Dropout
  → Linear(H, H) + Norm + Activation + Dropout  [repeat n_layers - 1 times]
  → Linear(H, G) + Sigmoid  [outputs probabilities for G genes]
```

**Output**: \( \hat{m} \in [0,1]^G \) - Predicted mask for each gene

### Configurable Hyperparameters

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| `n_latent` | 12-32 | Latent dimension size |
| `n_hidden` | 512 | Hidden layer width |
| `n_layers` | 2 | Number of hidden layers (encoder/decoder) |
| `norm` | 'layer' | Normalization: 'layer', 'batch', or None |
| `nonlin` | 'elu' | Activation: 'elu', 'relu', 'leakyrelu' |
| `dropout` | 0.0 | Dropout probability |
| `variational` | True | Use variational (stochastic) sampling |

---

## Training Procedure

### Optimization

**Optimizer**: Adam  
- Adaptive learning rates per parameter
- Default: lr = 1e-4, weight_decay (L2) = 0.0

**Batch Processing**:
- Mini-batch gradient descent (typical batch size: 128-256)
- Samples shuffled each epoch
- Full pass through training data = 1 epoch

### Loss Computation

For each batch:

1. **Forward Pass**:
   ```python
   μ, logvar = encoder(x)
   z = reparameterize(μ, logvar)  # Sample from latent distribution
   x_hat = decoder(z)
   ```

2. **Reconstruction Loss** (MSE):
   ```python
   L_recon = mean((x - x_hat)²)
   ```

3. **KL Divergence** (closed-form for Gaussians):
   ```python
   L_KL = -0.5 * mean(1 + logvar - μ² - exp(logvar))
   ```

4. **Total Loss**:
   ```python
   L_total = L_recon + β * L_KL
   ```

5. **Backward Pass**: Compute gradients via backpropagation

6. **Parameter Update**: Apply Adam optimizer step

### β-Annealing

To prevent posterior collapse (where the model ignores the latent space), β is gradually increased:

```python
if epoch < T:  # T = 75% of total epochs
    β = β_target * 0.5 * (1 - cos(π * epoch / T))  # Cosine annealing
else:
    β = β_target
```

**Effect**:
- Early training: Focus on reconstruction (low β)
- Later training: Gradually enforce latent space structure (increase β to target)
- Helps model learn meaningful latent representations

**Alternative**: Set `anneal=False` for constant β throughout training

### Early Stopping

Monitor validation set performance to prevent overfitting:

1. Evaluate validation ELBO after each epoch
2. Track best validation ELBO and number of epochs without improvement (patience counter)
3. If patience counter exceeds threshold (e.g., 100-1500 epochs), stop training
4. Return model with best validation ELBO

**Benefits**:
- Prevents overfitting to training data
- Reduces unnecessary computation
- Automatically determines optimal training duration

### VIME Masking (Optional)

If `masked_prob > 0`:

1. **Mask Input**:
   ```python
   mask = Bernoulli(masked_prob)  # Binary mask (0 = keep, 1 = mask)
   x_permuted = permute_genes(x)  # Shuffle within each sample
   x_masked = x * (1 - mask) + x_permuted * mask
   ```

2. **Forward Pass**: Use `x_masked` as input

3. **Additional Loss**:
   ```python
   m_hat = mask_classifier(z)  # Predict mask
   L_mask = binary_cross_entropy(mask, m_hat)
   L_total = L_recon + β * L_KL + L_mask
   ```

**Recommendation**: Start with `masked_prob = 0.0`, add masking if overfitting occurs

---

## Evaluation Metrics

### Reconstruction Quality

**Mean Squared Error (MSE)**:
\[
\text{MSE} = \frac{1}{N \cdot G} \sum_{i=1}^{N} \sum_{g=1}^{G} (x_{ig} - \hat{x}_{ig})^2
\]

- Lower is better
- Directly measures reconstruction accuracy
- Sensitive to scale of data

**R² Score (Coefficient of Determination)**:
\[
R^2 = 1 - \frac{\sum_{i,g} (x_{ig} - \hat{x}_{ig})^2}{\sum_{i,g} (x_{ig} - \bar{x}_g)^2}
\]

- Range: (-∞, 1], where 1 = perfect reconstruction
- Measures proportion of variance explained
- Accounts for baseline variability

**Evaluation Protocol**:
- Compute metrics on train, validation, and test sets separately
- Use deterministic encoding (μ, not sampled z) for stable evaluation
- Compare against PCA baseline with same latent dimension

### Evidence Lower Bound (ELBO)

\[
\text{ELBO} = -L_{\text{recon}} - L_{\text{KL}}
\]

- Primary metric for VAE optimization
- Higher ELBO = better fit
- Used for model selection and early stopping

### Baseline Comparison: PCA

Train PCA with same number of components as VAE latent dimension:

```python
pca = PCA(n_components=n_latent)
pca.fit(X_train)
X_hat = pca.inverse_transform(pca.transform(X_test))
```

Compare VAE vs PCA on test set MSE and R²:
- If VAE ≈ PCA: Model learned linear relationships (VAE may be overkill)
- If VAE >> PCA: Model captured nonlinear patterns (VAE justified)

### Clinical Validation

**UMAP Visualization**:
- Project latent representations to 2D using UMAP
- Color points by clinical variables:
  - Survival time (continuous)
  - AML subtypes (categorical)
  - Genetic fusions (binary)
- Assess whether latent space separates clinically relevant groups

**Correlation Analysis**:
- Compute Spearman correlation between each latent dimension and clinical variables
- Adjust p-values for multiple testing (Benjamini-Hochberg FDR)
- Identify latent dimensions significantly associated with outcomes

**Silhouette Score** (for known clusters):
\[
s = \frac{b - a}{\max(a, b)}
\]
where a = intra-cluster distance, b = inter-cluster distance

---

## Hyperparameter Tuning

### Search Space

Typical hyperparameters to tune:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `lr` | [1e-5, 1e-3] | Learning rate (log scale) |
| `n_hidden` | [128, 256, 512, 1024] | Hidden layer width |
| `n_layers` | [1, 2, 3, 4] | Number of layers |
| `n_latent` | [8, 12, 16, 24, 32, 64] | Latent dimension |
| `beta` | [0.1, 0.5, 1.0, 2.0, 5.0] | KL weight |
| `dropout` | [0.0, 0.1, 0.2, 0.3] | Dropout rate |
| `l2` | [0.0, 1e-5, 1e-4, 1e-3] | Weight decay |
| `batch_size` | [64, 128, 256, 512] | Batch size |

### Tuning Procedure (Ray Tune)

1. **Define Search Space**: Specify distributions for each hyperparameter
   ```python
   config = {
       "lr": tune.loguniform(1e-5, 1e-3),
       "n_hidden": tune.choice([128, 256, 512, 1024]),
       "n_latent": tune.choice([8, 12, 16, 24, 32]),
       # ... etc
   }
   ```

2. **Select Search Algorithm**:
   - Random search: Sample configurations uniformly
   - Bayesian optimization: Model performance surface, sample promising regions
   - Population-based training: Evolve configurations over time

3. **Define Objective**: Maximize validation R² (or minimize validation MSE/ELBO)

4. **Run Trials**:
   - Train models with different configurations in parallel (multi-GPU)
   - Each trial runs for full training (with early stopping)
   - Report validation metrics after each epoch

5. **Select Best**: Choose configuration with best validation performance

6. **Retrain**: Train final model on train+validation data with best hyperparameters

**Computational Cost**:
- Tuning 100 configurations × 1500 epochs = ~150,000 training epochs
- Requires GPU cluster or cloud resources
- Consider starting with smaller search space or fewer trials

### Practical Tips

1. **Start Simple**: First tune architecture (layers, hidden units) with fixed lr/beta
2. **Then Regularization**: Tune beta, dropout, L2 with fixed architecture
3. **Finally Optimization**: Tune lr, batch size
4. **Warm Start**: Use good configurations from previous runs as starting points
5. **Early Stopping**: Set patience to 100-200 for tuning (faster), then increase to 1500 for final training
6. **Monitor Overfitting**: Watch train vs validation gap

---

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.
- Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR.
- Yoon et al. (2020). VIME: Value Imputation and Mask Estimation. NeurIPS.
- Hoadley et al. (2018). Cell-of-Origin Patterns Dominate the Molecular Classification of 10,000 Tumors from 33 Types of Cancer. Cell.
- Lopez et al. (2018). Deep generative modeling for single-cell transcriptomics. Nature Methods.

---

**Last Updated**: October 2025







