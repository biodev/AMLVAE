# AMLVAE: Executive Summary

## Overview

AMLVAE is a deep learning framework that applies Variational Autoencoders (VAEs) to learn compact, biologically meaningful representations of gene expression data from Acute Myeloid Leukemia (AML) and Myelodysplastic Syndrome (MDS) patients. The model transforms high-dimensional gene expression profiles into low-dimensional latent spaces that capture essential biological variation while filtering out noise.

## Approach

The project employs a **Variational Autoencoder (VAE)** architecture with the following key components:

1. **Encoder-Decoder Architecture**: Multi-layer perceptrons (MLPs) encode gene expression profiles into a low-dimensional latent space and reconstruct them back to the original space
2. **Probabilistic Latent Representation**: Uses the reparameterization trick to learn distributions (μ, σ) rather than point estimates, enabling uncertainty quantification
3. **VIME-Inspired Self-Supervision**: Optional masking mechanism where random genes are masked and the model learns to predict both the reconstruction and which genes were masked
4. **β-VAE Framework**: Adjustable KL-divergence weighting (β parameter) to control the trade-off between reconstruction fidelity and latent space regularization
5. **Annealing Strategy**: Gradual increase of β during training (cosine annealing over 75% of training) to prevent posterior collapse

## Data Used

### Primary Data Sources
- **BeatAML Dataset**: Gene expression data (FPKM values) from AML patients
- **MDS Dataset**: Gene expression data from Myelodysplastic Syndrome patients
- **Clinical Data**: Patient metadata including survival outcomes, AML subtypes (FAB classification), and genetic fusions

### Data Characteristics
- **Input**: RNA-seq gene expression quantified as FPKM (Fragments Per Kilobase Million) or TPM (Transcripts Per Million)
- **Scale**: Typically 1,000-2,500 genes selected from ~20,000 total genes
- **Sample Size**: Hundreds of patient samples split into train/validation/test sets
- **Features**: Continuous expression values normalized to zero mean and unit variance

## Methodology

### 1. Data Preprocessing
- **Gene Selection**: Three methods available:
  - *Variance-based*: Select top N most variable genes
  - *TCGA Protocol*: Filter by noise threshold and coefficient of variation (as used in TCGA studies)
  - *WGCNA Protocol*: EdgeR-style CPM filtering + MAD filtering + variance ranking (reproduces R-based workflows)
- **Normalization**: log₂(FPKM + 1) transformation followed by either z-score or min-max scaling
- **Data Splitting**: Stratified partitioning into training, validation, and test sets

### 2. Model Training
- **Architecture**: Symmetric encoder-decoder with configurable hidden layers (typically 2 layers, 512 hidden units)
- **Latent Dimension**: Typically 12-32 dimensions
- **Loss Function**: 
  - Reconstruction loss (MSE between input and output)
  - KL divergence (regularization term weighted by β)
  - Optional mask prediction loss (VIME)
- **Optimization**: Adam optimizer with early stopping based on validation ELBO
- **Hyperparameter Tuning**: Ray Tune integration for automated hyperparameter search over architecture, learning rate, regularization, etc.

### 3. Evaluation
- **Reconstruction Quality**: MSE and R² metrics on held-out test data
- **Baseline Comparison**: Performance compared against PCA with equivalent dimensionality
- **Clinical Validation**: UMAP visualization colored by clinical variables (survival, AML subtypes, genetic fusions)
- **Biological Interpretation**: Analysis of latent dimensions and their correlations with clinical outcomes

### 4. Downstream Applications
- **Dimensionality Reduction**: Compression of gene expression for visualization and clustering
- **Feature Extraction**: Latent representations for downstream predictive modeling
- **Sample Similarity**: Distance metrics in latent space for identifying similar patients
- **Gene Attribution**: Understanding which genes contribute most to latent representations

## Potential Limitations

### Technical Limitations
1. **Sample Size Dependency**: VAE performance is sensitive to training set size; small datasets may lead to overfitting
2. **Gene Selection Bias**: Pre-filtering to top 1,000-2,500 genes may discard biologically relevant but low-variance genes
3. **Hyperparameter Sensitivity**: Model performance depends on careful tuning of architecture (layers, hidden units, latent dimensions) and regularization (β, dropout, L2)
4. **Computational Requirements**: Training requires GPU for reasonable training times, especially during hyperparameter search
5. **Interpretability**: While latent dimensions can be analyzed post-hoc, they don't have inherent biological meaning like pathways or gene sets

### Biological Limitations
1. **Batch Effects**: No explicit modeling of technical batch effects (sequencing platform, processing date, etc.)
2. **Tissue Heterogeneity**: Gene expression averages over cell populations; doesn't capture single-cell heterogeneity
3. **Dataset Specificity**: Models trained on AML/MDS may not generalize to other cancer types without retraining
4. **Causality**: Model captures correlations but cannot establish causal relationships between genes and phenotypes
5. **Clinical Validation**: Findings require validation in independent clinical cohorts and prospective studies

### Methodological Limitations
1. **Gaussian Assumption**: VAE assumes latent space follows normal distribution, which may not capture multimodal or skewed biological distributions
2. **Linear Decoder**: While the encoder is nonlinear, the final reconstruction relies on learned linear combinations
3. **Missing Data**: Current implementation doesn't explicitly handle missing gene expression values
4. **No Uncertainty Quantification**: While the model learns variance, this uncertainty isn't propagated to downstream analyses

## Potential Future Improvements

### Model Enhancements
1. **Conditional VAE**: Incorporate clinical covariates (age, sex, AML subtype) directly into the model architecture
2. **Hierarchical VAE**: Multi-level latent representations (e.g., pathway-level and gene-level)
3. **Mixture Models**: Replace single Gaussian prior with mixture-of-Gaussians to capture distinct AML subtypes
4. **Attention Mechanisms**: Add attention layers to identify which genes are most important for each sample

### Biological Integration
1. **Pathway Constraints**: Incorporate gene pathway/ontology information to encourage biologically interpretable latent dimensions
2. **Multi-Omic Integration**: Extend to jointly model gene expression, mutations, copy number alterations, and methylation
3. **Drug Response Prediction**: Add decoder branches to predict drug sensitivity from latent representations
4. **Survival Modeling**: Integrate survival analysis directly into the VAE objective

### Technical Improvements
1. **Batch Correction**: Integrate methods like scVI-style batch correction or adversarial training
2. **Uncertainty Quantification**: Bayesian neural networks or ensemble methods for robust uncertainty estimates
3. **Transfer Learning**: Pre-train on large pan-cancer datasets (TCGA, TARGET) and fine-tune on AML
4. **Scalability**: Implement mini-batch corrections for very large cohorts or federated learning for multi-site data

### Validation & Applications
1. **External Validation**: Test on independent AML cohorts (TCGA-LAML, TARGET-AML, etc.)
2. **Clinical Decision Support**: Develop risk stratification tools based on latent representations
3. **Biomarker Discovery**: Systematic search for latent dimensions associated with treatment response or survival
4. **Sample Size Analysis**: Power calculations to determine minimum cohort size for reliable training

### Software Engineering
1. **API Development**: RESTful API for model inference on new samples
2. **Interactive Visualization**: Web dashboard for exploring latent spaces and clinical associations
3. **Automated Pipelines**: Snakemake or Nextflow workflows for end-to-end reproducibility
4. **Documentation**: Expand tutorials with real-world use cases and troubleshooting guides

---

**Last Updated**: October 2025  
**Version**: 0.0  
**Maintainer**: AMLVAE Development Team







