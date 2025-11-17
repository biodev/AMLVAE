# AMLVAE Documentation

Welcome to the AMLVAE documentation! This directory contains comprehensive documentation for the AMLVAE (Acute Myeloid Leukemia Variational Autoencoder) project.

## Documentation Structure

### Core Documentation

- **[SUMMARY.md](SUMMARY.md)** - Executive summary with approach, data, methodology, limitations, and future improvements
- **[METHODOLOGY.md](METHODOLOGY.md)** - Detailed description of the methodology and theoretical background
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Model architecture and implementation details
- **[DATA.md](DATA.md)** - Data sources, preprocessing, and format specifications
- **[USAGE.md](USAGE.md)** - Practical guide for using the codebase
- **[API.md](API.md)** - API reference for key modules and functions

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AMLVAE

# Create conda environment
conda env create -f workflows/envs/amlvae.yaml
conda activate amlvae

# Install the package
pip install -e .
```

### Basic Workflow

1. **Prepare Data**: Process raw gene expression data
2. **Train Model**: Fit VAE to training data with hyperparameter tuning
3. **Evaluate**: Assess reconstruction quality and clinical associations
4. **Analyze**: Extract latent representations and visualize results

For detailed instructions, see [USAGE.md](USAGE.md).

## Project Overview

AMLVAE is a deep learning framework for analyzing gene expression data from Acute Myeloid Leukemia (AML) patients. The model uses Variational Autoencoders (VAEs) to:

- **Compress** high-dimensional gene expression into low-dimensional latent representations
- **Denoise** gene expression profiles by learning robust features
- **Discover** biologically meaningful patterns in patient data
- **Enable** downstream tasks like clustering, classification, and survival prediction

### Key Features

- **Flexible Architecture**: Configurable encoder/decoder depths, widths, and normalization strategies
- **Multiple Preprocessing Pipelines**: Support for TCGA, WGCNA, and variance-based gene selection
- **Hyperparameter Optimization**: Integrated Ray Tune for automated model selection
- **Clinical Integration**: Tools for associating latent dimensions with clinical outcomes
- **Visualization**: UMAP/t-SNE projections with clinical metadata overlay

## Repository Structure

```
AMLVAE/
├── amlvae/                 # Core package
│   ├── data/              # Data processing modules
│   ├── models/            # VAE and MLP architectures
│   ├── train/             # Training loops and utilities
│   └── utils/             # Helper functions
├── workflows/             # Snakemake/scripts for pipelines
│   ├── AML/              # AML-specific configurations
│   ├── MDS/              # MDS-specific configurations
│   └── scripts/          # Standalone scripts (train, eval, tune)
├── data/                  # Raw and processed data
├── notebooks/             # Jupyter notebooks for exploration
├── docs/                  # Documentation (this directory)
└── deprecated/            # Old model implementations
```

## Citing This Work

If you use AMLVAE in your research, please cite:

```
@software{amlvae2025,
  title = {AMLVAE: Variational Autoencoders for AML Gene Expression Analysis},
  year = {2025},
  version = {0.0}
}
```

## Contributing

Contributions are welcome! Please see our contributing guidelines for:
- Code style conventions
- Testing requirements
- Documentation standards
- Pull request process

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Consult the documentation in this directory
- Check existing notebooks for examples

## License

See [LICENSE](../LICENSE) file in the root directory.

---

**Documentation Version**: 1.0  
**Last Updated**: October 2025







