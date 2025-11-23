# FincGAN: A Gan Framework of Imbalanced Node Classification on Heterogeneous Graph Neural Network

FincGAN is a fraud detection framework combining Graph Neural Networks (GNN) and Generative Adversarial Networks (GAN) to address class imbalance in fraud detection.

---

## Quick Start

### Automated Script

FincGAN supports multiple datasets with dedicated training scripts:

```bash
# Make scripts executable
chmod +x scripts/run_fincgan_amazon.sh
chmod +x scripts/run_fincgan_yelp.sh

# Run complete pipeline for Amazon dataset
./scripts/run_fincgan_amazon.sh

# Run complete pipeline for Yelp dataset
./scripts/run_fincgan_yelp.sh

# Quick test for Amazon (5-10 minutes)
./scripts/run_fincgan_amazon.sh --n-epoch-emb 5 --n-epoch-gan 2 --n-epoch-uu 10 --n-epoch-up 5 --n-epoch-train 10

# Quick test for Yelp (5-10 minutes)
./scripts/run_fincgan_yelp.sh --n-epoch-emb 5 --n-epoch-gan 2 --n-epoch-uu 10 --n-epoch-up 5 --n-epoch-train 10
```

**For manual step-by-step execution, see [Complete Workflow](docs/workflow.md).**

---

## Installation

### CPU Version (PyTorch 1.11.0 + CUDA 11.3)

```bash
# Run setup script (one command to install everything)
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

### GPU Version (PyTorch 2.6.0 + CUDA 12.4)

```bash
# Run GPU setup script
chmod +x scripts/setup_env_gpu.sh
./scripts/setup_env_gpu.sh
```

**For manual setup and different CUDA versions, see [Installation Guide](docs/installation.md).**

---

## FincGAN Pipeline

<p align="center">
  <img src="images/fincgan_framework_illustration.png" alt="FincGAN Framework" width="500">
</p>

```
Stage I: Feature Extraction (HGT embeddings)
    ↓
Stage II: Node Generator (GAN training)
    ↓
Stage III: Edge Generators (UU & UP)
    ↓
Stage IV: Graph Generation & Training
    ↓
Stage V: Visualization & Analysis
```

---

## Documentation

### Getting Started
- [Installation Guide](docs/installation.md)
- [Dataset Information](docs/dataset.md)
- [Complete Workflow](docs/workflow.md)

### Pipeline Stages
1. [Stage I: Feature Extraction](docs/stage1-embedding.md)
2. [Stage II: Node Generator](docs/stage2-node-generator.md)
3. [Stage III: Edge Generators](docs/stage3-edge-generator.md)
4. [Stage IV: Training](docs/stage4-training.md)
5. [Stage V: Visualization](docs/stage5-visualization.md)

### Resources
- [Troubleshooting](docs/troubleshooting.md)

---

## Project Structure

```
fincgan/
├── README.md                    # Project documentation
├── LICENSE                      # License file
│
├── fincgan/                     # Python source code package
│   ├── __init__.py             # Package initialization
│   ├── train.py                # Main training script
│   ├── hgt_model.py            # HGT model implementation
│   ├── node_generator.py       # GAN training for node generation
│   ├── edge_generator_uu.py    # User-User edge generator
│   ├── edge_generator_up.py    # User-Product edge generator
│   ├── graph_generator.py      # Graph generation
│   ├── utils.py                # Utility functions
│   ├── visualize.py            # Visualization tools
│   └── logger.py               # Logging utilities
│
├── scripts/                     # Automation and configuration
│   ├── run_fincgan_amazon.sh   # Automated pipeline for Amazon dataset
│   ├── run_fincgan_yelp.sh     # Automated pipeline for Yelp dataset
│   ├── setup_env.sh            # CPU environment setup
│   ├── setup_env_gpu.sh        # GPU environment setup
│   ├── environment.yml         # CPU dependencies
│   └── environment-gpu.yml     # GPU dependencies
│
├── docs/                        # Documentation
│   ├── installation.md
│   ├── workflow.md
│   ├── stage1-embedding.md
│   ├── stage2-node-generator.md
│   ├── stage3-edge-generator.md
│   ├── stage4-training.md
│   ├── stage5-visualization.md
│   └── troubleshooting.md
│
├── data/                        # Input data directory
├── graph/                       # Original graph data
│   ├── amazon.bin              # Amazon dataset
│   └── yelp.bin                # Yelp dataset
├── images/                      # Images for documentation
│
└── (Output directories - generated during training)
    ├── embed/                   # Node embeddings
    │   ├── amazon/             # Amazon embeddings
    │   └── yelp/               # Yelp embeddings
    ├── generator/               # GAN generators
    │   ├── amazon/             # Amazon generators
    │   └── yelp/               # Yelp generators
    ├── graph_output/            # Generated graphs
    │   ├── amazon/             # Amazon graphs
    │   └── yelp/               # Yelp graphs
    ├── results/                 # Training results
    │   ├── amazon/             # Amazon results
    │   └── yelp/               # Yelp results
    ├── figures/                 # Visualization outputs
    │   ├── amazon/             # Amazon visualizations
    │   └── yelp/               # Yelp visualizations
    ├── tsne/                    # t-SNE visualizations
    │   ├── amazon/             # Amazon t-SNE plots
    │   └── yelp/               # Yelp t-SNE plots
    ├── tmp/                     # Temporary files
    │   ├── amazon/             # Amazon temp files
    │   └── yelp/               # Yelp temp files
    └── logs/                    # Training logs
```

---

## Citation

If you find this repository useful, please cite FincGAN with the following:

```bibtex
@INPROCEEDINGS{10448064,
  author    = {Hsu, Hung Chun and Lin, Ting-Le and Wu, Bo-Jun and Hong, Ming-Yi and Lin, Che and Wang, Chih-Yu},
  booktitle = {ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title     = {FincGAN: A Gan Framework of Imbalanced Node Classification on Heterogeneous Graph Neural Network},
  year      = {2024},
  pages     = {5750-5754},
  doi       = {10.1109/ICASSP48485.2024.10448064}
}
```

---

## Quick Links

- [Installation →](docs/installation.md)
- [Complete Workflow →](docs/workflow.md)
- [Troubleshooting →](docs/troubleshooting.md)
