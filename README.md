# FincGAN: A Gan Framework of Imbalanced Node Classification on Heterogeneous Graph Neural Network

FincGAN is a fraud detection framework combining Graph Neural Networks (GNN) and Generative Adversarial Networks (GAN) to address class imbalance in fraud detection.

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
- [Project Structure](docs/project-structure.md)

---

## Project Structure

```
fincgan/
├── fincgan/         # Python source code (train.py, models, generators)
├── scripts/         # Automation scripts (run_fincgan_*.sh, setup_env*.sh)
├── docs/            # Documentation
├── graph/           # Input datasets (amazon.bin, yelp.bin)
└── Output directories (generated during training):
    ├── embed/       # Node embeddings (amazon/, yelp/)
    ├── generator/   # GAN models (amazon/, yelp/)
    ├── results/     # Training metrics (amazon/, yelp/)
    └── figures/     # Visualizations (amazon/, yelp/)
```

**[View complete project structure →](docs/project-structure.md)**

---

## Quick Links

- [Installation →](docs/installation.md)
- [Complete Workflow →](docs/workflow.md)
- [Project Structure →](docs/project-structure.md)
