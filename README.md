# FincGAN: A Gan Framework of Imbalanced Node Classification on Heterogeneous Graph Neural Network

FincGAN is a fraud detection framework combining Graph Neural Networks (GNN) and Generative Adversarial Networks (GAN) to address class imbalance in fraud detection using the Amazon Musical Instruments review dataset.

---

## Quick Start

### Automated Pipeline (Recommended)

```bash
# Make script executable
chmod +x run_fincgan.sh

# Run complete pipeline
./run_fincgan.sh

# Quick test (5-10 minutes)
./run_fincgan.sh --n-epoch-emb 5 --n-epoch-gan 2 --n-epoch-uu 10 --n-epoch-up 5 --n-epoch-train 10

# View all options
./run_fincgan.sh --help
```

**For manual step-by-step execution, see [Complete Workflow](docs/workflow.md).**

---

## Installation

```bash
# Run setup script (one command to install everything)
chmod +x setup_env.sh
./setup_env.sh
```

**For manual setup and different CUDA versions, see [Installation Guide](docs/installation.md).**

---

## FincGAN Pipeline

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

**Dataset**: 7,017 users, 4,684 products, 661,260 edges (highly imbalanced)

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
- [Evaluation Metrics](docs/metrics.md)
- [Troubleshooting](docs/troubleshooting.md)

---

## Project Structure

```
fincgan/
├── README.md                 # This file
├── run_fincgan.sh           # Automated pipeline script
├── docs/                    # Detailed documentation
├── train.py                 # Main training script
├── node_generator.py        # GAN training
├── edge_generator_uu.py     # User-User edge generator
├── edge_generator_up.py     # User-Product edge generator
├── graph_generator.py       # Graph generation
├── visualize.py             # Visualization
├── hgt_model.py            # HGT model
├── utils.py                # Utilities
└── graph/                  # Original dataset
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
