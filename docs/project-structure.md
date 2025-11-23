# Project Structure

Complete directory structure of the FincGAN project:

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
│   ├── project-structure.md    # This file
│   ├── stage1-embedding.md
│   ├── stage2-node-generator.md
│   ├── stage3-edge-generator.md
│   ├── stage4-training.md
│   └── stage5-visualization.md
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

## Directory Descriptions

### Source Code (`fincgan/`)
Contains all Python modules for training and graph generation:
- **train.py**: Main training entry point for HGT model
- **hgt_model.py**: Implementation of Heterogeneous Graph Transformer
- **node_generator.py**: GAN-based synthetic node generation
- **edge_generator_uu.py**: User-User edge prediction
- **edge_generator_up.py**: User-Product edge prediction
- **graph_generator.py**: Combines nodes and edges into complete graphs
- **utils.py**: Shared utility functions and helpers
- **visualize.py**: Result visualization and plotting
- **logger.py**: Centralized logging system

### Scripts (`scripts/`)
Automation scripts for environment setup and training:
- **run_fincgan_amazon.sh**: Complete pipeline for Amazon dataset
- **run_fincgan_yelp.sh**: Complete pipeline for Yelp dataset
- **setup_env.sh**: CPU environment installation
- **setup_env_gpu.sh**: GPU environment installation
- **environment.yml**: Conda dependencies for CPU
- **environment-gpu.yml**: Conda dependencies for GPU

### Documentation (`docs/`)
Comprehensive guides and documentation:
- **installation.md**: Environment setup instructions
- **workflow.md**: Step-by-step training workflow
- **stage1-embedding.md**: Feature extraction guide
- **stage2-node-generator.md**: GAN training guide
- **stage3-edge-generator.md**: Edge generator guide
- **stage4-training.md**: Final training guide
- **stage5-visualization.md**: Result visualization guide

### Input Data
- **graph/**: Original datasets in DGL binary format
- **data/**: Additional input data (if needed)
- **images/**: Documentation images and figures

### Output Directories
All output directories are organized by dataset with subdirectories for `amazon/` and `yelp/`:

- **embed/**: Saved node embeddings from HGT
- **generator/**: Trained GAN models (G, D, edge generators)
- **graph_output/**: Generated synthetic graphs
- **results/**: Training metrics and evaluation results
- **figures/**: Visualization plots and figures
- **tsne/**: t-SNE visualization images
- **tmp/**: Temporary model checkpoints
- **logs/**: Training logs with timestamps

## File Naming Conventions

### Model Files
- `{dataset}_hgt_model_{setting}_ratio_{ratio}_seed_{seed}.pt` - HGT checkpoints
- `{dataset}_G.pt` - Generator model
- `{dataset}_D.pt` - Discriminator model
- `uu_generator.pt` - User-User edge generator
- `up_generator.pt` - User-Product edge generator

### Data Files
- `{dataset}_hgt_user_emb.pt` - User embeddings
- `{dataset}_hgt_product_emb.pt` - Product embeddings
- `{dataset}_gan_ratio_{ratio}_seed_{seed}.bin` - Generated graphs

### Result Files
- `{dataset}_hgt_model_{setting}.txt` - Training metrics
- `tsne*.jpg` - t-SNE visualizations
- `figure_3.png` - Performance comparison plot
