# Dataset Information

## Amazon Musical Instruments Review Dataset

This project uses the Amazon Musical Instruments review dataset for fraud detection.

### Dataset Statistics

- **User Nodes**: 7,017
- **Product Nodes**: 4,684
- **Total Nodes**: 11,701

### Edge Types

| Edge Type | Count | Description |
|-----------|-------|-------------|
| Product-Product (p-p) | 101,678 | Product similarity/co-purchase edges |
| Product-User (p-u) | 12,169 | Product to user review edges |
| User-Product (u-p) | 12,169 | User to product review edges |
| User-User (u-u) | 535,244 | User similarity edges |

**Total Edges**: 661,260

### Graph Structure

The dataset forms a heterogeneous graph with two node types (users and products) and multiple edge types representing different relationships:

```
Graph Structure:
├── Users (7,017)
│   ├── Benign users
│   └── Spam/Fraudulent users (minority class)
├── Products (4,684)
│   └── Musical instruments
└── Edges (661,260)
    ├── User-Product interactions
    ├── User-User relationships
    └── Product-Product relationships
```

### Data Characteristics

**Imbalanced Classification Problem:**
- The dataset exhibits severe class imbalance
- Fraudulent/spam users are the minority class
- This imbalance is the primary challenge FincGAN addresses

**Graph Properties:**
- Heterogeneous graph structure
- Multiple relation types
- Sparse edge connectivity for user-user relationships
- Dense edge connectivity for user-product relationships

### Data Location

The original graph data is stored in:
```
fincgan/graph/music_instrument_25.bin
```

This is a preprocessed DGL binary graph file containing:
- Node features
- Edge indices
- Node labels (benign/spam)
- Graph metadata

### Data Preprocessing

The dataset has been preprocessed to:
1. Extract user and product features
2. Construct heterogeneous graph structure
3. Label benign and fraudulent users
4. Create train/validation/test splits

### Usage in FincGAN

FincGAN uses this dataset to:
1. **Train feature extractors** (Stage I) on the heterogeneous graph
2. **Learn fraud patterns** from the imbalanced distribution
3. **Generate synthetic nodes** to balance the dataset
4. **Evaluate fraud detection performance** on the augmented graph

### Citation

If you use this dataset, please cite the original Amazon review dataset paper and the FincGAN paper.

---

**Related Documentation:**
- [Installation Guide](installation.md)
- [Stage I: Embedding](stage1-embedding.md)
- [Main README](../README.md)
