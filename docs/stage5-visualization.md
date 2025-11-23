# Stage V: Result Analysis and Visualization

## Objective

Analyze experimental results and generate visualization plots comparing FincGAN with baseline methods.

This final stage helps you:
1. View and analyze results from all methods
2. Generate comparison plots
3. Process statistics across different ratios
4. Create publication-ready figures

## Prerequisites

- **Stage IV completed**: Result files must exist in `results/` directory
  - `music_hgt_model_gan.txt`
  - `music_hgt_model_smote.txt`
  - `music_hgt_model_noise.txt`
  - etc.

---

## 5.1 View Experimental Results

### Using Python

```python
import pandas as pd

# Load FincGAN results
fincgan_df = pd.read_csv("results/music_hgt_model_gan.txt")
print("FincGAN Results:")
print(fincgan_df)

# Load baseline results
smote_df = pd.read_csv("results/music_hgt_model_smote.txt")
print("\nSMOTE Results:")
print(smote_df)

noise_df = pd.read_csv("results/music_hgt_model_noise.txt")
print("\nNoise Results:")
print(noise_df)
```

### Using Command Line

```bash
# View all result files
ls -lh results/

# Quick view of FincGAN results
cat results/music_hgt_model_gan.txt

# View first few lines
head results/music_hgt_model_gan.txt

# Compare multiple results side by side
paste results/music_hgt_model_gan.txt results/music_hgt_model_smote.txt
```

### Result Format

Each result file contains:
```csv
ratio,seed,AUC-PRC,AUC-ROC,F-score,precision,recall,ACC
0.1007,10,0.4515,0.8851,0.4545,0.4630,0.4464,0.9147
0.1007,11,0.4521,0.8847,0.4551,0.4635,0.4470,0.9145
```

---

## 5.2 Automatic Plot Generation

### Method 1: Automatic Processing (Recommended)

The easiest way to generate plots is using the automatic function:

```python
from visualize import auto_plot_figure_3

# Automatically search all result files and plot
fig_path = auto_plot_figure_3(result_dir='results/', save_fig=True)
print(f"Figure saved to: {fig_path}")
```

#### Command Line Usage

```bash
# Generate visualization directly
python3 -c "from visualize import auto_plot_figure_3; auto_plot_figure_3(result_dir='results/', save_fig=True)"
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `result_dir` | Directory containing result files | `'results/'` |
| `save_fig` | Whether to save figure | `True` |
| `fig_dir` | Directory to save figures | `'figures/'` |

#### Output

- **Figure file**: `figures/figure_3.png`
- Comparison plot showing all methods across different metrics

---

### Method 2: Manual Processing

For more control over the visualization:

```python
from visualize import plot_figure_3

# Requires manual data processing first (see section 5.3)
# Then generate plot
fig_path = plot_figure_3(save_fig=True)
```

**Note**: This method requires you to process the data manually (see section 5.3 below).

---

## 5.3 Result Statistics Processing

### Calculate Average Results Across Seeds

```python
import numpy as np
import pandas as pd

# Load results
fincgan_df = pd.read_csv("results/music_hgt_model_gan.txt")

# Calculate average values for different ratios
gan_result = []
for ratio in np.arange(0.1, 1.3, 0.1):
    ratio = np.round_(ratio, 1)
    temp_df = fincgan_df[fincgan_df['ratio'] == ratio]
    if len(temp_df) > 0:
        gan_result.append(temp_df.mean().tolist()[-6:])

# Create statistics DataFrame
gan_df = pd.DataFrame(gan_result)
gan_df.columns = ["AUC-PRC(mean)", "AUC-ROC(mean)", "F-score(mean)",
                  "Precision(mean)", "Recall(mean)", "Acc(mean)"]

# Save summary
gan_df.to_csv("gan_summary.csv", index=False)
print(gan_df)
```

### Compare All Methods

```python
import pandas as pd
import numpy as np

methods = ['gan', 'smote', 'noise', 'oversampling', 'reweight', 'graphsmote']
summaries = {}

for method in methods:
    try:
        df = pd.read_csv(f"results/music_hgt_model_{method}.txt")
        # Calculate mean across seeds for each ratio
        summary = df.groupby('ratio').mean().reset_index()
        summaries[method] = summary
        print(f"\n{method.upper()} Summary:")
        print(summary)
    except FileNotFoundError:
        print(f"Results for {method} not found")

# Combine all summaries
all_results = pd.concat(
    [df.assign(method=method) for method, df in summaries.items()],
    ignore_index=True
)
all_results.to_csv("all_methods_summary.csv", index=False)
```

### Find Best Performing Method

```python
import pandas as pd

# Load all results summary
all_results = pd.read_csv("all_methods_summary.csv")

# For each metric, find the best method
metrics = ['AUC-PRC', 'AUC-ROC', 'F-score', 'precision', 'recall', 'ACC']

print("Best performing methods:")
for metric in metrics:
    best = all_results.loc[all_results[metric].idxmax()]
    print(f"{metric}: {best['method']} (ratio={best['ratio']:.2f}, value={best[metric]:.4f})")
```

---

## 5.4 Custom Visualizations

### Plot Single Metric Comparison

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load summaries
gan_df = pd.read_csv("gan_summary.csv")
smote_df = pd.read_csv("smote_summary.csv")

# Plot AUC-PRC comparison
plt.figure(figsize=(10, 6))
plt.plot(gan_df.index, gan_df['AUC-PRC(mean)'], marker='o', label='FincGAN')
plt.plot(smote_df.index, smote_df['AUC-PRC(mean)'], marker='s', label='SMOTE')
plt.xlabel('Ratio Index')
plt.ylabel('AUC-PRC')
plt.title('AUC-PRC Comparison: FincGAN vs SMOTE')
plt.legend()
plt.grid(True)
plt.savefig('figures/auc_prc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Plot Performance Across Ratios

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/music_hgt_model_gan.txt")

# Group by ratio and calculate mean
ratio_perf = df.groupby('ratio').mean()

# Plot all metrics
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics = ['AUC-PRC', 'AUC-ROC', 'F-score', 'precision', 'recall', 'ACC']

for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
    ax.plot(ratio_perf.index, ratio_perf[metric], marker='o')
    ax.set_xlabel('Synthetic Node Ratio')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Ratio')
    ax.grid(True)

plt.tight_layout()
plt.savefig('figures/metrics_vs_ratio.png', dpi=300, bbox_inches='tight')
plt.show()
```
