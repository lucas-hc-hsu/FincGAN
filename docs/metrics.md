# Evaluation Metrics

This guide explains the evaluation metrics used in FincGAN for fraud detection performance assessment.

## Overview

FincGAN uses **6 standard classification metrics** to evaluate fraud detection performance:

1. **AUC-PRC** - Area Under the Precision-Recall Curve
2. **AUC-ROC** - Area Under the ROC Curve
3. **F1-Score** - Harmonic mean of precision and recall
4. **Precision** - Correctness of positive predictions
5. **Recall** - Coverage of actual positives
6. **Accuracy** - Overall correctness

---

## Metrics Explained

### 1. AUC-PRC (Area Under Precision-Recall Curve)

**Range**: 0.0 to 1.0 (higher is better)

**Typical FincGAN range**: 0.40 - 0.50

**What it measures**:
- Performance across different classification thresholds
- Particularly useful for **imbalanced datasets** (like fraud detection)
- Emphasizes performance on the minority class (fraudsters)

**Why it matters**:
- More informative than AUC-ROC for imbalanced data
- Shows trade-off between precision and recall
- Better reflects real-world fraud detection scenarios

**Formula**:
```
AUC-PRC = ∫ Precision(Recall) dRecall
```

**Interpretation**:
- **0.40-0.50**: Good performance on imbalanced fraud data
- **0.50-0.70**: Very good performance
- **>0.70**: Excellent performance
- **<0.40**: Poor performance, needs improvement

**When to use**: Primary metric for imbalanced classification problems

---

### 2. AUC-ROC (Area Under ROC Curve)

**Range**: 0.0 to 1.0 (higher is better)

**Typical FincGAN range**: 0.85 - 0.90

**What it measures**:
- Trade-off between True Positive Rate and False Positive Rate
- Model's ability to distinguish between classes
- Performance across all classification thresholds

**Why it matters**:
- Widely used and understood metric
- Good for comparing different models
- Threshold-independent performance measure

**Formula**:
```
AUC-ROC = ∫ TPR(FPR) dFPR
where:
  TPR = True Positive Rate (Recall)
  FPR = False Positive Rate
```

**Interpretation**:
- **0.90-1.00**: Excellent discrimination
- **0.80-0.90**: Good discrimination (typical for FincGAN)
- **0.70-0.80**: Fair discrimination
- **0.50-0.70**: Poor discrimination
- **0.50**: Random guessing
- **<0.50**: Worse than random (model is inverted)

**Note**: Can be misleading for highly imbalanced datasets (use AUC-PRC instead)

---

### 3. F1-Score

**Range**: 0.0 to 1.0 (higher is better)

**Typical FincGAN range**: 0.40 - 0.50

**What it measures**:
- Harmonic mean of precision and recall
- Balance between precision and recall
- Single metric combining both aspects

**Formula**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why it matters**:
- Balances false positives and false negatives
- More informative than accuracy for imbalanced data
- Good for comparing different approaches

**Interpretation**:
- **0.40-0.50**: Good performance on fraud data
- **0.50-0.70**: Very good performance
- **>0.70**: Excellent performance
- **<0.40**: Poor performance

**Variants**:
- **F2-Score**: Weighs recall higher (2× recall importance)
- **F0.5-Score**: Weighs precision higher (2× precision importance)

---

### 4. Precision

**Range**: 0.0 to 1.0 (higher is better)

**Typical FincGAN range**: 0.50 - 0.60

**What it measures**:
- Of all predicted fraudsters, how many are actually fraudsters?
- Correctness of positive predictions
- Also called Positive Predictive Value (PPV)

**Formula**:
```
Precision = TP / (TP + FP)

where:
  TP = True Positives (correctly identified fraudsters)
  FP = False Positives (benign users incorrectly flagged)
```

**Why it matters**:
- **High precision**: Few false alarms, investigations are productive
- **Low precision**: Many false alarms, waste of investigator time
- Critical when false positives are costly

**Interpretation**:
- **0.90-1.00**: Very few false positives
- **0.70-0.90**: Acceptable false positive rate
- **0.50-0.70**: Moderate false positive rate (typical for FincGAN)
- **<0.50**: High false positive rate, many false alarms

**Trade-off**: Increasing precision often decreases recall

---

### 5. Recall

**Range**: 0.0 to 1.0 (higher is better)

**Typical FincGAN range**: 0.35 - 0.45

**What it measures**:
- Of all actual fraudsters, how many did we catch?
- Coverage of actual positives
- Also called Sensitivity or True Positive Rate (TPR)

**Formula**:
```
Recall = TP / (TP + FN)

where:
  TP = True Positives (correctly identified fraudsters)
  FN = False Negatives (fraudsters we missed)
```

**Why it matters**:
- **High recall**: Catching most fraudsters
- **Low recall**: Missing many fraudsters
- Critical when false negatives are costly

**Interpretation**:
- **0.90-1.00**: Catching almost all fraudsters
- **0.70-0.90**: Catching most fraudsters
- **0.50-0.70**: Catching half of fraudsters
- **0.35-0.45**: Missing many fraudsters (typical for FincGAN due to class imbalance)
- **<0.35**: Missing most fraudsters

**Trade-off**: Increasing recall often decreases precision

---

### 6. Accuracy

**Range**: 0.0 to 1.0 (higher is better)

**Typical FincGAN range**: 0.92 - 0.93

**What it measures**:
- Overall correctness across all predictions
- Simple ratio of correct predictions

**Formula**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

where:
  TP = True Positives
  TN = True Negatives
  FP = False Positives
  FN = False Negatives
```

**Why it matters**:
- Easy to understand
- Good for balanced datasets
- **Misleading for imbalanced datasets**

**Interpretation for FincGAN**:
- **0.92-0.93**: Seems high, but...
- With 10% fraud rate, a model that predicts "all benign" gets 90% accuracy!
- **Not a good metric for fraud detection** - use AUC-PRC instead

**Limitation**: Not recommended as primary metric for imbalanced fraud detection

---

## Metrics Summary Table

| Metric | Range | FincGAN Typical | Best For | Key Insight |
|--------|-------|-----------------|----------|-------------|
| **AUC-PRC** | 0-1 | 0.40-0.50 | Imbalanced data | Primary metric for fraud detection |
| **AUC-ROC** | 0-1 | 0.85-0.90 | Model comparison | Overall discrimination ability |
| **F1-Score** | 0-1 | 0.40-0.50 | Balanced view | Precision-recall balance |
| **Precision** | 0-1 | 0.50-0.60 | False positive cost | How many alerts are real? |
| **Recall** | 0-1 | 0.35-0.45 | False negative cost | How many frauds did we catch? |
| **Accuracy** | 0-1 | 0.92-0.93 | Balanced data | **Misleading for fraud detection** |

---

## Understanding Trade-offs

### Precision-Recall Trade-off

**High Precision, Low Recall**:
- Conservative model
- Few false alarms, but misses many fraudsters
- Good when investigation resources are limited

**Low Precision, High Recall**:
- Aggressive model
- Catches most fraudsters, but many false alarms
- Good when missing fraud is very costly

**Balanced (F1-Score)**:
- Middle ground
- Optimize F1-score for balanced approach

**Visualization**:
```
Precision
   ^
   |
1.0|     •
   |    / \
   |   /   \
   |  /     \
0.5| /       \  ← Operating point
   |/         \
   |___________•____> Recall
   0          0.5    1.0
```

---

## Metric Selection Guide

### Primary Metric: AUC-PRC

**Use when**:
- Dataset is imbalanced (fraud detection ✓)
- You care about minority class performance
- Need threshold-independent metric

**Why for FincGAN**:
- Fraud is rare (~10% of users)
- Emphasizes performance on fraudsters
- More informative than AUC-ROC or Accuracy

---

### Secondary Metrics

**F1-Score**:
- Quick summary metric
- Compare different methods
- Balance precision and recall

**Precision**:
- When false positives are costly
- Limited investigation resources
- Need to prioritize quality over quantity

**Recall**:
- When false negatives are very costly
- Must catch as many fraudsters as possible
- Can tolerate false alarms

**AUC-ROC**:
- Compare with other papers
- Overall model quality
- Supplement to AUC-PRC

**Accuracy**:
- Reporting to non-technical stakeholders
- Include with caveat about imbalance
- Not for model optimization

---

## Computing Metrics in Python

### Using scikit-learn

```python
from sklearn.metrics import (
    precision_recall_curve, auc,
    roc_auc_score, f1_score,
    precision_score, recall_score,
    accuracy_score
)
import numpy as np

# Assuming:
# y_true: true labels (0 = benign, 1 = fraud)
# y_pred: predicted probabilities for fraud class

# AUC-PRC
precision, recall, _ = precision_recall_curve(y_true, y_pred)
auc_prc = auc(recall, precision)

# AUC-ROC
auc_roc = roc_auc_score(y_true, y_pred)

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred_binary = (y_pred > 0.5).astype(int)

# F1-Score
f1 = f1_score(y_true, y_pred_binary)

# Precision
precision_val = precision_score(y_true, y_pred_binary)

# Recall
recall_val = recall_score(y_true, y_pred_binary)

# Accuracy
accuracy = accuracy_score(y_true, y_pred_binary)

print(f"AUC-PRC: {auc_prc:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"F1: {f1:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"Accuracy: {accuracy:.4f}")
```

---

## Interpreting FincGAN Results

### Example Result

```
AUC-PRC: 0.4515
AUC-ROC: 0.8851
F1: 0.4545
Precision: 0.4630
Recall: 0.4464
ACC: 0.9147
```

### Analysis

✅ **Good indicators**:
- **AUC-ROC: 0.8851**: Strong discrimination ability
- **AUC-PRC: 0.4515**: Good performance on imbalanced data
- **F1: 0.4545**: Balanced precision-recall trade-off

⚠️ **Areas for improvement**:
- **Precision: 0.4630**: ~54% false positive rate
  - About 1 in 2 alerts are false alarms
- **Recall: 0.4464**: Missing ~55% of fraudsters
  - Catching less than half of actual fraud

📊 **High Accuracy: 0.9147**:
- Looks impressive but misleading
- With 10% fraud rate, predicting all benign gives 90% accuracy
- Not a useful metric for evaluation

---

## Comparing Methods

### Example Comparison Table

| Method | AUC-PRC | AUC-ROC | F1 | Precision | Recall | Accuracy |
|--------|---------|---------|-----|-----------|--------|----------|
| **FincGAN** | **0.4515** | **0.8851** | **0.4545** | 0.4630 | **0.4464** | 0.9147 |
| SMOTE | 0.4203 | 0.8642 | 0.4201 | 0.4512 | 0.3924 | 0.9102 |
| GraphSMOTE | 0.4384 | 0.8734 | 0.4312 | 0.4589 | 0.4063 | 0.9121 |
| Noise | 0.3987 | 0.8512 | 0.3945 | 0.4234 | 0.3698 | 0.9045 |
| Oversampling | 0.3845 | 0.8423 | 0.3812 | 0.4123 | 0.3542 | 0.8998 |

**Conclusion**: FincGAN outperforms all baselines on **AUC-PRC** (primary metric)

---

## Visualization

### Precision-Recall Curve

```python
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Plot PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
auc_prc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'FincGAN (AUC={auc_prc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc_roc = roc_auc_score(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'FincGAN (AUC={auc_roc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
```

---

## Best Practices

### For Fraud Detection

1. **Primary metric**: AUC-PRC
2. **Secondary metrics**: F1-Score, AUC-ROC
3. **Ignore**: Accuracy (misleading)
4. **Report**: Precision and Recall separately
5. **Compare**: Against baselines on AUC-PRC

### For Reporting

1. **Technical audience**: Report all 6 metrics
2. **Non-technical**: Focus on F1-Score and explain trade-offs
3. **Papers**: Use AUC-PRC and AUC-ROC
4. **Production**: Monitor precision (false alarm rate)

### For Optimization

1. **Optimize** for AUC-PRC (not accuracy!)
2. **Validate** on held-out test set
3. **Use multiple seeds** and report mean ± std
4. **Plot** PR and ROC curves for visualization

---

## References

For more information on evaluation metrics:

- Scikit-learn documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Precision-Recall curve interpretation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
- Imbalanced learning metrics: https://imbalanced-learn.org/stable/

---

**Related Documentation:**
- [Stage V: Visualization](stage5-visualization.md)
- [Complete Workflow](workflow.md)
- [Main README](../README.md)
