# Metrics and Evaluation Guide

This guide explains the accuracy metrics and visualizations available for your project report.

## Quick Start: Training with Metrics

To enable all metrics and visualizations for your report:

```bash
python src/train_pointer.py \
  --json_root cs230_data \
  --split train \
  --epochs 20 \
  --batch_size 32 \
  --mixed_precision \
  --save_plots \
  --val_eval_freq 1 \
  --val_num_zones 100
```

## Available Metrics

### 1. Training Metrics

- **Training Loss**: Cross-entropy loss during training
  - Should decrease over epochs
  - Lower is better
  - Used to monitor training convergence

- **Learning Rate**: Current learning rate at each epoch
  - Follows cosine annealing schedule
  - Starts at `--lr` and decreases to `lr * 0.01`

### 2. Validation Metrics

Computed on a held-out validation set (10% of data by default):

#### Kendall Tau (τ)
- **Range**: -1 to +1
- **Meaning**: Rank correlation between predicted and true sequence
- **Interpretation**:
  - +1: Perfect agreement with true sequence
  - 0: No correlation
  - -1: Perfect inversion
- **Good Performance**: τ > 0.7
- **Excellent**: τ > 0.9

#### Sequence Accuracy
- **Range**: 0 to 1 (percentage)
- **Meaning**: Exact sequence match percentage
- **Interpretation**: Fraction of zones where predicted sequence exactly matches true sequence
- **Note**: Usually lower than other metrics; focus on Kendall Tau for overall performance

#### Distance Ratio
- **Formula**: Predicted Tour Distance / True Tour Distance
- **Range**: Typically 0.8 to 2.0+
- **Interpretation**:
  - 1.0: Optimal (predicted route same length as true route)
  - < 1.2: Excellent
  - < 1.5: Good
  - > 2.0: Needs improvement
- **Note**: Lower is generally better (but not if under 1.0 it might indicate shorter path found)

#### Position Accuracy @k
- **Meaning**: Percentage of stops predicted within k positions of correct location
- **Available**: @k=1, @k=3, @k=5
- **Interpretation**: 
  - @k=1: Very strict (exact or adjacent position)
  - @k=3: Moderate (within 3 positions)
  - @k=5: Lenient (within 5 positions)
- **Higher is better**

#### Mean/Median Absolute Position Error
- **Range**: 0 to N (number of stops)
- **Meaning**: Average difference between predicted and true position
- **Lower is better**

## Output Files

### Directory Structure

After training, you'll find a timestamped directory:

```
outputs/run_20240101_120000/
├── final_model.pt              # Final trained model
├── best_model.pt               # Best model (highest validation tau)
├── metrics.json                # All metrics in JSON format
├── training_metrics.csv        # Per-epoch metrics (Excel-friendly)
├── training_loss.png           # Loss curve plot
├── learning_rate.png           # LR schedule plot
├── validation_metrics.png      # 4-panel validation metrics
└── combined_training_curves.png # Loss + LR combined
```

### metrics.json Structure

```json
{
  "training_metrics": {
    "epoch": [1, 2, 3, ...],
    "train_loss": [2.5, 2.1, 1.8, ...],
    "learning_rate": [0.0003, 0.00028, ...],
    "val_kendall_tau_mean": [0.65, 0.72, 0.78, ...],
    "val_sequence_accuracy_mean": [0.12, 0.15, ...],
    ...
  },
  "final_validation_metrics": {
    "kendall_tau_mean": 0.82,
    "kendall_tau_std": 0.15,
    "sequence_accuracy_mean": 0.23,
    "distance_ratio_mean": 1.18,
    ...
  },
  "per_zone_metrics": [...],
  "config": {...},
  "model_config": {...}
}
```

### training_metrics.csv

CSV format with columns:
- epoch
- train_loss
- learning_rate
- val_kendall_tau_mean
- val_sequence_accuracy_mean
- val_distance_ratio_mean
- val_position_acc_k1_mean

Easy to import into Excel/Google Sheets for additional analysis.

## Visualizations

All plots are saved as high-resolution PNG files (300 DPI) suitable for reports.

### 1. Training Loss (`training_loss.png`)
- Line plot showing loss over epochs
- Use to show training convergence
- Should show smooth decreasing trend

### 2. Learning Rate Schedule (`learning_rate.png`)
- Shows LR decay over training
- Demonstrates learning rate annealing

### 3. Validation Metrics (`validation_metrics.png`)
- **4-panel plot** showing:
  1. Kendall Tau over epochs
  2. Sequence Accuracy over epochs
  3. Distance Ratio over epochs (with optimal line at 1.0)
  4. Position Accuracy @k=1 over epochs
- Key visualization for showing model improvement

### 4. Combined Training Curves (`combined_training_curves.png`)
- Dual y-axis plot
- Shows loss and learning rate together
- Useful for correlating LR changes with loss

## Using Metrics in Your Report

### Recommended Metrics to Report

1. **Primary Metric**: Validation Kendall Tau (final and best)
2. **Secondary Metrics**: 
   - Distance Ratio (shows route quality)
   - Position Accuracy @k=3 (shows positional accuracy)
3. **Training Metrics**:
   - Training loss convergence
   - Learning rate schedule effectiveness

### Example Report Section

```markdown
## Results

The model was trained for 20 epochs on [X] routes. 
Validation metrics were computed on a held-out set of [Y] routes.

### Performance Metrics

- **Kendall Tau**: 0.82 ± 0.15 (mean ± std)
  - This indicates strong rank correlation between predicted and true sequences
- **Distance Ratio**: 1.18 ± 0.12
  - Predicted routes are on average 18% longer than optimal
- **Position Accuracy @k=3**: 0.67
  - 67% of stops are predicted within 3 positions of correct location
- **Sequence Accuracy**: 0.23
  - 23% of zones have exactly correct sequence predictions

### Training Convergence

[Include training_loss.png plot]
The training loss decreases smoothly from 2.5 to 0.8, indicating 
successful convergence without overfitting.

[Include validation_metrics.png plot]
Validation metrics show consistent improvement over training, with 
Kendall Tau increasing from 0.65 to 0.82.
```

## Command-Line Options for Metrics

```bash
--val_split 0.1              # Fraction of data for validation (default: 0.1)
--val_eval_freq 1            # Evaluate every N epochs (default: 1)
--val_num_zones 100          # Number of zones to evaluate (default: 50)
--save_plots                 # Generate visualization plots
--output_dir ./outputs       # Output directory
```

## Tips for Project Report

1. **Use Best Model**: The `best_model.pt` is saved based on highest validation Kendall Tau
2. **Include Multiple Metrics**: Show both sequence accuracy (Kendall Tau) and route quality (Distance Ratio)
3. **Visualizations**: Include 2-3 key plots (loss, validation metrics)
4. **Statistical Summary**: Report mean ± std for validation metrics
5. **Comparison**: Compare metrics at different epochs to show improvement
6. **Baseline Comparison**: Compare against baseline methods if available

## Troubleshooting

### No validation metrics appearing
- Check that validation set is being created (look for "Split: X training routes, Y validation routes")
- Ensure zones are being evaluated (check for "Evaluating on validation set..." messages)

### Metrics showing NaN
- Some zones may be too small or have issues
- The aggregation function automatically handles NaNs
- Check console for warnings about invalid zones

### Plots not generating
- Ensure matplotlib is installed: `pip install matplotlib`
- Use `--save_plots` flag
- Check for matplotlib import errors in console

