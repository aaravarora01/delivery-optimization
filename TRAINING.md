# Training Guide - Pointer Transformer

This guide provides commands for training the Pointer Transformer model with full compute utilization.

## Quick Start

### Basic Training (Full Dataset, Default Settings)

```bash
python src/train_pointer.py \
  --json_root cs230_data \
  --split train \
  --epochs 20 \
  --batch_size 32 \
  --mixed_precision
```

### High-Performance Training (Larger Model, All Optimizations)

```bash
python src/train_pointer.py \
  --json_root cs230_data \
  --split train \
  --epochs 20 \
  --batch_size 32 \
  --d_model 256 \
  --nhead 8 \
  --d_ff 512 \
  --nlayers_enc 4 \
  --nlayers_dec 4 \
  --max_zone 80 \
  --lr 3e-4 \
  --mixed_precision \
  --compile \
  --num_workers 8
```

### With GNN (Graph Neural Network)

```bash
python src/train_pointer.py \
  --json_root cs230_data \
  --split train \
  --use_gnn \
  --epochs 10 \
  --batch_size 32 \
  --d_model 256 \
  --mixed_precision \
  --save_plots
```

## Key Optimizations Made

1. **Full Dataset**: Removed `max_routes` limitation - uses all training data by default
2. **Larger Zones**: Increased `max_zone` from 40 to 80 (configurable)
3. **Larger Model**: Default model size increased (d_model: 256, more layers)
4. **Full Learning Rate**: Removed 0.1 multiplier on learning rate
5. **Mixed Precision**: Optional FP16 training for 2x speedup on modern GPUs
6. **Model Compilation**: Optional torch.compile for additional speedup (PyTorch 2.0+)
7. **Better Scheduling**: Cosine annealing learning rate schedule
8. **Efficient Data Loading**: Multi-worker data loading with pin_memory

## Arguments

### Required
- `--json_root`: Root directory containing dataset JSON files (e.g., `cs230_data`)

### Training
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Effective batch size via gradient accumulation (default: 32)
- `--lr`: Learning rate (default: 3e-4, now uses full value)
- `--weight_decay`: Weight decay for AdamW (default: 1e-3)

### Model Architecture
- `--d_model`: Model dimension (default: 256, was 128)
- `--nhead`: Number of attention heads (default: 8)
- `--d_ff`: Feedforward dimension (default: 512)
- `--nlayers_enc`: Number of encoder layers (default: 4, was 2)
- `--nlayers_dec`: Number of decoder layers (default: 4, was 2)
- `--dropout`: Dropout rate (default: 0.1)
- `--use_gnn`: Use GraphSAGE GNN before transformer

### Data
- `--max_zone`: Maximum zone size for route splitting (default: 80, was 40)
- `--max_routes`: Limit number of routes for testing (default: None = use all)
- `--num_workers`: DataLoader workers (default: 8)

### Compute Optimizations
- `--mixed_precision`: Enable mixed precision (FP16) training
- `--compile`: Compile model with torch.compile (PyTorch 2.0+)
- `--device`: Device to use (default: cuda if available)

### Other
- `--config`: Path to YAML config file (optional)
- `--split`: Dataset split (default: train)

## Output

The training process creates a timestamped run directory with comprehensive outputs:

```
outputs/run_YYYYMMDD_HHMMSS/
├── final_model.pt           # Final trained model
├── best_model.pt            # Best model based on validation Kendall Tau
├── metrics.json             # All training and validation metrics
├── training_metrics.csv     # Per-epoch metrics in CSV format
├── training_loss.png        # Training loss curve
├── learning_rate.png        # Learning rate schedule
├── validation_metrics.png   # Validation metrics over epochs
└── combined_training_curves.png  # Combined loss and LR plot
```

The model files include:
- Model state dict
- GNN state dict (if used)
- Model configuration
- Training arguments
- Training metrics

## Metrics and Evaluation

### Training Metrics Tracked

1. **Training Loss**: Cross-entropy loss during training
2. **Learning Rate**: Current learning rate at each epoch
3. **Validation Metrics** (evaluated periodically):
   - **Kendall Tau**: Rank correlation between predicted and true sequence (-1 to 1, higher is better)
   - **Sequence Accuracy**: Exact sequence match percentage (0-1)
   - **Distance Ratio**: Predicted tour distance / True tour distance (closer to 1.0 is better)
   - **Position Accuracy @k=1**: Percentage of stops within 1 position of correct location
   - **Position Accuracy @k=3**: Percentage of stops within 3 positions
   - **Position Accuracy @k=5**: Percentage of stops within 5 positions
   - **Mean/Median Absolute Position Error**: Average position error

### Metrics Files

- **metrics.json**: Complete metrics dictionary with:
  - Per-epoch training metrics
  - Final validation metrics (aggregated statistics)
  - Sample per-zone metrics
  - Model and training configuration

- **training_metrics.csv**: Tabular format for easy analysis in spreadsheet tools

### Visualizations

Enable with `--save_plots` flag. Generated plots include:
1. **Training Loss**: Shows loss convergence over epochs
2. **Learning Rate Schedule**: Visualizes LR schedule
3. **Validation Metrics**: 4-panel plot showing key validation metrics
4. **Combined Training Curves**: Loss and LR on same plot with dual y-axes

## Validation

The training script automatically:
- Splits data into train/validation sets (default 90/10)
- Evaluates validation metrics every N epochs (default: every epoch)
- Saves best model based on validation Kendall Tau
- Provides comprehensive final evaluation report

## Examples

### Testing with Limited Routes

```bash
python src/train_pointer.py \
  --json_root cs230_data \
  --split train \
  --max_routes 100 \
  --epochs 5 \
  --batch_size 16
```

### Maximum Performance Training

```bash
python src/train_pointer.py \
  --json_root cs230_data \
  --split train \
  --epochs 30 \
  --batch_size 64 \
  --d_model 512 \
  --nhead 16 \
  --d_ff 1024 \
  --nlayers_enc 6 \
  --nlayers_dec 6 \
  --max_zone 100 \
  --mixed_precision \
  --compile \
  --num_workers 16 \
  --use_gnn \
  --save_plots
```

### Training with Metrics and Visualizations

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

### CPU Training (slower, but works)

```bash
python src/train_pointer.py \
  --json_root cs230_data \
  --split train \
  --device cpu \
  --batch_size 8 \
  --num_workers 4
```

## Monitoring Training

Training output includes:
- Epoch progress with average loss
- Learning rate schedule
- Number of zones processed
- Periodic step-level loss updates (every 50 gradient steps)

## Notes

- Gradient accumulation is used to achieve the effective batch size while processing variable-length zones
- Mixed precision requires CUDA-capable GPU
- Model compilation requires PyTorch 2.0+
- All routes are used by default - remove `--max_routes` argument for full dataset
- Validation metrics are computed on a held-out subset (10% by default)
- Best model is automatically saved based on validation Kendall Tau
- Use `--save_plots` to generate visualization plots for your report
- All metrics are saved to JSON and CSV for easy analysis

## Interpreting Metrics

### For Project Report:

1. **Kendall Tau** (τ): 
   - Measures rank correlation between predicted and true sequence
   - Range: -1 (perfect inversion) to +1 (perfect agreement)
   - Good performance: τ > 0.7, Excellent: τ > 0.9

2. **Sequence Accuracy**:
   - Percentage of routes with exactly correct sequence
   - Useful for understanding perfect predictions
   - Typically lower, focus on Kendall Tau for overall performance

3. **Distance Ratio**:
   - Predicted tour length / Optimal tour length
   - 1.0 = optimal, < 1.2 = excellent, < 1.5 = good

4. **Position Accuracy @k**:
   - Percentage of stops within k positions of correct location
   - Shows how "close" predictions are even if not perfect

5. **Training Loss**:
   - Should decrease smoothly over epochs
   - Use to diagnose overfitting or training issues

