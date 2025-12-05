# CS230 Last Mile Routing

A machine learning solution for last-mile delivery route optimization using Pointer Transformer networks with optional Graph Neural Network (GNN) preprocessing.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Official data is stored under `cs230_data/`:

- **Training**: `cs230_data/almrrc2021-data-training/model_build_inputs/{route_data.json, package_data.json, actual_sequences.json}`
- **Evaluation**: `cs230_data/almrrc2021-data-evaluation/model_apply_inputs/{eval_route_data.json, eval_package_data.json, eval_travel_times.json}`

## Running Models

### Baseline (Nearest Neighbor)

```bash
python src/run_baseline_json.py --json_root cs230_data --split train
```

Outputs: `outputs/route_<RID>_baseline.csv`

### Learned Model (Pointer Transformer)

Train and evaluate on a specific route:

```bash
python src/run_learned.py \
  --json_root_train cs230_data --split_train train \
  --json_root_test cs230_data --split_test eval \
  --route_id <RID>
```

Outputs: `outputs/route_<RID>_learned.csv`

## Training

### Training Arguments

#### Required
- `--json_root`: Root directory containing dataset JSON files

#### Training Parameters
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Effective batch size via gradient accumulation (default: 32)
- `--lr`: Learning rate (default: 3e-4)
- `--weight_decay`: Weight decay for AdamW (default: 1e-3)

#### Model Architecture
- `--d_model`: Model dimension (default: 256)
- `--nhead`: Number of attention heads (default: 8)
- `--d_ff`: Feedforward dimension (default: 512)
- `--nlayers_enc`: Number of encoder layers (default: 4)
- `--nlayers_dec`: Number of decoder layers (default: 4)
- `--dropout`: Dropout rate (default: 0.1)
- `--use_gnn`: Use GraphSAGE GNN before transformer

#### Data
- `--max_zone`: Maximum zone size for route splitting (default: 80)
- `--max_routes`: Limit number of routes for testing (default: None = use all)
- `--num_workers`: DataLoader workers (default: 8)
- `--split`: Dataset split (default: train)

#### Compute Optimizations
- `--mixed_precision`: Enable mixed precision (FP16) training for 2x speedup
- `--compile`: Compile model with torch.compile (PyTorch 2.0+)
- `--device`: Device to use (default: cuda if available)

#### Other
- `--config`: Path to YAML config file (optional)
- `--save_plots`: Generate visualization plots

## Training Output

The training process creates a timestamped run directory:

```
outputs/run_YYYYMMDD_HHMMSS/
├── final_model.pt                    # Final trained model
├── best_model.pt                     # Best model based on validation Kendall Tau
├── metrics.json                      # All training and validation metrics
├── training_metrics.csv              # Per-epoch metrics in CSV format
├── training_loss.png                 # Training loss curve
├── learning_rate.png                 # Learning rate schedule
├── validation_metrics.png            # Validation metrics over epochs
└── combined_training_curves.png      # Combined loss and LR plot
```

## Metrics and Evaluation

### Metrics Tracked

**Training Metrics:**
- Training Loss: Cross-entropy loss during training
- Learning Rate: Current learning rate at each epoch

**Validation Metrics:**
- **Kendall Tau** (τ): Rank correlation between predicted and true sequence (-1 to 1, higher is better)
- **Sequence Accuracy**: Exact sequence match percentage (0-1)
- **Distance Ratio**: Predicted tour distance / True tour distance (closer to 1.0 is better)
- **Position Accuracy @k={1,3,5}**: Percentage of stops within k positions of correct location
- **Mean/Median Absolute Position Error**: Average position error


## Training Example

```bash
python src/train_pointer.py \
  --json_root cs230_data \
  --split train \
  --epochs 20 \
  --batch_size 32 \
  --mixed_precision
```

## Key Optimizations

1. **Full Dataset**: Uses all training data by default
2. **Larger Zones**: Increased max_zone from 40 to 80 for better context
3. **Larger Model**: Increased default model size (d_model: 256, more layers)
4. **Mixed Precision**: Optional FP16 training for 2x speedup on modern GPUs
5. **Model Compilation**: Optional torch.compile for additional speedup (PyTorch 2.0+)
6. **Cosine Annealing**: Learning rate schedule for better convergence
7. **Efficient Data Loading**: Multi-worker data loading with pin_memory

## Notes

- Gradient accumulation is used to achieve effective batch size while processing variable-length zones
- Mixed precision requires CUDA-capable GPU
- Model compilation requires PyTorch 2.0+
- Validation metrics are computed on a held-out subset (10% by default)
- Best model is automatically saved based on validation Kendall Tau
- Use `--save_plots` to generate visualization plots for reports
- All metrics are saved to JSON and CSV for easy analysis
