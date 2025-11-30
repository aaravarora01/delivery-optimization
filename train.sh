#!/bin/bash
# Quick training script - modify paths and parameters as needed

# Set your data path here
JSON_ROOT="${JSON_ROOT:-cs230_data}"

echo "Training Pointer Transformer with full compute..."
echo "Data root: $JSON_ROOT"
echo ""

# Basic full-dataset training with optimizations
python src/train_pointer.py \
  --json_root "$JSON_ROOT" \
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

# Alternative: Use config file
# python src/train_pointer.py \
#   --json_root "$JSON_ROOT" \
#   --split train \
#   --config configs/pointer_large.yaml \
#   --mixed_precision \
#   --compile

# Alternative: With GNN
# python src/train_pointer.py \
#   --json_root "$JSON_ROOT" \
#   --split train \
#   --use_gnn \
#   --epochs 20 \
#   --batch_size 32 \
#   --mixed_precision

echo ""
echo "Training complete! Model saved to outputs/pointer_transformer.pt"

