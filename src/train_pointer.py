# src/train_pointer.py

import argparse, math, random, json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.cluster import KMeans
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_amzl_dataset
from models.pointer_transformer import PointerTransformer, PTConfig
from models.gnn_sage import GraphSAGE
from utils.knn_graph import knn_adj
from utils.features import node_features, edge_bias_features
from utils.metrics import evaluate_zone_predictions, aggregate_metrics

# Try importing matplotlib for visualizations
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")

def build_zones(df_route, max_zone=80, seed=0):
    coords = df_route[['lat','lon']].to_numpy()
    n = len(coords)
    
    if n <= max_zone:
        return [df_route.copy()]
    
    k = math.ceil(n / max_zone)
    km = KMeans(n_clusters=k, random_state=seed, n_init='auto')
    z = km.fit_predict(coords)
    
    zones = []
    for c in range(k):
        zones.append(df_route[z==c].copy().reset_index(drop=True))
    
    return zones

def collate_zone(zone_df):
    coords = torch.tensor(zone_df[['lat','lon']].to_numpy(), dtype=torch.float32).unsqueeze(0)  # (1,N,2)
    
    # target index by true seq
    sid_to_pos = {sid:i for i,sid in enumerate(zone_df['stop_id'].tolist())}
    true_order = [sid_to_pos[sid] for sid in zone_df.sort_values('seq')['stop_id'].tolist()]
    target_idx = torch.tensor(true_order, dtype=torch.long).unsqueeze(0)  # (1,N)
    
    return coords, target_idx

def create_visualizations(training_metrics, final_metrics, output_dir, val_eval_freq=1):
    """Create visualization plots for training metrics."""
    if not HAS_MATPLOTLIB:
        return
    
    output_dir = Path(output_dir)
    
    epochs = training_metrics['epoch']
    
    # Plot 1: Training Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, training_metrics['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Learning Rate Schedule
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, training_metrics['learning_rate'], 'g-', linewidth=2, label='Learning Rate')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Validation Metrics
    if training_metrics['val_kendall_tau_mean']:
        val_epochs = epochs[::val_eval_freq]
        if len(val_epochs) > len(training_metrics['val_kendall_tau_mean']):
            val_epochs = val_epochs[:len(training_metrics['val_kendall_tau_mean'])]
        elif len(val_epochs) < len(training_metrics['val_kendall_tau_mean']):
            val_epochs = epochs[:len(training_metrics['val_kendall_tau_mean'])]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Kendall Tau
        axes[0, 0].plot(val_epochs, training_metrics['val_kendall_tau_mean'], 'r-', linewidth=2, marker='o')
        axes[0, 0].set_xlabel('Epoch', fontsize=10)
        axes[0, 0].set_ylabel('Kendall Tau', fontsize=10)
        axes[0, 0].set_title('Validation Kendall Tau', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sequence Accuracy
        axes[0, 1].plot(val_epochs, training_metrics['val_sequence_accuracy_mean'], 'm-', linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Epoch', fontsize=10)
        axes[0, 1].set_ylabel('Sequence Accuracy', fontsize=10)
        axes[0, 1].set_title('Validation Sequence Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distance Ratio
        axes[1, 0].plot(val_epochs, training_metrics['val_distance_ratio_mean'], 'c-', linewidth=2, marker='^')
        axes[1, 0].set_xlabel('Epoch', fontsize=10)
        axes[1, 0].set_ylabel('Distance Ratio', fontsize=10)
        axes[1, 0].set_title('Validation Distance Ratio (Pred/True)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        
        # Position Accuracy @k=1
        axes[1, 1].plot(val_epochs, training_metrics['val_position_acc_k1_mean'], 'orange', linewidth=2, marker='d')
        axes[1, 1].set_xlabel('Epoch', fontsize=10)
        axes[1, 1].set_ylabel('Position Accuracy @k=1', fontsize=10)
        axes[1, 1].set_title('Validation Position Accuracy', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Combined Training Curves
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(epochs, training_metrics['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, training_metrics['learning_rate'], 'g--', linewidth=2, label='Learning Rate')
    ax2.set_ylabel('Learning Rate', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    ax1.set_title('Training Loss and Learning Rate', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_dir / 'combined_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated visualization plots")

class RouteZoneDataset(Dataset):
    def __init__(self, df, routes, max_zone=80, min_stops=3, max_zones=None):
        self.df = df
        self.routes = routes
        self.max_zone = max_zone
        self.min_stops = min_stops
        self.zones = []
        
        print(f"Building zones from {len(routes)} routes (max_zone={max_zone})...")
        for rid in routes:
            r = df[df.route_id==rid].copy().reset_index(drop=True)
            route_zones = build_zones(r, max_zone=max_zone)
            for z in route_zones:
                if len(z) >= min_stops:
                    self.zones.append(z)
                    
                    # Stop if we've reached the limit
                    if max_zones is not None and len(self.zones) >= max_zones:
                        break
            # Break outer loop if limit reached
            if max_zones is not None and len(self.zones) >= max_zones:
                break
        
        print(f"Created {len(self.zones)} zones from {len(routes)} routes")
    
    def __len__(self):
        return len(self.zones)
    
    def __getitem__(self, idx):
        return collate_zone(self.zones[idx])

def main():
    ap = argparse.ArgumentParser(description="Train Pointer Transformer for TSP - Optimized for Full Compute")
    
    ap.add_argument("--json_root", required=True, help="Root directory containing dataset JSON files")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    ap.add_argument("--split", default="train", choices=["train","eval"])
    
    # Training args
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate (full rate, not scaled)")
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32, help="Effective batch size via gradient accumulation")
    ap.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers (reduced for memory)")
    
    # Model args
    ap.add_argument("--use_gnn", action="store_true")
    ap.add_argument("--d_model", type=int, default=256, help="Model dimension (increased from 128)")
    ap.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    ap.add_argument("--d_ff", type=int, default=512, help="Feedforward dimension")
    ap.add_argument("--nlayers_enc", type=int, default=4, help="Number of encoder layers")
    ap.add_argument("--nlayers_dec", type=int, default=4, help="Number of decoder layers")
    ap.add_argument("--dropout", type=float, default=0.1)
    
    # Data args
    ap.add_argument("--max_zone", type=int, default=60, help="Maximum zone size (reduced for memory)")
    ap.add_argument("--max_routes", type=int, default=None, help="Limit number of routes (None = use all training data)")
    ap.add_argument("--max_zones", type=int, default=None, help="Limit number of zones in training dataset (None = use all)")
    ap.add_argument("--val_split", type=float, default=0.1, help="Fraction of routes to use for validation")
    ap.add_argument("--val_eval_freq", type=int, default=1, help="Evaluate validation every N epochs")
    ap.add_argument("--val_num_zones", type=int, default=50, help="Number of zones to evaluate during validation")
    
    # Compute args
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training (faster)")
    ap.add_argument("--compile", action="store_true", help="Compile model with torch.compile (PyTorch 2.0+)")
    
    # Metrics and visualization
    ap.add_argument("--save_plots", action="store_true", help="Save training plots")
    ap.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    
    args = ap.parse_args()
    
    # Load config file if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Override args with config values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    # Ensure outputs directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    print(f"Loading dataset from {args.json_root}...")
    df = load_amzl_dataset(args.json_root, split=args.split)
    routes = df['route_id'].unique().tolist()
    
    # Limit routes if specified (but default to all)
    if args.max_routes:
        routes = routes[:args.max_routes]
        print(f"Limited to {len(routes)} routes for testing")
    else:
        print(f"Using ALL {len(routes)} routes - no compute restrictions!")
    
    # Split into train/val
    random.shuffle(routes)
    n_val = max(1, int(len(routes) * args.val_split))
    val_routes = routes[:n_val]
    train_routes = routes[n_val:]
    
    print(f"Split: {len(train_routes)} training routes, {len(val_routes)} validation routes")
    
    # Create datasets
    train_dataset = RouteZoneDataset(df, train_routes, max_zone=args.max_zone, min_stops=3, max_zones=args.max_zones)
    val_dataset = RouteZoneDataset(df, val_routes, max_zone=args.max_zone, min_stops=3)
    
    print(f"Training zones: {len(train_dataset.zones)}, Validation zones: {len(val_dataset.zones)}")
    
    # Metrics tracking
    training_metrics = {
        'epoch': [],
        'train_loss': [],
        'learning_rate': [],
        'val_kendall_tau_mean': [],
        'val_sequence_accuracy_mean': [],
        'val_distance_ratio_mean': [],
        'val_position_acc_k1_mean': [],
    }
    
    # Model configuration
    pt_config = PTConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        nlayers_enc=args.nlayers_enc,
        nlayers_dec=args.nlayers_dec,
        dropout=args.dropout,
        use_edge_bias=True
    )
    
    # Models
    d_node = 4  # [norm_lat, norm_lon, r, theta]
    
    if args.use_gnn:
        gnn = GraphSAGE(in_dim=d_node, hidden=args.d_model, out_dim=args.d_model).to(args.device)
        model = PointerTransformer(d_in=args.d_model, cfg=pt_config).to(args.device)
        params = list(model.parameters()) + list(gnn.parameters())
    else:
        gnn = None
        model = PointerTransformer(d_in=d_node, cfg=pt_config).to(args.device)
        params = list(model.parameters())
    
    # Initialize weights properly to prevent NaN
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller gain for stability
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m, mean=0.0, std=0.01)  # Small initialization for query_start
    
    model.apply(init_weights)
    if gnn:
        gnn.apply(init_weights)
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        if gnn:
            gnn = torch.compile(gnn)
    
    # Optimizer - use full learning rate (removed 0.1 multiplier)
    opt = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and args.device == "cuda" else None
    
    total_params = sum(p.numel() for p in params)
    print(f"\n{'='*60}")
    print(f"Model Configuration:")
    print(f"  Parameters: {total_params:,}")
    print(f"  d_model: {args.d_model}, nhead: {args.nhead}")
    print(f"  Encoder layers: {args.nlayers_enc}, Decoder layers: {args.nlayers_dec}")
    print(f"  Training on: {args.device}")
    print(f"  Batch size (gradient accumulation): {args.batch_size}")
    print(f"  Max zone size: {args.max_zone}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Total zones: {len(train_dataset.zones)}")
    print(f"{'='*60}\n")
    
    model.train()
    best_val_tau = -1.0
    
    # Create DataLoader once - don't recreate each epoch to save memory
    # Use a sampler that shuffles each epoch instead
    from torch.utils.data import RandomSampler
    train_sampler = RandomSampler(train_dataset, replacement=False)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1,
        sampler=train_sampler,  # Use sampler instead of shuffle=True
        num_workers=min(4, args.num_workers),  # Reduce workers to save memory
        pin_memory=True if args.device == "cuda" else False,
        persistent_workers=False,  # Don't persist to save memory
    )
    
    for epoch in range(1, args.epochs + 1):
        # Create new sampler each epoch for shuffling (lighter than new DataLoader)
        train_sampler = RandomSampler(train_dataset, replacement=False)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            sampler=train_sampler,
            num_workers=min(4, args.num_workers),
            pin_memory=True if args.device == "cuda" else False,
            persistent_workers=False,
        )
        
        total_loss = 0.0
        accumulated_loss = 0.0
        count = 0
        step_count = 0
        
        for batch_idx, (coords, target_idx) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}")):
            if batch_idx == 0:
                print(f"Processing first batch: coords shape={coords.shape}, target_idx shape={target_idx.shape}")
            
            coords = coords.to(args.device)
            target_idx = target_idx.to(args.device)
            
            # Squeeze extra batch dimension if DataLoader added one
            # collate_zone already adds batch dim, so DataLoader creates (1, 1, N, 2) -> (1, N, 2)
            while coords.dim() > 3:
                coords = coords.squeeze(0)
            # Ensure coords is exactly 3D (B, N, 2)
            if coords.dim() < 3:
                coords = coords.unsqueeze(0)
            
            # Only squeeze target_idx if it has 3 dimensions (1, 1, N) -> (1, N)
            # Don't squeeze if it's already (1, N) as we need the batch dimension
            while target_idx.dim() > 2:
                target_idx = target_idx.squeeze(0)
            # Ensure target_idx is exactly 2D (B, N)
            if target_idx.dim() < 2:
                target_idx = target_idx.unsqueeze(0)
            
            if batch_idx == 0:
                print(f"After squeezing: coords shape={coords.shape}, target_idx shape={target_idx.shape}")
            
            # Features
            X = node_features(coords)
            
            if args.use_gnn:
                adj = knn_adj(coords, k=min(8, coords.shape[1]-1))
                X = gnn(X, adj.to(args.device))
                
                # Ensure correct shape: should be (B, N, d_model)
                if X.dim() == 2:
                    # X is (N, d) - add batch dimension
                    X = X.unsqueeze(0)
                elif X.dim() == 3:
                    # X is 3D, check if dimensions are correct
                    if X.shape[0] == X.shape[1] and X.shape[0] != coords.shape[0]:
                        # GNN output is (N, N, d) instead of (B, N, d) - extract diagonal
                        N = X.shape[0]
                        X = X[torch.arange(N), torch.arange(N), :]  # Extract diagonal: (N, d)
                        X = X.unsqueeze(0)  # Add batch: (1, N, d)
                    elif X.shape[0] != coords.shape[0]:
                        # Batch size mismatch but not (N, N, d) case
                        X = X[:coords.shape[0]]  # Take first B batches
                
                # Final check
                expected_shape = (coords.shape[0], coords.shape[1], args.d_model)
                if X.shape != expected_shape:
                    raise ValueError(f"GNN output shape {X.shape} doesn't match expected {expected_shape}")
                
                del adj  # Free memory
            
            edge_feats = edge_bias_features(coords)
            
            # Forward pass with mixed precision
            if args.mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    loss = model.forward_teacher_forced(X, target_idx, edge_feats=edge_feats)
            else:
                loss = model.forward_teacher_forced(X, target_idx, edge_feats=edge_feats)
            
            # Free intermediate tensors
            del X, edge_feats
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                if batch_idx < 5:  # Only print first few for debugging
                    print(f"Warning: Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
                del loss
                continue
            
            # Scale loss for gradient accumulation
            loss_scaled = loss / args.batch_size
            accumulated_loss += loss.item()
            
            # Backward pass
            if args.mixed_precision and scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            
            count += 1
            step_count += 1
            
            # Update weights after accumulating batch_size gradients
            if step_count % args.batch_size == 0:
                if args.mixed_precision and scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                opt.zero_grad()
                total_loss += accumulated_loss
                accumulated_loss = 0.0
                
                if (step_count // args.batch_size) % 50 == 0:
                    print(f"Epoch {epoch}/{args.epochs}, Step {step_count // args.batch_size}, Avg Loss: {total_loss/(step_count):.4f}")
        
        # Handle remaining accumulated gradients
        if step_count % args.batch_size != 0:
            if args.mixed_precision and scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            opt.zero_grad()
            total_loss += accumulated_loss
        
        # Clean up DataLoader and free memory
        del train_dataloader
        if args.device == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        avg_loss = total_loss / max(1, count)
        performed_update = step_count > 0
        
        # Validation evaluation
        val_metrics_dict = {}
        if epoch % args.val_eval_freq == 0 or epoch == args.epochs:
            print(f"\nEvaluating on validation set...")
            val_metrics = []
            
            # Sample validation zones
            val_zones_sample = random.sample(val_dataset.zones, min(args.val_num_zones, len(val_dataset.zones)))
            
            model.eval()
            if gnn:
                gnn.eval()
            
            for zone_df in val_zones_sample:
                try:
                    # Extract zone DataFrame from dict if needed
                    if isinstance(zone_df, dict):
                        zone_df = zone_df['zone']
                    
                    # Debug: Print first zone's prediction
                    if len(val_metrics) == 0:
                        print(f"\nDEBUG: First validation zone - size: {len(zone_df)}")
                        print(f"  True order (first 10): {zone_df.sort_values('seq')['stop_id'].tolist()[:10]}")
                    
                    metrics = evaluate_zone_predictions(
                        model, gnn, zone_df, args.device, 
                        use_gnn=args.use_gnn, greedy=True
                    )
                    
                    if len(val_metrics) == 0:
                        print(f"  Metrics: tau={metrics.get('kendall_tau', 0):.4f}, seq_acc={metrics.get('sequence_accuracy', 0):.4f}")
                    
                    val_metrics.append(metrics)
                except Exception as e:
                    print(f"Warning: Validation evaluation failed for zone: {e}")
                    continue
            
            if val_metrics:
                agg_metrics = aggregate_metrics(val_metrics)
                val_metrics_dict = agg_metrics
                
                # Track key metrics
                training_metrics['val_kendall_tau_mean'].append(agg_metrics.get('kendall_tau_mean', 0.0))
                training_metrics['val_sequence_accuracy_mean'].append(agg_metrics.get('sequence_accuracy_mean', 0.0))
                training_metrics['val_distance_ratio_mean'].append(agg_metrics.get('distance_ratio_mean', 1.0))
                training_metrics['val_position_acc_k1_mean'].append(agg_metrics.get('position_acc_k1_mean', 0.0))
                
                print(f"Validation Metrics:")
                print(f"  Kendall Tau: {agg_metrics.get('kendall_tau_mean', 0.0):.4f} ± {agg_metrics.get('kendall_tau_std', 0.0):.4f}")
                print(f"  Sequence Accuracy: {agg_metrics.get('sequence_accuracy_mean', 0.0):.4f}")
                print(f"  Distance Ratio: {agg_metrics.get('distance_ratio_mean', 1.0):.4f}")
                print(f"  Position Acc @k=1: {agg_metrics.get('position_acc_k1_mean', 0.0):.4f}")
                
                # Save best model
                val_tau = agg_metrics.get('kendall_tau_mean', -1.0)
                if val_tau > best_val_tau:
                    best_val_tau = val_tau
                    best_model_path = run_dir / "best_model.pt"
                    torch.save({
                        'model': model.state_dict(),
                        'use_gnn': args.use_gnn,
                        'gnn': (gnn.state_dict() if gnn else None),
                        'config': pt_config.__dict__,
                        'args': vars(args),
                        'epoch': epoch,
                        'val_metrics': agg_metrics
                    }, best_model_path)
                    print(f"  → Saved best model (tau={val_tau:.4f})")
            else:
                # Fill with defaults if validation failed
                training_metrics['val_kendall_tau_mean'].append(0.0)
                training_metrics['val_sequence_accuracy_mean'].append(0.0)
                training_metrics['val_distance_ratio_mean'].append(1.0)
                training_metrics['val_position_acc_k1_mean'].append(0.0)
            
            model.train()
            if gnn:
                gnn.train()
        
        if performed_update:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            print("Skipping scheduler step: no batches processed this epoch.")
            current_lr = opt.param_groups[0]['lr']
        
        # Track training metrics
        training_metrics['epoch'].append(epoch)
        training_metrics['train_loss'].append(avg_loss)
        training_metrics['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch}/{args.epochs} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Zones Processed: {count}\n")
    
    # Save final model
    save_path = run_dir / "final_model.pt"
    torch.save({
        'model': model.state_dict(),
        'use_gnn': args.use_gnn,
        'gnn': (gnn.state_dict() if gnn else None),
        'config': pt_config.__dict__,
        'args': vars(args),
        'training_metrics': training_metrics
    }, save_path)
    
    # Also save to default location for compatibility
    default_path = output_dir / "pointer_transformer.pt"
    torch.save({
        'model': model.state_dict(),
        'use_gnn': args.use_gnn,
        'gnn': (gnn.state_dict() if gnn else None),
        'config': pt_config.__dict__,
        'args': vars(args)
    }, default_path)
    
    print(f"\nSaved final model to {save_path}")
    
    # Final comprehensive evaluation
    print("\n" + "="*60)
    print("Running final comprehensive evaluation...")
    print("="*60)
    
    final_eval_zones = random.sample(val_dataset.zones, min(100, len(val_dataset.zones)))
    final_metrics_list = []
    
    model.eval()
    if gnn:
        gnn.eval()
    
    for zone_df in final_eval_zones:
        try:
            metrics = evaluate_zone_predictions(
                model, gnn, zone_df, args.device,
                use_gnn=args.use_gnn, greedy=True
            )
            final_metrics_list.append(metrics)
        except Exception as e:
            continue
    
    if final_metrics_list:
        final_agg_metrics = aggregate_metrics(final_metrics_list)
        
        print("\nFinal Validation Metrics (Summary):")
        print("-" * 60)
        for key in sorted(final_agg_metrics.keys()):
            if 'mean' in key:
                print(f"{key:40s}: {final_agg_metrics[key]:.4f}")
        
        # Save metrics to JSON
        metrics_json = {
            'training_metrics': training_metrics,
            'final_validation_metrics': final_agg_metrics,
            'per_zone_metrics': final_metrics_list[:10],  # Sample of per-zone metrics
            'config': vars(args),
            'model_config': pt_config.__dict__
        }
        
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"\nSaved metrics to {metrics_path}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(training_metrics)
        metrics_csv_path = run_dir / "training_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved training metrics CSV to {metrics_csv_path}")
        
        # Create visualizations
        if HAS_MATPLOTLIB and args.save_plots:
            print("\nGenerating visualizations...")
            create_visualizations(training_metrics, final_agg_metrics, run_dir, args.val_eval_freq)
            print(f"Saved plots to {run_dir}")
    
    # Print summary report
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {args.epochs}")
    print(f"Final train loss: {training_metrics['train_loss'][-1]:.4f}")
    if training_metrics['val_kendall_tau_mean']:
        print(f"Best validation Kendall Tau: {max(training_metrics['val_kendall_tau_mean']):.4f}")
    print(f"Model saved to: {run_dir}")
    print(f"Best model: {run_dir / 'best_model.pt'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
