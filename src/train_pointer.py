# src/train_pointer.py

import argparse
import math
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.cluster import KMeans

# Try importing matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_amzl_dataset
from models.pointer_transformer import PointerTransformer, PTConfig
from models.gnn_sage import GraphSAGE
from utils.knn_graph import knn_adj
from utils.features import node_features, edge_bias_features
from utils.metrics import evaluate_zone_predictions, aggregate_metrics
from baseline import kendall_tau


def build_zones(df_route, max_zone=80, seed=0):
    """Build zones from a route DataFrame."""
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
    """Collate a zone into tensors for training."""
    coords = torch.tensor(zone_df[['lat','lon']].to_numpy(), dtype=torch.float32).unsqueeze(0)  # (1,N,2)
    
    # target index by true seq
    sid_to_pos = {sid:i for i,sid in enumerate(zone_df['stop_id'].tolist())}
    true_order = [sid_to_pos[sid] for sid in zone_df.sort_values('seq')['stop_id'].tolist()]
    target_idx = torch.tensor(true_order, dtype=torch.long).unsqueeze(0)  # (1,N)
    
    return coords, target_idx, zone_df


def collate_fn(batch):
    """Custom collate function that handles DataFrames."""
    # batch is a list of (coords, target_idx, zone_df) tuples
    # For DataLoader, we only need coords and target_idx (zone_df is stored in dataset.zones)
    coords_list = [item[0] for item in batch]
    target_idx_list = [item[1] for item in batch]
    
    # Stack coords and target_idx
    coords = torch.cat(coords_list, dim=0)  # (B, N, 2)
    target_idx = torch.cat(target_idx_list, dim=0)  # (B, N)
    
    return coords, target_idx


class RouteZoneDataset(Dataset):
    """Dataset for route zones."""
    def __init__(self, df, routes, max_zone=80, min_stops=3):
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
        
        print(f"Created {len(self.zones)} zones from {len(routes)} routes")
    
    def __len__(self):
        return len(self.zones)
    
    def __getitem__(self, idx):
        coords, target_idx, zone_df = collate_zone(self.zones[idx])
        return coords, target_idx, zone_df


def train(args):
    """
    Train Pointer Transformer model on route sequencing task.
    """
    print("\n" + "="*80)
    print("TRAINING POINTER TRANSFORMER")
    print("="*80)
    
    # Load dataset
    print(f"Loading dataset from {args.json_root}...")
    df = load_amzl_dataset(args.json_root, split=args.split)
    routes = df['route_id'].unique().tolist()
    
    # Select routes for training
    num_routes = args.num_routes if args.num_routes else 1000
    if num_routes > len(routes):
        num_routes = len(routes)
        print(f"Warning: Requested {args.num_routes} routes but only {len(routes)} available. Using all routes.")
    
    random.seed(42)  # Fixed seed for reproducibility
    selected_routes = random.sample(routes, num_routes)
    print(f"Selected {len(selected_routes)} route(s) for training")
    
    # Create dataset
    dataset = RouteZoneDataset(df, selected_routes, max_zone=args.max_zone, min_stops=3)
    print(f"Total zones: {len(dataset.zones)}")
    
    if len(dataset.zones) == 0:
        print("ERROR: No zones created! Check data loading.")
        return False
    
    # Model setup
    pt_config = PTConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        nlayers_enc=args.nlayers_enc,
        nlayers_dec=args.nlayers_dec,
        dropout=args.dropout,
        use_edge_bias=True
    )
    
    d_node = 4  # [norm_lat, norm_lon, r, theta]
    
    if args.use_gnn:
        gnn = GraphSAGE(in_dim=d_node, hidden=args.d_model, out_dim=args.d_model).to(args.device)
        model = PointerTransformer(d_in=args.d_model, cfg=pt_config).to(args.device)
        params = list(model.parameters()) + list(gnn.parameters())
    else:
        gnn = None
        model = PointerTransformer(d_in=d_node, cfg=pt_config).to(args.device)
        params = list(model.parameters())
    
    # Initialize weights - use more conservative initialization for numerical stability
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller gain for numerical stability
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m, mean=0.0, std=0.01)  # Smaller std for numerical stability
    
    model.apply(init_weights)
    if gnn:
        gnn.apply(init_weights)
    
    # Initialize query_start with smaller std for numerical stability
    if hasattr(model, 'query_start'):
        nn.init.normal_(model.query_start, mean=0.0, std=0.01)
    
    # Use lower learning rate to prevent gradient explosion that can cause NaN
    # Default lr is 1e-3, but use 5e-4 for better stability
    effective_lr = min(args.lr, 5e-4) if args.lr > 5e-4 else args.lr
    opt = AdamW(params, lr=effective_lr, weight_decay=args.weight_decay)
    
    # Training loop
    model.train()
    if gnn:
        gnn.train()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  # Fixed order for reproducibility
    
    print(f"\nTraining for {args.epochs} epochs on {len(dataset.zones)} zones...\n")
    
    losses = []
    kendall_taus = []
    eval_epochs = []  # Track which epochs we evaluated at
    
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        
        # Add progress bar for batches in this epoch
        for batch_idx, (coords, target_idx) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")):
            coords = coords.to(args.device)
            target_idx = target_idx.to(args.device)
            
            # Ensure correct dimensions
            while coords.dim() > 3:
                coords = coords.squeeze(0)
            if coords.dim() < 3:
                coords = coords.unsqueeze(0)
            
            while target_idx.dim() > 2:
                target_idx = target_idx.squeeze(0)
            if target_idx.dim() < 2:
                target_idx = target_idx.unsqueeze(0)
            
            # Features
            X = node_features(coords)
            
            if args.use_gnn:
                adj = knn_adj(coords, k=min(8, coords.shape[1]-1))
                X = gnn(X, adj.to(args.device))
            
            edge_feats = edge_bias_features(coords)
            
            # Forward pass
            loss = model.forward_teacher_forced(X, target_idx, edge_feats=edge_feats)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at epoch {epoch}, batch {batch_idx}")
                continue
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        losses.append(avg_loss)
        
    
        model.eval()
        if gnn:
            gnn.eval()
        
        eval_metrics = []
        for zone_df in dataset.zones:  # Evaluate on all zones
            try:
                metrics = evaluate_zone_predictions(
                    model, gnn, zone_df, args.device,
                    use_gnn=args.use_gnn, greedy=True
                )
                eval_metrics.append(metrics)
            except Exception as e:
                print(f"Warning: Evaluation failed: {e}")
                continue
        
        # Print epoch summary once after evaluating all zones
        if eval_metrics:
            agg_metrics = aggregate_metrics(eval_metrics)
            tau = agg_metrics.get('kendall_tau_mean', 0.0)
            kendall_taus.append(tau)
            eval_epochs.append(epoch)  # Track which epoch this corresponds to
            
            print(f"Epoch {epoch:3d}/{args.epochs}: Loss={avg_loss:.6f}, Kendall τ={tau:.4f}")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs}: Loss={avg_loss:.6f}, Kendall τ=N/A")
        
        model.train()
        if gnn:
            gnn.train()
    
    # Final evaluation
    model.eval()
    if gnn:
        gnn.eval()
    
    final_metrics = []
    for zone_df in dataset.zones:
        try:
            metrics = evaluate_zone_predictions(
                model, gnn, zone_df, args.device,
                use_gnn=args.use_gnn, greedy=True
            )
            final_metrics.append(metrics)
        except Exception as e:
            continue
    
    if final_metrics:
        agg_metrics = aggregate_metrics(final_metrics)
        final_loss = losses[-1]
        final_tau = agg_metrics.get('kendall_tau_mean', 0.0)
        seq_acc = agg_metrics.get('sequence_accuracy_mean', 0.0)
        
        print("\n" + "-"*80)
        print("TRAINING RESULTS:")
        print("-"*80)
        print(f"Final Loss: {final_loss:.6f}")
        print(f"Final Kendall τ: {final_tau:.4f}")
        print(f"Sequence Accuracy: {seq_acc:.4f}")
        print(f"Zones evaluated: {len(final_metrics)}")
        print("-"*80)
        
        # Generate plots
        if HAS_MATPLOTLIB and len(losses) > 0:
            epochs_all = list(range(1, len(losses) + 1))
            
            # Plot 1: Loss vs Epochs
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs_all, losses, 'b-', linewidth=2, label='Training Loss')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            
            loss_plot_path = Path('training_loss.png')
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nSaved loss plot to {loss_plot_path}")
            
            # Plot 2: Kendall Tau vs Epochs
            if len(kendall_taus) > 0 and len(eval_epochs) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(eval_epochs, kendall_taus, 'r-o', linewidth=2, markersize=6, label='Kendall τ')
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Kendall τ', fontsize=12)
                ax.set_title('Validation Kendall τ vs Epochs', fontsize=14, fontweight='bold')
                ax.set_ylim([0, 1.0])
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                
                tau_plot_path = Path('training_kendall_tau.png')
                plt.savefig(tau_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved Kendall tau plot to {tau_plot_path}")
        
        return True
    else:
        print("ERROR: Could not evaluate final metrics")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train Pointer Transformer for route sequencing")
    
    # Data args
    parser.add_argument("--json_root", required=True, help="Root directory containing dataset JSON files")
    parser.add_argument("--split", default="train", choices=["train", "eval"])
    
    # Training args
    parser.add_argument("--num_routes", type=int, default=1000, help="Number of routes to use for training")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    # Model args
    parser.add_argument("--use_gnn", action="store_true", help="Use Graph Neural Network")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Feedforward dimension")
    parser.add_argument("--nlayers_enc", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--nlayers_dec", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Data args
    parser.add_argument("--max_zone", type=int, default=60, help="Maximum zone size")
    
    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Run training
    success = train(args)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

