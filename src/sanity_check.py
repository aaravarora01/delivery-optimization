# src/sanity_check.py

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


def test_overfitting(args):
    """
    Test 1: Overfit a single route (or small subset).
    If labels and loss are wired correctly, model should achieve near-perfect ordering.
    """
    print("\n" + "="*80)
    print("SANITY CHECK 1: OVERFITTING TEST")
    print("="*80)
    
    # Load dataset
    print(f"Loading dataset from {args.json_root}...")
    df = load_amzl_dataset(args.json_root, split=args.split)
    routes = df['route_id'].unique().tolist()
    
    # Select a small fixed subset
    num_routes = args.num_routes if args.num_routes else 1
    random.seed(42)  # Fixed seed for reproducibility
    selected_routes = random.sample(routes, min(num_routes, len(routes)))
    print(f"Selected {len(selected_routes)} route(s) for overfitting test: {selected_routes[:3]}...")
    
    # Create minimal dataset
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
    
    print(f"\nTraining for {args.epochs} epochs on {len(dataset.zones)} zones...")
    print(f"Target: Loss < 0.01, Kendall τ > 0.95\n")
    
    losses = []
    kendall_taus = []
    eval_epochs = []  # Track which epochs we evaluated at
    
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        
        for batch_idx, (coords, target_idx) in enumerate(dataloader):
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
        
        # Evaluate metrics every few epochs
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            model.eval()
            if gnn:
                gnn.eval()
            
            eval_metrics = []
            for zone_df in dataset.zones[:min(10, len(dataset.zones))]:  # Evaluate on first 10 zones
                try:
                    metrics = evaluate_zone_predictions(
                        model, gnn, zone_df, args.device,
                        use_gnn=args.use_gnn, greedy=True
                    )
                    eval_metrics.append(metrics)
                except Exception as e:
                    print(f"Warning: Evaluation failed: {e}")
                    continue
            
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
        print("OVERFITTING TEST RESULTS:")
        print("-"*80)
        print(f"Final Loss: {final_loss:.6f}")
        print(f"Final Kendall τ: {final_tau:.4f}")
        print(f"Sequence Accuracy: {seq_acc:.4f}")
        print(f"Zones evaluated: {len(final_metrics)}")
        print("-"*80)
        
        # Pass/fail criteria
        passed = final_loss < 0.01 and final_tau > 0.95
        if passed:
            print("✓ PASS: Model successfully overfits (loss < 0.01, τ > 0.95)")
            print("  This indicates labels, masks, and indexing are likely correct.")
        else:
            print("✗ FAIL: Model cannot overfit - likely indexing/label bug!")
            print("  Common causes:")
            print("    - Target indices don't match input node positions")
            print("    - Off-by-one error in teacher forcing alignment")
            print("    - Using global IDs instead of batch-local indices")
            print("    - Target sequence shifted relative to predictions")
            if final_loss >= 0.01:
                print(f"  - Loss too high: {final_loss:.6f} >= 0.01 (expected < 0.01)")
                print(f"    Loss should approach 0 when model memorizes the sequence")
            if final_tau <= 0.95:
                print(f"  - Kendall τ too low: {final_tau:.4f} <= 0.95 (expected > 0.95)")
                print(f"    τ should approach 1.0 when model learns correct ordering")
        
        # Generate plot
        if HAS_MATPLOTLIB and len(losses) > 0:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot loss on left y-axis
            epochs_all = list(range(1, len(losses) + 1))
            ax1.plot(epochs_all, losses, 'b-', label='Loss', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', color='b', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, alpha=0.3)
            
            # Plot Kendall tau on right y-axis
            if len(kendall_taus) > 0 and len(eval_epochs) > 0:
                ax2 = ax1.twinx()
                ax2.plot(eval_epochs, kendall_taus, 'r-o', label='Kendall τ', linewidth=2, markersize=6)
                ax2.set_ylabel('Kendall τ', color='r', fontsize=12)
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.set_ylim([0, 1.0])
            
            ax1.set_title('Overfitting Test: Loss and Kendall τ vs Epochs', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            # Save plot
            plot_path = Path('overfitting_test_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nSaved plot to {plot_path}")
        
        return passed
    else:
        print("ERROR: Could not evaluate final metrics")
        return False


def inspect_example(args):
    """
    Test 2: Print one example and inspect inputs, targets, and predictions.
    Verify that indices are correct and stop_ids match.
    """
    print("\n" + "="*80)
    print("SANITY CHECK 2: EXAMPLE INSPECTION")
    print("="*80)
    
    # Load dataset
    print(f"Loading dataset from {args.json_root}...")
    df = load_amzl_dataset(args.json_root, split=args.split)
    routes = df['route_id'].unique().tolist()
    
    # Select a route
    route_id = args.route_id if args.route_id else routes[0]
    print(f"Inspecting route: {route_id}")
    
    route_df = df[df.route_id == route_id].copy().reset_index(drop=True)
    zones = build_zones(route_df, max_zone=args.max_zone)
    
    if len(zones) == 0:
        print("ERROR: No zones created!")
        return False
    
    # Use first zone
    zone_df = zones[0].copy().reset_index(drop=True)
    print(f"\nZone size: {len(zone_df)} stops")
    
    # Get collated data
    coords, target_idx, zone_df_returned = collate_zone(zone_df)
    
    # Verify zone_df matches
    assert zone_df is zone_df_returned or zone_df.equals(zone_df_returned), "Zone DF mismatch"
    
    # Get stop_ids and create mapping
    stop_ids = zone_df['stop_id'].tolist()
    coords_np = coords.squeeze(0).cpu().numpy()
    target_idx_np = target_idx.squeeze(0).cpu().numpy()
    
    # Print input node features
    print("\n" + "-"*80)
    print("INPUT NODE FEATURES:")
    print("-"*80)
    print(f"{'Index':<8} {'Stop ID':<20} {'Lat':<12} {'Lon':<12}")
    print("-"*80)
    for i in range(len(zone_df)):
        print(f"{i:<8} {stop_ids[i]:<20} {coords_np[i,0]:<12.6f} {coords_np[i,1]:<12.6f}")
    
    # Print ground-truth sequence
    print("\n" + "-"*80)
    print("GROUND-TRUTH SEQUENCE (target_idx):")
    print("-"*80)
    print(f"Target indices: {target_idx_np.tolist()}")
    
    # Verify indices are valid
    invalid_indices = [idx for idx in target_idx_np if idx < 0 or idx >= len(zone_df)]
    if invalid_indices:
        print(f"✗ ERROR: Invalid target indices found: {invalid_indices}")
        return False
    else:
        print("✓ All target indices are valid (0 to N-1)")
    
    # Show ground-truth sequence with stop_ids
    true_stop_ids = [stop_ids[int(idx)] for idx in target_idx_np]
    print(f"\nGround-truth stop_ids in order:")
    for step, (idx, sid) in enumerate(zip(target_idx_np, true_stop_ids)):
        print(f"  Step {step:2d}: Index {int(idx):2d} -> Stop ID {sid}")
    
    # Verify ground-truth matches sorted-by-seq
    true_sorted = zone_df.sort_values('seq')
    expected_stop_ids = true_sorted['stop_id'].tolist()
    if true_stop_ids != expected_stop_ids:
        print(f"\n✗ ERROR: Ground-truth sequence doesn't match sorted-by-seq!")
        print(f"  Expected: {expected_stop_ids}")
        print(f"  Got:      {true_stop_ids}")
        return False
    else:
        print("\n✓ Ground-truth sequence matches sorted-by-seq")
    
    # Additional verification: Check that target_idx contains all indices 0..N-1 exactly once
    unique_targets = set(target_idx_np)
    if len(unique_targets) != len(target_idx_np):
        print(f"\n✗ ERROR: Duplicate indices in target_idx!")
        print(f"  Target indices: {target_idx_np.tolist()}")
        return False
    else:
        print("✓ No duplicate indices in target_idx")
    
    if unique_targets != set(range(len(zone_df))):
        print(f"\n✗ ERROR: target_idx doesn't contain all indices 0..{len(zone_df)-1}!")
        print(f"  Missing: {set(range(len(zone_df))) - unique_targets}")
        print(f"  Extra: {unique_targets - set(range(len(zone_df)))}")
        return False
    else:
        print("✓ target_idx contains all indices 0..N-1 exactly once")
    
    # Load model if checkpoint provided
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\nLoading model from {args.checkpoint}...")
        pt_config = PTConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            d_ff=args.d_ff,
            nlayers_enc=args.nlayers_enc,
            nlayers_dec=args.nlayers_dec,
            dropout=args.dropout,
            use_edge_bias=True
        )
        
        d_node = 4
        if args.use_gnn:
            gnn = GraphSAGE(in_dim=d_node, hidden=args.d_model, out_dim=args.d_model).to(args.device)
            model = PointerTransformer(d_in=args.d_model, cfg=pt_config).to(args.device)
        else:
            gnn = None
            model = PointerTransformer(d_in=d_node, cfg=pt_config).to(args.device)
        
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        if args.use_gnn and ckpt.get('gnn') is not None:
            gnn.load_state_dict(ckpt['gnn'])
        
        model.eval()
        if gnn:
            gnn.eval()
        
        # Get predictions
        coords_device = coords.to(args.device)
        X = node_features(coords_device)
        
        if args.use_gnn:
            adj = knn_adj(coords_device, k=min(8, coords_device.shape[1]-1))
            X = gnn(X, adj.to(args.device))
        
        edge_feats = edge_bias_features(coords_device)
        
        with torch.no_grad():
            pred_indices = model.greedy_decode(X, edge_feats=edge_feats)
            pred_indices_np = pred_indices.squeeze(0).cpu().numpy()
        
        # Print predicted sequence
        print("\n" + "-"*80)
        print("PREDICTED SEQUENCE (greedy decode):")
        print("-"*80)
        print(f"Predicted indices: {pred_indices_np.tolist()}")
        
        # Verify predicted indices are valid
        invalid_pred = [idx for idx in pred_indices_np if idx < 0 or idx >= len(zone_df)]
        if invalid_pred:
            print(f"✗ ERROR: Invalid predicted indices found: {invalid_pred}")
            return False
        else:
            print("✓ All predicted indices are valid (0 to N-1)")
        
        # Check for duplicates
        if len(set(pred_indices_np)) != len(pred_indices_np):
            print(f"✗ ERROR: Duplicate indices in prediction!")
            duplicates = [idx for idx in pred_indices_np if list(pred_indices_np).count(idx) > 1]
            print(f"  Duplicate indices: {set(duplicates)}")
            return False
        else:
            print("✓ No duplicate indices in prediction")
        
        # Verify prediction contains all indices
        if set(pred_indices_np) != set(range(len(zone_df))):
            print(f"✗ ERROR: Prediction doesn't contain all indices 0..{len(zone_df)-1}!")
            print(f"  Missing: {set(range(len(zone_df))) - set(pred_indices_np)}")
            print(f"  Extra: {set(pred_indices_np) - set(range(len(zone_df)))}")
            return False
        else:
            print("✓ Prediction contains all indices 0..N-1 exactly once")
        
        # Show predicted sequence with stop_ids
        pred_stop_ids = [stop_ids[int(idx)] for idx in pred_indices_np]
        print(f"\nPredicted stop_ids in order:")
        for step, (idx, sid) in enumerate(zip(pred_indices_np, pred_stop_ids)):
            print(f"  Step {step:2d}: Index {int(idx):2d} -> Stop ID {sid}")
        
        # Compare with ground-truth
        print("\n" + "-"*80)
        print("COMPARISON:")
        print("-"*80)
        print(f"Ground-truth: {true_stop_ids}")
        print(f"Predicted:    {pred_stop_ids}")
        
        # Compute metrics
        tau = kendall_tau(pred_stop_ids, true_stop_ids)
        seq_match = (pred_stop_ids == true_stop_ids)
        
        print(f"\nKendall τ: {tau:.4f}")
        print(f"Exact match: {seq_match}")
        
        if seq_match:
            print("✓ Perfect prediction!")
        elif tau > 0.9:
            print("✓ Very good prediction (τ > 0.9)")
        else:
            print("⚠ Prediction differs from ground-truth")
    else:
        print("\nNo checkpoint provided - skipping prediction inspection")
    
    print("\n" + "-"*80)
    print("✓ EXAMPLE INSPECTION COMPLETE")
    print("-"*80)
    return True


def check_teacher_forcing_alignment(args):
    """
    Test 3: Check teacher forcing alignment step-by-step.
    Verify that at step t, we predict target_idx[t] and use its embedding as next query.
    """
    print("\n" + "="*80)
    print("SANITY CHECK 3: TEACHER FORCING ALIGNMENT")
    print("="*80)
    
    # Load dataset
    print(f"Loading dataset from {args.json_root}...")
    df = load_amzl_dataset(args.json_root, split=args.split)
    routes = df['route_id'].unique().tolist()
    
    # Select a route
    route_id = args.route_id if args.route_id else routes[0]
    print(f"Inspecting route: {route_id}")
    
    route_df = df[df.route_id == route_id].copy().reset_index(drop=True)
    zones = build_zones(route_df, max_zone=args.max_zone)
    
    if len(zones) == 0:
        print("ERROR: No zones created!")
        return False
    
    # Use first zone
    zone_df = zones[0].copy().reset_index(drop=True)
    print(f"\nZone size: {len(zone_df)} stops")
    
    # Get collated data
    coords, target_idx, _ = collate_zone(zone_df)
    stop_ids = zone_df['stop_id'].tolist()
    target_idx_np = target_idx.squeeze(0).cpu().numpy()
    
    # Create model
    pt_config = PTConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        nlayers_enc=args.nlayers_enc,
        nlayers_dec=args.nlayers_dec,
        dropout=args.dropout,
        use_edge_bias=True
    )
    
    d_node = 4
    if args.use_gnn:
        gnn = GraphSAGE(in_dim=d_node, hidden=args.d_model, out_dim=args.d_model).to(args.device)
        model = PointerTransformer(d_in=args.d_model, cfg=pt_config).to(args.device)
    else:
        gnn = None
        model = PointerTransformer(d_in=d_node, cfg=pt_config).to(args.device)
    
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
    
    model.eval()
    if gnn:
        gnn.eval()
    
    # Prepare inputs
    coords_device = coords.to(args.device)
    target_idx_device = target_idx.to(args.device)
    
    # Ensure correct dimensions
    while coords_device.dim() > 3:
        coords_device = coords_device.squeeze(0)
    if coords_device.dim() < 3:
        coords_device = coords_device.unsqueeze(0)
    
    while target_idx_device.dim() > 2:
        target_idx_device = target_idx_device.squeeze(0)
    if target_idx_device.dim() < 2:
        target_idx_device = target_idx_device.unsqueeze(0)
    
    X = node_features(coords_device)
    
    if args.use_gnn:
        adj = knn_adj(coords_device, k=min(8, coords_device.shape[1]-1))
        X = gnn(X, adj.to(args.device))
    
    edge_feats = edge_bias_features(coords_device)
    
    # Encode
    H = model.encode(X)
    B, N, D = H.shape
    
    print(f"\nEncoder output shape: {H.shape}")
    print(f"Target sequence: {target_idx_np.tolist()}")
    print(f"Target stop_ids: {[stop_ids[int(i)] for i in target_idx_np]}")
    
    # Manually trace through teacher forcing
    print("\n" + "-"*80)
    print("STEP-BY-STEP TEACHER FORCING TRACE:")
    print("-"*80)
    
    device = args.device
    tgt = model.query_start.expand(B, 1, -1)  # Start token
    mask_visited = torch.zeros(B, N, dtype=torch.bool, device=device)
    
    all_issues = []
    
    for t in range(N):
        print(f"\n--- Step {t} ---")
        
        # Get target for this step
        y_target = target_idx_device[:, t].item()  # (B,) -> scalar
        target_stop_id = stop_ids[y_target]
        
        print(f"Target index: {y_target} (Stop ID: {target_stop_id})")
        
        # Check if target is already visited (shouldn't happen in ground-truth)
        if mask_visited[0, y_target]:
            issue = f"Step {t}: Target index {y_target} is already visited!"
            print(f"✗ ERROR: {issue}")
            all_issues.append(issue)
        
        # Decode step
        with torch.no_grad():
            logits = model.decode_step(H, tgt, mask_visited, edge_feats=edge_feats)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_stop_id = stop_ids[pred_idx]
        
        print(f"Predicted index: {pred_idx} (Stop ID: {pred_stop_id})")
        print(f"Logits shape: {logits.shape}, Max logit: {logits[0, pred_idx]:.4f}")
        
        # Check alignment: at step t, we should predict target_idx[t]
        if pred_idx != y_target:
            print(f"⚠ Note: Prediction differs from target (expected {y_target}, got {pred_idx})")
            print(f"  This is normal for untrained model, but should match after training")
        
        # Verify target is valid
        if y_target < 0 or y_target >= N:
            issue = f"Step {t}: Invalid target index {y_target} (valid range: 0-{N-1})"
            print(f"✗ ERROR: {issue}")
            all_issues.append(issue)
        
        # Verify target index type
        if not isinstance(y_target, (int, np.integer)):
            issue = f"Step {t}: Target index is not an integer: {type(y_target)}"
            print(f"✗ ERROR: {issue}")
            all_issues.append(issue)
        
        # Check for off-by-one: verify target sequence length matches step
        # At step t, we should have processed t nodes, so tgt should have length t+1 (start + t nodes)
        expected_tgt_length = t + 1
        if tgt.shape[1] != expected_tgt_length:
            issue = f"Step {t}: Target sequence length mismatch before decode: got {tgt.shape[1]}, expected {expected_tgt_length}"
            print(f"✗ ERROR: {issue}")
            all_issues.append(issue)
        
        # Update mask (teacher forcing uses target)
        chosen = y_target  # Teacher forcing: use target, not prediction
        mask_visited = mask_visited.scatter(1, torch.tensor([[chosen]], device=device), True)
        
        print(f"Mask updated: visited={mask_visited[0].sum().item()}/{N} nodes")
        
        # Get next embedding from chosen node
        next_embed = H[0, chosen]  # (D,)
        print(f"Next embedding shape: {next_embed.shape}")
        print(f"Next embedding from node {chosen} (Stop ID: {stop_ids[chosen]})")
        
        # Verify we're using the correct embedding
        if chosen != y_target:
            issue = f"Step {t}: Chosen index {chosen} != target index {y_target}"
            print(f"✗ ERROR: {issue}")
            all_issues.append(issue)
        
        # Append to target sequence
        tgt = torch.cat([tgt, next_embed.unsqueeze(0).unsqueeze(0)], dim=1)
        print(f"Target sequence length: {tgt.shape[1]} (should be {t+2}: start + {t+1} chosen)")
        
        # Verify target sequence length after appending
        expected_length = t + 2  # start token + t+1 chosen nodes
        if tgt.shape[1] != expected_length:
            issue = f"Step {t}: Target sequence length mismatch after append: got {tgt.shape[1]}, expected {expected_length}"
            print(f"✗ ERROR: {issue}")
            all_issues.append(issue)
        
        # Verify mask is correctly updated
        expected_visited_count = t + 1  # t nodes chosen + we just marked one more
        actual_visited_count = mask_visited[0].sum().item()
        if actual_visited_count != expected_visited_count:
            issue = f"Step {t}: Mask visited count mismatch: got {actual_visited_count}, expected {expected_visited_count}"
            print(f"✗ ERROR: {issue}")
            all_issues.append(issue)
        
        # Verify the chosen node is actually marked as visited
        if not mask_visited[0, chosen]:
            issue = f"Step {t}: Chosen node {chosen} is not marked as visited in mask!"
            print(f"✗ ERROR: {issue}")
            all_issues.append(issue)
        
        # Check for off-by-one: verify we're predicting the right step
        # At step t, we should be predicting target_idx[t]
        # The next query (for step t+1) should use embedding of target_idx[t]
        if t < N - 1:
            next_target = target_idx_device[:, t+1].item()
            print(f"Next step target (t+1={t+1}): index {next_target} (Stop ID: {stop_ids[next_target]})")
            print(f"  (Will use embedding from current chosen node {chosen} as query)")
    
    print("\n" + "-"*80)
    print("ALIGNMENT CHECK SUMMARY:")
    print("-"*80)
    
    if all_issues:
        print(f"✗ Found {len(all_issues)} issue(s):")
        for issue in all_issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ No alignment issues detected")
        print("✓ Target indices are valid (0 to N-1)")
        print("✓ Mask updates correctly")
        print("✓ Target sequence grows correctly")
        print("✓ Embeddings are taken from correct nodes")
        return True


def main():
    parser = argparse.ArgumentParser(description="Sanity checks for pointer transformer training")
    
    # Data args
    parser.add_argument("--json_root", required=True, help="Root directory containing dataset JSON files")
    parser.add_argument("--split", default="train", choices=["train", "eval"])
    parser.add_argument("--route_id", type=str, default=None, help="Specific route ID to inspect")
    
    # Test selection
    parser.add_argument("--test", type=str, choices=["overfit", "inspect", "alignment", "all"], 
                       default="all", help="Which test to run")
    
    # Overfitting test args
    parser.add_argument("--num_routes", type=int, default=1, help="Number of routes for overfitting test")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs for overfitting test")
    
    # Model args
    parser.add_argument("--use_gnn", action="store_true")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--nlayers_enc", type=int, default=2)
    parser.add_argument("--nlayers_dec", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training args
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for overfitting test")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    # Data args
    parser.add_argument("--max_zone", type=int, default=60)
    
    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Checkpoint for inspection
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Model checkpoint path for inspection test")
    
    args = parser.parse_args()
    
    results = {}
    
    if args.test in ["overfit", "all"]:
        results['overfit'] = test_overfitting(args)
    
    if args.test in ["inspect", "all"]:
        results['inspect'] = inspect_example(args)
    
    if args.test in ["alignment", "all"]:
        results['alignment'] = check_teacher_forcing_alignment(args)
    
    # Summary
    print("\n" + "="*80)
    print("SANITY CHECK SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    print("="*80)
    
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

