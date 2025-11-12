# src/train_pointer.py



import argparse, math, random

import numpy as np, pandas as pd, torch

import torch.nn as nn

from torch.optim import AdamW

from sklearn.cluster import KMeans

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_amzl_dataset

from models.pointer_transformer import PointerTransformer, PTConfig

from models.gnn_sage import GraphSAGE

from utils.knn_graph import knn_adj

from utils.features import node_features, edge_bias_features

def build_zones(df_route, max_zone=40, seed=0):

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

    t = zone_df.sort_values('seq').reset_index().index.values  # mapping: we need indices 0..N-1 in true order

    # We want target_idx s.t. we output stops in the order of 'seq'.

    # zone_df is current order; create a mapping from stop_id -> position in zone_df

    sid_to_pos = {sid:i for i,sid in enumerate(zone_df['stop_id'].tolist())}

    true_order = [sid_to_pos[sid] for sid in zone_df.sort_values('seq')['stop_id'].tolist()]

    target_idx = torch.tensor(true_order, dtype=torch.long).unsqueeze(0)  # (1,N)

    return coords, target_idx

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--json_root", required=True)

    ap.add_argument("--split", default="train", choices=["train","eval"])

    ap.add_argument("--epochs", type=int, default=15)

    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--weight_decay", type=float, default=1e-3)

    ap.add_argument("--use_gnn", action="store_true")

    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    
    ap.add_argument("--max_routes", type=int, default=None, help="Limit number of routes for faster testing")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    df = load_amzl_dataset(args.json_root, split=args.split)

    routes = df['route_id'].unique().tolist()

    # Models

    d_node = 4  # [norm_lat, norm_lon, r, theta]

    if args.use_gnn:

        gnn = GraphSAGE(in_dim=d_node, hidden=128, out_dim=128).to(args.device)

        model = PointerTransformer(d_in=128).to(args.device)

        params = list(model.parameters()) + list(gnn.parameters())

    else:

        gnn = None

        model = PointerTransformer(d_in=d_node).to(args.device)

        params = list(model.parameters())

    opt = AdamW(params, lr=args.lr * 0.1, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(1, args.epochs+1):
        random.shuffle(routes)
        
        # Limit routes if specified
        epoch_routes = routes[:args.max_routes] if args.max_routes else routes
        
        total_loss = 0.0
        count = 0
        accumulated_loss = 0.0
        
        for rid in epoch_routes:
            r = df[df.route_id==rid].copy().reset_index(drop=True)
            zones = build_zones(r, max_zone=40)
            
            for z in zones:
                if len(z) < 3:
                    continue
                
                coords, target_idx = collate_zone(z)
                coords = coords.to(args.device)
                target_idx = target_idx.to(args.device)
                
                # Features
                X = node_features(coords)
                
                if args.use_gnn:
                    adj = knn_adj(coords, k=8)
                    X = gnn(X, adj.to(args.device))
                
                edge_feats = edge_bias_features(coords)
                
                loss = model.forward_teacher_forced(X, target_idx, edge_feats=edge_feats)
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected (NaN/Inf) for zone with {len(z)} stops. Skipping.")
                    continue
                
                # Scale loss by batch_size for gradient accumulation
                loss = loss / args.batch_size
                loss.backward()
                
                accumulated_loss += loss.item() * args.batch_size  # Unscale for logging
                count += 1
                
                # Update weights after accumulating batch_size gradients
                if count % args.batch_size == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad()
                    total_loss += accumulated_loss
                    accumulated_loss = 0.0
        
        # Handle remaining accumulated gradients
        if count % args.batch_size != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            total_loss += accumulated_loss
        
        
        
        avg = total_loss / max(1, count)
        print(f"Epoch {epoch}/{args.epochs} - avg loss: {avg:.4f} (processed {count} zones)")

    torch.save({'model': model.state_dict(),

                'use_gnn': args.use_gnn,

                'gnn': (gnn.state_dict() if gnn else None)},

               f'./outputs/pointer_transformer.pt')

    print("Saved ./outputs/pointer_transformer.pt")

if __name__ == "__main__":

    main()

