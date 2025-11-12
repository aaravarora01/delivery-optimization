# src/decode_pointer.py



import argparse, math, numpy as np, pandas as pd, torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.cluster import KMeans

from data_loader import load_amzl_dataset

from models.pointer_transformer import PointerTransformer, PTConfig

from models.gnn_sage import GraphSAGE

from utils.knn_graph import knn_adj

from utils.features import node_features, edge_bias_features

from utils.beam_search import beam_search_pointer

from baseline import two_opt, pairwise_distance_matrix, tour_length

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

        zone_df = df_route[z==c].copy()

        # Preserve original indices for mapping back

        zones.append(zone_df.reset_index(drop=True))

    return zones

def zone_decode(model, gnn, zone_df, device, beam=4, refine_2opt=True):

    coords = torch.tensor(zone_df[['lat','lon']].to_numpy(), dtype=torch.float32, device=device).unsqueeze(0)

    X = node_features(coords)

    if gnn is not None:

        adj = knn_adj(coords, k=8)

        X = gnn(X, adj.to(device))

    edge_feats = edge_bias_features(coords)

    seq_idx, _ = beam_search_pointer(model, X, edge_feats=edge_feats, beam_size=beam)  # (1,N)

    order = seq_idx.squeeze(0).cpu().numpy().tolist()

    # Optional 2-opt refine on geometric distance

    if refine_2opt:

        c = zone_df[['lat','lon']].to_numpy()

        D = pairwise_distance_matrix(c)

        order = two_opt(D, order)

    return order

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--json_root", required=True)

    ap.add_argument("--split", default="eval", choices=["train","eval"])

    ap.add_argument("--route_id", default=None)

    ap.add_argument("--checkpoint", default="./outputs/pointer_transformer.pt")

    ap.add_argument("--use_gnn", action="store_true")

    ap.add_argument("--beam", type=int, default=4)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    df = load_amzl_dataset(args.json_root, split=args.split)

    rid = args.route_id or df['route_id'].iloc[0]

    rid = str(rid)

    r = df[df.route_id==rid].copy().reset_index(drop=True)

    # build models & load weights

    d_node = 4

    model = PointerTransformer(d_in=(128 if args.use_gnn else d_node)).to(args.device)

    gnn = GraphSAGE(in_dim=d_node, hidden=128, out_dim=128).to(args.device) if args.use_gnn else None

    ckpt = torch.load(args.checkpoint, map_location=args.device)

    model.load_state_dict(ckpt['model'])

    if args.use_gnn and ckpt.get('gnn') is not None:

        gnn.load_state_dict(ckpt['gnn'])

    model.eval()

    if gnn is not None:

        gnn.eval()

    # zone-first decode

    zones = build_zones(r, max_zone=40)

    full_order = []

    for z in zones:

        order = zone_decode(model, gnn, z, args.device, beam=args.beam, refine_2opt=True)

        # map back to stop_ids in the zone, then find their indices in r

        zone_stop_ids = z['stop_id'].tolist()

        ordered_stop_ids = [zone_stop_ids[i] for i in order]

        # Find indices in original route r

        idxs = [r[r['stop_id'] == sid].index[0] for sid in ordered_stop_ids]

        full_order.extend(idxs)

    pred = r.iloc[full_order].copy().reset_index(drop=True)

    pred['pred_seq'] = np.arange(1, len(pred)+1)

    out = f'./outputs/route_{rid}_pointer.csv'

    pred.to_csv(out, index=False)

    # report distance

    D = pairwise_distance_matrix(pred[['lat','lon']].to_numpy())

    dist_km = tour_length(D, list(range(len(pred))))

    # Kendall-tau if ground truth available

    tau = None

    if 'seq' in r.columns:

        true_sorted = r.sort_values('seq')

        tau = 0

        # simple tau: compare orders

        pos = {sid:i for i,sid in enumerate(pred['stop_id'].tolist())}

        s_true = true_sorted['stop_id'].tolist()

        concord, discord = 0, 0

        for i in range(len(s_true)):

            for j in range(i+1, len(s_true)):

                a = pos[s_true[i]] < pos[s_true[j]]

                b = i < j

                concord += int(a==b); discord += int(a!=b)

        denom = concord+discord

        tau = (concord-discord)/denom if denom else 0.0

    print(f"[pointer] route {rid}: dist_km={dist_km:.2f}" + (f"  kendall_tau={tau:.3f}" if tau is not None else ""))

    print(f"Saved {out}")

if __name__ == "__main__":

    main()

