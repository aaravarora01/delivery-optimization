import math, numpy as np, pandas as pd
from typing import List, Tuple

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    D = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            D[i,j] = D[j,i] = d
    return D

def nearest_neighbor(D: np.ndarray, start: int = 0) -> List[int]:
    n = D.shape[0]
    seen = [False]*n
    tour = [start]
    seen[start] = True
    for _ in range(n-1):
        last = tour[-1]
        nxt = np.ma.array(D[last], mask=seen).argmin().item()
        tour.append(nxt)
        seen[nxt] = True
    return tour

def tour_length(D: np.ndarray, tour: List[int], return_to_start=False) -> float:
    L = 0.0
    for i in range(len(tour)-1):
        L += D[tour[i], tour[i+1]]
    if return_to_start:
        L += D[tour[-1], tour[0]]
    return L

def two_opt(D: np.ndarray, tour: List[int], max_iters: int = 10000) -> List[int]:
    n = len(tour)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                a, b = tour[i-1], tour[i]
                c, d = tour[k], tour[k+1]
                delta = D[a, c] + D[b, d] - (D[a, b] + D[c, d])
                if delta < -1e-9:
                    tour[i:k+1] = reversed(tour[i:k+1])
                    improved = True
        # quick break if no change
    return tour

def solve_route(df_route: pd.DataFrame, return_to_start=False):
    # Expect columns: stop_id, lat, lon
    coords = df_route[['lat','lon']].to_numpy()
    D = pairwise_distance_matrix(coords)

    # start at the stop closest to the centroid to mimic "route origin" if not given
    centroid = coords.mean(axis=0)
    start = np.argmin([haversine(centroid[0], centroid[1], r[0], r[1]) for r in coords])

    nn = nearest_neighbor(D, start=start)
    best = two_opt(D, nn)

    out = df_route.copy().reset_index(drop=True)
    out = out.iloc[best].reset_index(drop=True)
    out['pred_seq'] = range(1, len(best)+1)
    dist_km = tour_length(D, best, return_to_start=return_to_start)
    return out, dist_km

def kendall_tau(pred_order: List[int], true_order: List[int]) -> float:
    n = len(pred_order)
    pos = {sid:i for i, sid in enumerate(pred_order)}
    concord, discord = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            a = pos[true_order[i]] < pos[true_order[j]]
            b = i < j
            concord += int(a == b)
            discord += int(a != b)
    denom = concord + discord
    return (concord - discord)/denom if denom else 0.0

def evaluate(df_route_pred: pd.DataFrame, df_route_true: pd.DataFrame):
    # both must have stop_id; df_route_true may have 'seq' (driver order)
    if 'seq' in df_route_true.columns:
        true_sorted = df_route_true.sort_values('seq')
        tau = kendall_tau(
            df_route_pred['stop_id'].tolist(),
            true_sorted['stop_id'].tolist()
        )
    else:
        tau = None
    return tau