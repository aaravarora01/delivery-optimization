import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from baseline import haversine, pairwise_distance_matrix, nearest_neighbor, two_opt, tour_length

def edge_features(stops_df: pd.DataFrame, i: int, j: int):
    # Minimal, cheap features. You can extend later.
    a = stops_df.iloc[i]; b = stops_df.iloc[j]
    dist = haversine(a.lat, a.lon, b.lat, b.lon)
    dlat = b.lat - a.lat
    dlon = b.lon - a.lon
    return np.array([dist, dlat, dlon])

def build_training_edges(df: pd.DataFrame, max_routes=200):
    # Expect: route_id, stop_id, lat, lon, seq
    X, y = [], []
    rid_list = df['route_id'].drop_duplicates().head(max_routes).tolist()
    for rid in rid_list:
        r = df[df.route_id==rid].sort_values('seq').reset_index(drop=True)
        n = len(r)
        if n < 4: continue
        # positives: consecutive edges in driver order
        for i in range(n-1):
            X.append(edge_features(r, i, i+1)); y.append(1)
            # negatives: sample a few alternative j's
            for j in np.random.choice([k for k in range(n) if k not in [i, i+1]], size=min(3, n-2), replace=False):
                X.append(edge_features(r, i, j)); y.append(0)
    X = np.vstack(X); y = np.array(y)
    return X, y

class EdgeScorer:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, df: pd.DataFrame, max_routes=200):
        X, y = build_training_edges(df, max_routes=max_routes)
        self.clf.fit(X, y)
        return self

    def score_matrix(self, stops_df: pd.DataFrame) -> np.ndarray:
        n = len(stops_df)
        S = np.zeros((n,n))
        for i in range(n):
            feats = [edge_features(stops_df, i, j) for j in range(n)]
            # convert proba of “is next” to a cost
            p = self.clf.predict_proba(np.vstack(feats))[:,1]
            S[i,:] = p
            S[i,i] = 0.0
        # turn to costs (higher prob -> lower cost)
        C = -S
        return C

def solve_with_learned_cost(stops_df: pd.DataFrame, C: np.ndarray, start: int = 0):
    # greedy by learned cost, then 2-opt using distance (to avoid weird metric issues)
    n = len(stops_df)
    seen = [False]*n
    tour = [start]; seen[start] = True
    for _ in range(n-1):
        i = tour[-1]
        # mask seen
        costs = np.ma.array(C[i], mask=seen)
        nxt = costs.argmin().item()
        tour.append(nxt); seen[nxt] = True
    # Refine with 2-opt using geometric distance for stability
    D = pairwise_distance_matrix(stops_df[['lat','lon']].to_numpy())
    tour = two_opt(D, tour)
    dist_km = tour_length(D, tour)
    return tour, dist_km