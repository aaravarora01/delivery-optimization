import pandas as pd, numpy as np
from learn_edges import EdgeScorer, solve_with_learned_cost
from baseline import kendall_tau
from typing import Optional

def run_with_dataframes(train_df: pd.DataFrame, test_df: pd.DataFrame, route_id: Optional[str] = None):
    model = EdgeScorer().fit(train_df, max_routes=300)
    if route_id is None:
        route_id = test_df['route_id'].iloc[0]
    r = test_df[test_df.route_id==route_id].copy().reset_index(drop=True)
    C = model.score_matrix(r[['stop_id','lat','lon']])
    # start at first stop in driver seq if available, else 0
    start = 0
    if 'seq' in r.columns:
        start = r.sort_values('seq').index[0]
    tour, dist_km = solve_with_learned_cost(r, C, start=start)
    pred = r.iloc[tour].copy()
    pred['pred_seq'] = np.arange(1, len(pred)+1)
    pred.to_csv(f'./outputs/route_{route_id}_learned.csv', index=False)
    tau = None
    if 'seq' in r.columns:
        true_sorted = r.sort_values('seq')
        tau = kendall_tau(pred['stop_id'].tolist(), true_sorted['stop_id'].tolist())
    print(f'[learned] route {route_id}: dist_km={dist_km:.2f}' + (f'  kendall_tau={tau:.3f}' if tau is not None else ''))
    return route_id

def run(train_csv: Optional[str], test_csv: Optional[str], route_id=None,
        json_root_train: Optional[str]=None, split_train: str="train",
        json_root_test: Optional[str]=None, split_test: str="eval"):
    if train_csv and test_csv:
        train = pd.read_csv(train_csv)
        test = pd.read_csv(test_csv)
        run_with_dataframes(train, test, route_id)
        return
    if json_root_train and json_root_test:
        from data_loader import load_amzl_dataset
        train = load_amzl_dataset(json_root_train, split=split_train)
        test = load_amzl_dataset(json_root_test, split=split_test)
        run_with_dataframes(train, test, route_id)
        return
    raise ValueError("Provide either --train/--test CSVs or both JSON roots via --json_root_train and --json_root_test")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=False)
    ap.add_argument("--test", required=False)
    ap.add_argument("--json_root_train", required=False, help="Path containing almrrc2021-data-<split>/ for training")
    ap.add_argument("--split_train", default="train", choices=["train","eval"])
    ap.add_argument("--json_root_test", required=False, help="Path containing almrrc2021-data-<split>/ for testing")
    ap.add_argument("--split_test", default="eval", choices=["train","eval"])
    ap.add_argument("--route_id", default=None)
    args = ap.parse_args()
    run(args.train, args.test, args.route_id, args.json_root_train, args.split_train, args.json_root_test, args.split_test)
