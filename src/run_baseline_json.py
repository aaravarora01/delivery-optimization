import argparse

import pandas as pd
from data_loader import load_amzl_dataset
from baseline import solve_route, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_root", required=True, help="Path containing almrrc2021-data-<split>/")
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--route_id", default=None)
    ap.add_argument("--return_to_start", action="store_true")
    args = ap.parse_args()

    df = load_amzl_dataset(args.json_root, split=args.split)
    rid = args.route_id or df["route_id"].iloc[0]
    rid = str(rid)

    r = df[df.route_id == rid].copy().reset_index(drop=True)
    pred, dist_km = solve_route(r[["stop_id","lat","lon"]], return_to_start=args.return_to_start)
    tau = evaluate(pred[["stop_id"]], r[["stop_id","seq"]])

    out = f"./outputs/route_{rid}_baseline.csv"
    pred.to_csv(out, index=False)
    print(f"Route {rid}: distance_km={dist_km:.2f}  kendall_tau={tau:.3f}  -> {out}")

if __name__ == "__main__":
    main()


