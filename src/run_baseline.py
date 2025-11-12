import pandas as pd
from baseline import solve_route, evaluate

# Quick assumptions:
# data file has columns: route_id, stop_id, lat, lon, (optional) seq
# We’ll run on one route by id, or the first route in the file.

def run(input_csv: str, route_id=None, return_to_start=False):
    df = pd.read_csv(input_csv)
    if route_id is None:
        route_id = df['route_id'].iloc[0]
    df_r = df[df['route_id'] == route_id].copy()

    pred, dist_km = solve_route(df_r, return_to_start=return_to_start)
    tau = evaluate(pred[['stop_id']], df_r[['stop_id','seq']] if 'seq' in df_r.columns else df_r[['stop_id']])
    pred.to_csv(f'./outputs/route_{route_id}_baseline.csv', index=False)
    print(f'Route {route_id}: distance_km={dist_km:.2f}', end='')
    if tau is not None:
        print(f'  kendall_tau_vs_driver={tau:.3f}')
    else:
        print()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--route_id", default=None)
    ap.add_argument("--return_to_start", action="store_true")
    args = ap.parse_args()
    run(args.input, args.route_id, args.return_to_start)