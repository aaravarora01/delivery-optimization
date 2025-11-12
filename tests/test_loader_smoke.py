from pathlib import Path

from data_loader import load_amzl_dataset

def test_loader_smoke():
    root = "cs230_data"
    df = load_amzl_dataset(root, split="train")
    assert set(["route_id","stop_id","lat","lon","seq"]).issubset(df.columns)
    assert len(df) > 0
    # basic type/dtype checks
    assert df["route_id"].dtype == object
    assert df["stop_id"].dtype == object
    assert df["lat"].dtype.kind == "f"
    assert df["lon"].dtype.kind == "f"
    assert df["seq"].dtype.kind in ("i", "u")


