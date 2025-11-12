import json

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import pandas as pd

def _first_key(d: Dict[str, Any], keys: Iterable[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

def _get_lat_lon(rec: Dict[str, Any]) -> Tuple[float, float]:
    lat = _first_key(rec, ["lat", "latitude"])
    lon = _first_key(rec, ["lon", "lng", "longitude"])
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    loc = rec.get("location")
    if isinstance(loc, dict):
        lat = _first_key(loc, ["lat", "latitude"])
        lon = _first_key(loc, ["lon", "lng", "longitude"])
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    raise KeyError("Missing coordinates: expected 'lat/lon' or 'location.{lat,lng}'")

def _get_stop_id(rec: Dict[str, Any]) -> str:
    sid = _first_key(rec, ["stop_id", "package_id", "consignment_id", "id"])
    if sid is None:
        raise KeyError("Missing stop identifier: expected one of stop_id/package_id/consignment_id/id")
    return str(sid)

def _iter_packages(package_data: Any) -> List[Dict[str, Any]]:
    if isinstance(package_data, list):
        return package_data
    if isinstance(package_data, dict):
        # Common: {'packages': [...]} or {'data': [...]}
        for key in ("packages", "data"):
            if key in package_data and isinstance(package_data[key], list):
                return package_data[key]
        # Fallback: keyed by route_id → list[records]
        flat: List[Dict[str, Any]] = []
        for rid, lst in package_data.items():
            if isinstance(lst, list):
                for rec in lst:
                    rec = {**rec, "route_id": rid}
                    flat.append(rec)
            elif isinstance(lst, dict):
                # Handle nested structure: {RouteID: {ZoneID: {PackageID: {...}}}}
                # ZoneID is the stop identifier
                for zone_id, packages in lst.items():
                    if isinstance(packages, dict):
                        # Create a record for this zone (stop)
                        rec = {"route_id": rid, "stop_id": zone_id}
                        flat.append(rec)
        if flat:
            return flat
    raise ValueError("Unexpected format for package_data.json")

def _iter_sequences(actual_sequences: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(actual_sequences, list):
        # [{'route_id': '...', 'sequence': ['sid1', ...]}]
        for obj in actual_sequences:
            rid = str(_first_key(obj, ["route_id", "routeId", "routeID"]))
            seq_list = _first_key(obj, ["sequence", "actual_sequence", "stops"])
            if isinstance(seq_list, list):
                for idx, sid in enumerate(seq_list, start=1):
                    rows.append({"route_id": rid, "stop_id": str(sid), "seq": idx})
        return rows
    if isinstance(actual_sequences, dict):
        # Handle: {'<route_id>': ['sid1', ...]}  (simple dict)
        # OR: {'<route_id>': {'actual': {'ZoneID': seq_num, ...}}}  (nested with 'actual')
        for rid, route_seq in actual_sequences.items():
            if isinstance(route_seq, list):
                # Simple list format
                for idx, sid in enumerate(route_seq, start=1):
                    rows.append({"route_id": str(rid), "stop_id": str(sid), "seq": idx})
            elif isinstance(route_seq, dict):
                # Check if it has an 'actual' key with zone->seq mapping
                actual = route_seq.get("actual")
                if isinstance(actual, dict):
                    # Format: {RouteID: {actual: {ZoneID: seq_num}}}
                    for zone_id, seq_num in actual.items():
                        rows.append({"route_id": str(rid), "stop_id": str(zone_id), "seq": int(seq_num)})
                else:
                    # Fallback: treat as {ZoneID: seq_num} directly
                    for zone_id, seq_num in route_seq.items():
                        if isinstance(seq_num, (int, float)):
                            rows.append({"route_id": str(rid), "stop_id": str(zone_id), "seq": int(seq_num)})
        return rows
    raise ValueError("Unexpected format for actual_sequences.json")

def load_amzl_dataset(root: str, split: str = "train") -> pd.DataFrame:
    """
    Load Amazon Last Mile dataset JSON and return a normalized DataFrame with:
    columns = ['route_id', 'stop_id', 'lat', 'lon', 'seq'].
    """
    base = Path(root)
    
    if split == "train":
        route_path = base / "almrrc2021-data-training" / "model_build_inputs" / "route_data.json"
        package_path = base / "almrrc2021-data-training" / "model_build_inputs" / "package_data.json"
        sequences_path = base / "almrrc2021-data-training" / "model_build_inputs" / "actual_sequences.json"
    elif split == "eval":
        route_path = base / "almrrc2021-data-evaluation" / "model_apply_inputs" / "eval_route_data.json"
        package_path = base / "almrrc2021-data-evaluation" / "model_apply_inputs" / "eval_package_data.json"
        sequences_path = base / "almrrc2021-data-evaluation" / "model_score_inputs" / "eval_actual_sequences.json"
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train' or 'eval'")

    def jload(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")
        with open(path, "r") as f:
            return json.load(f)

    route_data = jload(route_path)
    package_data = jload(package_path)
    actual_sequences = jload(sequences_path)

    # Build DataFrame from route_data (has coordinates) and package_data (has zone mapping)
    pkg_rows = []
    for route_id, route_info in route_data.items():
        if not isinstance(route_info, dict):
            continue
        stops = route_info.get("stops", {})
        if not isinstance(stops, dict):
            continue
        
        for zone_id, stop_info in stops.items():
            if not isinstance(stop_info, dict):
                continue
            lat = _first_key(stop_info, ["lat", "latitude"])
            lon = _first_key(stop_info, ["lon", "lng", "longitude"])
            if lat is not None and lon is not None:
                pkg_rows.append({
                    "route_id": route_id,
                    "stop_id": zone_id,
                    "lat": float(lat),
                    "lon": float(lon)
                })

    df_pkg = pd.DataFrame(pkg_rows)
    seq_rows = _iter_sequences(actual_sequences)
    df_seq = pd.DataFrame(seq_rows)
    
    # Merge on route_id and stop_id (zone_id)
    df = df_pkg.merge(df_seq, on=["route_id", "stop_id"], how="inner")

    needed = ["route_id", "stop_id", "lat", "lon", "seq"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after load: {missing}")

    df["route_id"] = df["route_id"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["seq"] = df["seq"].astype(int)
    return df