# src/utils/metrics.py

import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Optional
from baseline import pairwise_distance_matrix, tour_length, kendall_tau

def compute_sequence_accuracy(pred_order: List[int], true_order: List[int]) -> float:
    """Compute exact sequence match accuracy."""
    if len(pred_order) != len(true_order):
        return 0.0
    return 1.0 if pred_order == true_order else 0.0

def compute_position_accuracy(pred_order: List[int], true_order: List[int], k: int = 1) -> float:
    """Compute accuracy of predicting true position within k positions."""
    if len(pred_order) != len(true_order):
        return 0.0
    
    pos_pred = {item: idx for idx, item in enumerate(pred_order)}
    correct = 0
    
    for idx, true_item in enumerate(true_order):
        pred_pos = pos_pred[true_item]
        if abs(pred_pos - idx) <= k:
            correct += 1
    
    return correct / len(true_order)

def compute_tour_distance(coords: np.ndarray, order: List[int]) -> float:
    """Compute total tour distance in km."""
    D = pairwise_distance_matrix(coords)
    return tour_length(D, order)

def evaluate_predictions(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    coords_cols: Tuple[str, str] = ('lat', 'lon')
) -> Dict[str, float]:
    """
    Comprehensive evaluation of predictions.
    
    Args:
        pred_df: DataFrame with 'stop_id' and predicted order
        true_df: DataFrame with 'stop_id' and 'seq' (true order)
        coords_cols: Tuple of (lat_col, lon_col) names
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Get orders
    true_sorted = true_df.sort_values('seq')
    pred_order = pred_df['stop_id'].tolist()
    true_order = true_sorted['stop_id'].tolist()
    
    # Sequence accuracy (exact match)
    metrics['sequence_accuracy'] = compute_sequence_accuracy(pred_order, true_order)
    
    # Kendall tau correlation
    metrics['kendall_tau'] = kendall_tau(pred_order, true_order)
    
    # Position accuracy at different thresholds
    metrics['position_acc_k1'] = compute_position_accuracy(pred_order, true_order, k=1)
    metrics['position_acc_k3'] = compute_position_accuracy(pred_order, true_order, k=3)
    metrics['position_acc_k5'] = compute_position_accuracy(pred_order, true_order, k=5)
    
    # Tour distances
    coords = pred_df[list(coords_cols)].to_numpy()
    pred_tour = list(range(len(pred_order)))
    metrics['predicted_tour_distance_km'] = compute_tour_distance(coords, pred_tour)
    
    # True tour distance (using true sequence)
    true_coords = true_sorted[list(coords_cols)].to_numpy()
    true_tour = list(range(len(true_order)))
    metrics['true_tour_distance_km'] = compute_tour_distance(true_coords, true_tour)
    
    # Distance ratio (predicted / true)
    if metrics['true_tour_distance_km'] > 0:
        metrics['distance_ratio'] = metrics['predicted_tour_distance_km'] / metrics['true_tour_distance_km']
    else:
        metrics['distance_ratio'] = np.nan
    
    # Mean absolute position error
    pos_pred = {item: idx for idx, item in enumerate(pred_order)}
    pos_errors = []
    for idx, true_item in enumerate(true_order):
        pred_pos = pos_pred[true_item]
        pos_errors.append(abs(pred_pos - idx))
    metrics['mean_abs_position_error'] = np.mean(pos_errors)
    metrics['median_abs_position_error'] = np.median(pos_errors)
    
    return metrics

def evaluate_zone_predictions(
    model,
    gnn,
    zone_df: pd.DataFrame,
    device: str,
    use_gnn: bool = False,
    greedy: bool = True
) -> Dict[str, float]:
    """Evaluate predictions for a single zone."""
    from utils.features import node_features, edge_bias_features
    from utils.knn_graph import knn_adj
    
    # Reset index to ensure proper indexing
    zone_df = zone_df.reset_index(drop=True).copy()
    if len(zone_df) < 2:
        raise ValueError("Zone has fewer than 2 stops; cannot evaluate.")
    
    coords = torch.tensor(zone_df[['lat','lon']].to_numpy(), dtype=torch.float32, device=device).unsqueeze(0)
    
    # Get features
    X = node_features(coords)
    if use_gnn and gnn is not None:
        adj = knn_adj(coords, k=min(8, coords.shape[1]-1))
        X = gnn(X, adj.to(device))
    
    edge_feats = edge_bias_features(coords)
    
    # Decode
    with torch.no_grad():
        pred_order_indices = model.greedy_decode(X, edge_feats=edge_feats)
        # Debug: check shape before reshape
        print(f"DEBUG: pred_order_indices shape before reshape: {pred_order_indices.shape}, zone size: {len(zone_df)}")
        pred_order_indices = pred_order_indices.reshape(-1).cpu().numpy().tolist()
        print(f"DEBUG: pred_order_indices length after reshape: {len(pred_order_indices)}")
    
    # Validate indices are in range
    num_stops = len(zone_df)
    if len(pred_order_indices) != num_stops:
        raise ValueError(f"Predicted sequence length ({len(pred_order_indices)}) doesn't match zone size ({num_stops})")
    
    # Check all indices are valid
    invalid_indices = [i for i in pred_order_indices if i < 0 or i >= num_stops]
    if invalid_indices:
        raise ValueError(f"Invalid indices in prediction: {invalid_indices[:5]}... (zone size: {num_stops})")
    
    # Get stop_ids in predicted and true order
    stop_ids = zone_df['stop_id'].tolist()
    pred_stop_ids = [stop_ids[i] for i in pred_order_indices]
    
    # Get true order (by sequence)
    true_sorted = zone_df.sort_values('seq').reset_index(drop=True)
    true_stop_ids = true_sorted['stop_id'].tolist()
    
    # Create dataframes for evaluation - reorder by stop_id lists
    # For predicted order
    pred_df = pd.DataFrame({'stop_id': pred_stop_ids})
    pred_df = pred_df.merge(zone_df[['stop_id', 'lat', 'lon']], on='stop_id', how='left')
    
    # For true order
    true_df = true_sorted[['stop_id', 'lat', 'lon', 'seq']].copy()
    
    return evaluate_predictions(pred_df, true_df)

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across multiple zones/routes."""
    if not metrics_list:
        return {}
    
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if not np.isnan(m.get(key, np.nan))]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_median'] = np.median(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
    
    return aggregated

