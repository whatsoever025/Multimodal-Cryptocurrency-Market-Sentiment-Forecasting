"""
Shared metrics for baseline evaluation.
Mirrors _compute_metrics() in src/training/train.py exactly.
"""

import numpy as np
from typing import Optional, Dict


TARGET_NAMES = ["y_baseline", "y_heuristic", "y_vol_adj_return"]


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_name: Optional[str] = None,
    train_targets_mean: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R², R²_OOS with target-specific benchmark.

    R²_OOS benchmark selection (mirrors train.py):
        y_baseline        → Historical Mean  (structural positive bias; mean ≠ 0)
        y_heuristic       → Zero-predictor   (GKX 2020; mean ≈ 0 after Z-scoring)
        y_vol_adj_return  → Zero-predictor   (GKX 2020; log-return mean ≈ 0)
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    mae = float(np.mean(np.abs(predictions - targets)))
    mse = float(np.mean((predictions - targets) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((predictions - targets) ** 2))

    # Standard R²  (benchmark = test-set mean)
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # R²_OOS  — GKX 2020: zero-predictor
    ss_tot_gkx = float(np.sum(targets ** 2))
    r2_oos_gkx = 1.0 - ss_res / ss_tot_gkx if ss_tot_gkx > 0 else 0.0

    # R²_OOS  — Historical Mean benchmark
    if train_targets_mean is not None:
        ss_tot_hist = float(np.sum((targets - train_targets_mean) ** 2))
        r2_oos_hist = 1.0 - ss_res / ss_tot_hist if ss_tot_hist > 0 else 0.0
    else:
        r2_oos_hist = 0.0

    # Select primary R²_OOS per target
    if target_name == "y_baseline":
        r2_oos = r2_oos_hist
    else:
        r2_oos = r2_oos_gkx

    # Pearson correlation
    if len(predictions) > 1 and predictions.std() > 0 and targets.std() > 0:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
    else:
        correlation = 0.0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "r2_oos": r2_oos,
        "r2_oos_gkx": r2_oos_gkx,
        "r2_oos_hist": r2_oos_hist,
        "correlation": correlation,
        "prediction_bias": float(np.mean(predictions - targets)),
        "n_samples": int(len(targets)),
    }
