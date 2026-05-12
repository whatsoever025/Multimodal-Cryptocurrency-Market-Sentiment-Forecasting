"""
Regime Shift Diagnostic Script
================================
Runs 3 independent tests to confirm whether test R² < 0 is caused by
market regime shift rather than model bugs.

Usage (Kaggle or local):
    python -m src.training.diagnose_regime_shift \
        --features-dir /kaggle/input/crypto-sentiment-embeddings \
        --checkpoint  /kaggle/working/checkpoints/best_model.pt

Tests performed:
    1. Rolling R²   - R² computed on 30-day sliding windows over test period
    2. Feature drift - KS test on each tabular feature (train vs test distribution)
    3. Error drift   - Mean prediction error over time (systematic bias → regime shift)
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

logger = logging.getLogger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────────

def r2_oos(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Gu, Kelly & Xiu (2020) R²_OOS = 1 - SS_res / SS_tot (demeaned by zero, not sample mean)."""
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum(targets ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def r2_standard(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Standard R² = 1 - SS_res / Var(targets)."""
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ─── Test 1: Rolling R² over test period ──────────────────────────────────────

def test_rolling_r2(predictions: np.ndarray, targets: np.ndarray,
                    window_hours: int = 720) -> dict:
    """
    Compute R² on sliding windows across the test period.

    Regime shift signature:
        R² starts near val level then DROPS sharply at a specific date
        (e.g., Bitcoin halving Apr 2024, ETF inflows Jan 2024).

    Args:
        predictions: model predictions on test set (N,)
        targets:     ground truth on test set (N,)
        window_hours: window size in hours (default 720 = ~30 days)

    Returns:
        dict with window-level R² values and shift detection result
    """
    N = len(predictions)
    step = window_hours // 4  # 75% overlap

    windows = []
    r2_values = []

    for start in range(0, N - window_hours, step):
        end = start + window_hours
        p = predictions[start:end]
        t = targets[start:end]
        r2 = r2_standard(p, t)
        windows.append((start, end))
        r2_values.append(r2)

    r2_arr = np.array(r2_values)
    first_half_mean = r2_arr[: len(r2_arr) // 2].mean()
    second_half_mean = r2_arr[len(r2_arr) // 2 :].mean()
    shift_detected = (first_half_mean - second_half_mean) > 0.05  # > 5pp drop

    print("\n" + "=" * 60)
    print("TEST 1: Rolling R² over test period")
    print("=" * 60)
    for i, ((s, e), r2) in enumerate(zip(windows, r2_values)):
        bar = "█" * max(0, int((r2 + 0.3) * 20))
        print(f"  Window {i+1:2d}  [{s:5d}:{e:5d}]  R²={r2:+.4f}  {bar}")
    print(f"\n  First-half avg R²: {first_half_mean:+.4f}")
    print(f"  Second-half avg R²: {second_half_mean:+.4f}")
    print(f"  Regime shift detected: {'YES ⚠' if shift_detected else 'NO ✓'}")

    return {
        "windows": windows,
        "r2_values": r2_values,
        "first_half_mean": first_half_mean,
        "second_half_mean": second_half_mean,
        "shift_detected": shift_detected,
    }


# ─── Test 2: Feature distribution drift (KS test) ─────────────────────────────

def test_feature_drift(tabular_train: np.ndarray, tabular_test: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov test on each tabular feature: train vs test distribution.

    Regime shift signature:
        High KS statistic + low p-value on price/volume/sentiment features
        means train and test come from different distributions.

    Features: return_1h, volume, funding_rate, fear_greed_value,
              gdelt_econ_volume, gdelt_econ_tone, gdelt_conflict_volume
    """
    feature_names = [
        "return_1h", "volume", "funding_rate", "fear_greed_value",
        "gdelt_econ_volume", "gdelt_econ_tone", "gdelt_conflict_volume",
    ]

    print("\n" + "=" * 60)
    print("TEST 2: Feature distribution drift (KS test, train vs test)")
    print("=" * 60)
    print(f"  {'Feature':<25} {'KS stat':>8}  {'p-value':>10}  {'Drifted?':>10}")
    print("  " + "-" * 58)

    drifted_features = []
    results = {}

    for i, name in enumerate(feature_names):
        tr = tabular_train[:, i]
        te = tabular_test[:, i]
        ks_stat, p_val = stats.ks_2samp(tr, te)
        drifted = p_val < 0.01  # 1% significance
        flag = "YES ⚠" if drifted else "no"
        print(f"  {name:<25} {ks_stat:>8.4f}  {p_val:>10.2e}  {flag:>10}")
        results[name] = {"ks_stat": ks_stat, "p_value": p_val, "drifted": drifted}
        if drifted:
            drifted_features.append(name)

    print(f"\n  Drifted features ({len(drifted_features)}/{len(feature_names)}): "
          f"{', '.join(drifted_features) if drifted_features else 'none'}")
    overall_drift = len(drifted_features) >= 3

    print(f"  Overall regime shift conclusion: {'YES ⚠' if overall_drift else 'NO ✓'}")
    return {"features": results, "drifted_features": drifted_features,
            "overall_drift": overall_drift}


# ─── Test 3: Prediction error drift over time ──────────────────────────────────

def test_error_drift(predictions: np.ndarray, targets: np.ndarray,
                     chunk_size: int = 500) -> dict:
    """
    Check if prediction errors are stationary or have a systematic trend.

    Regime shift signature:
        Error mean drifts consistently in one direction → model's learned
        relationship no longer holds in the test period.
    """
    N = len(predictions)
    errors = predictions - targets
    n_chunks = N // chunk_size

    chunk_means = []
    for i in range(n_chunks):
        chunk = errors[i * chunk_size : (i + 1) * chunk_size]
        chunk_means.append(chunk.mean())

    chunk_means = np.array(chunk_means)
    # Mann-Kendall trend test (monotone drift detection)
    n = len(chunk_means)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(chunk_means[j] - chunk_means[i])
    # Variance under H0
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z_mk = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else 0
    trend_detected = abs(z_mk) > 1.96  # 95% confidence

    print("\n" + "=" * 60)
    print("TEST 3: Prediction error drift over test period")
    print("=" * 60)
    print(f"  {'Chunk':>6}  {'Error mean':>12}  {'Trend bar'}")
    print("  " + "-" * 50)
    for i, m in enumerate(chunk_means):
        bar_pos = "+" * max(0, int(m * 20)) if m > 0 else ""
        bar_neg = "-" * max(0, int(-m * 20)) if m < 0 else ""
        print(f"  {i+1:>6}  {m:>+12.4f}  {bar_neg}{bar_pos}")

    print(f"\n  Mann-Kendall Z-stat: {z_mk:+.3f}  (|Z| > 1.96 → trend)")
    print(f"  Error drift detected: {'YES ⚠' if trend_detected else 'NO ✓'}")

    return {
        "chunk_means": chunk_means.tolist(),
        "mk_z": float(z_mk),
        "trend_detected": trend_detected,
    }


# ─── Null model baselines ──────────────────────────────────────────────────────

def null_model_baselines(targets: np.ndarray) -> None:
    """
    Compare against trivial baselines.
    If even these fail on test, the problem is definitely the data, not the model.
    """
    zero_pred = np.zeros_like(targets)
    mean_pred = np.full_like(targets, targets.mean())
    rolling_mean = np.array([
        targets[max(0, i - 24) : i].mean() if i > 0 else 0
        for i in range(len(targets))
    ])

    print("\n" + "=" * 60)
    print("NULL MODEL BASELINES on test set")
    print("=" * 60)
    print(f"  Predict zero (random walk):   R²={r2_standard(zero_pred, targets):+.4f}  "
          f"R²_OOS={r2_oos(zero_pred, targets):+.4f}")
    print(f"  Predict test mean:            R²={r2_standard(mean_pred, targets):+.4f}  "
          f"R²_OOS={r2_oos(mean_pred, targets):+.4f}")
    print(f"  Predict 24h rolling mean:     R²={r2_standard(rolling_mean, targets):+.4f}  "
          f"R²_OOS={r2_oos(rolling_mean, targets):+.4f}")
    print()
    print("  Interpretation:")
    print("  - If 'predict zero' R² ≈ -0.15, BASELINE also fails → regime shift confirmed")
    print("  - If 'predict zero' R² > model R², model adds negative value on test")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    features_dir = Path(args.features_dir)

    # Load metadata
    meta_path = features_dir / "split_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"split_metadata.json not found at {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    train_end = meta["train_end_idx"]
    val_end   = meta["val_end_idx"]
    test_end  = meta["test_end_idx"]

    print("\n" + "=" * 60)
    print("REGIME SHIFT DIAGNOSTIC")
    print("=" * 60)
    print(f"  Train:  [0 : {train_end}]  ({train_end} samples)")
    print(f"  Val:    [{train_end} : {val_end}]  ({val_end - train_end} samples)")
    print(f"  Test:   [{val_end} : {test_end}]  ({test_end - val_end} samples)")

    # Load raw tabular features for distribution drift test
    tabular = torch.load(features_dir / "tabular_features.pt", map_location="cpu").numpy()
    tabular_train = tabular[:train_end]
    tabular_test  = tabular[val_end:test_end]

    # ── Load model predictions if checkpoint provided ──────────────────────
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\nLoading model predictions from checkpoint: {args.checkpoint}")
        _run_model_diagnostics(args, meta, tabular, features_dir)
    else:
        print("\n⚠ No checkpoint provided — skipping rolling R² and error drift tests.")
        print("  Provide --checkpoint path to run all 3 tests.")
        print("  Running feature drift test only.\n")
        test_feature_drift(tabular_train, tabular_test)
        print("\n" + "=" * 60)
        print("HOW TO INTERPRET & OVERCOME REGIME SHIFT")
        _print_recommendations()
        return

    print("\n" + "=" * 60)
    print("HOW TO INTERPRET & OVERCOME REGIME SHIFT")
    _print_recommendations()


def _run_model_diagnostics(args, meta, tabular, features_dir):
    """Run all 3 tests using a saved checkpoint."""
    from .model import MultimodalFusionNet
    from .config import ExperimentConfig
    from sklearn.preprocessing import StandardScaler, RobustScaler

    train_end = meta["train_end_idx"]
    val_end   = meta["val_end_idx"]
    test_end  = meta["test_end_idx"]

    # Load all tensors
    text_emb   = torch.load(features_dir / "text_embeddings.pt",  map_location="cpu")
    image_emb  = torch.load(features_dir / "image_embeddings.pt", map_location="cpu")
    tabular_t  = torch.load(features_dir / "tabular_features.pt", map_location="cpu")
    targets_t  = torch.load(features_dir / "target_scores.pt",    map_location="cpu")

    # Scale using training statistics
    tab_scaler = StandardScaler()
    tab_scaler.fit(tabular_t[:train_end].numpy())
    tabular_scaled = torch.from_numpy(tab_scaler.transform(tabular_t.numpy())).float()

    tgt_scaler = RobustScaler()
    tgt_scaler.fit(targets_t[:train_end].numpy().reshape(-1, 1))
    targets_scaled = torch.from_numpy(
        tgt_scaler.transform(targets_t.numpy().reshape(-1, 1)).squeeze(axis=1)
    ).float()

    # Load model
    config = ExperimentConfig()
    model = MultimodalFusionNet(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Inference on test set (sliding windows)
    seq_len = config.data.seq_len
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for idx in range(val_end, test_end - seq_len):
            batch = {
                "tabular":         tabular_scaled[idx:idx + seq_len].unsqueeze(0),
                "text_embedding":  text_emb[idx:idx + seq_len].unsqueeze(0),
                "image_embedding": image_emb[idx:idx + seq_len].unsqueeze(0),
            }
            pred = model(batch).item()
            tgt  = targets_scaled[idx + seq_len].item()
            test_predictions.append(pred)
            test_targets.append(tgt)

    pred_arr = np.array(test_predictions)
    tgt_arr  = np.array(test_targets)

    print(f"\n  Test inference complete: {len(pred_arr)} samples")
    print(f"  Test R² (standard): {r2_standard(pred_arr, tgt_arr):+.4f}")
    print(f"  Test R²_OOS:        {r2_oos(pred_arr, tgt_arr):+.4f}")

    # Run all 3 tests
    tabular_np = tabular_t.numpy()
    test_feature_drift(tabular_np[:train_end], tabular_np[val_end:test_end])
    test_rolling_r2(pred_arr, tgt_arr, window_hours=720)
    test_error_drift(pred_arr, tgt_arr)
    null_model_baselines(tgt_arr)


def _print_recommendations():
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  REGIME SHIFT CONFIRMED IF:                             │
  │  • Test 1: R² drops sharply in second half of test     │
  │  • Test 2: ≥3 features show significant drift (p<0.01) │
  │  • Test 3: Error mean drifts monotonically             │
  │  • Null baselines also show negative R² on test        │
  └─────────────────────────────────────────────────────────┘

  HOW TO OVERCOME:

  1. ROLLING NORMALIZATION (recommended, low effort)
     Normalize tabular features and targets relative to a
     rolling 30-day window instead of fixed train statistics.
     Makes the model regime-agnostic by construction.
     → Edit dataset.py: replace StandardScaler/RobustScaler
       with per-sample z-score using a rolling window.

  2. ADD REGIME INDICATOR FEATURES (medium effort)
     Add features that encode the current market regime:
     • 30d momentum (are we trending up or down?)
     • Realized volatility (30d rolling std of returns)
     • BTC dominance (institutional vs retail market)
     These let the model self-identify the regime.

  3. ONLINE FINE-TUNING (medium effort, most realistic)
     As test data arrives sequentially, fine-tune the model
     on the most recent N days before predicting the next day.
     This is how production ML systems work in finance.

  4. EXPAND TRAINING DATA (high effort)
     Include the post-ETF/halving period in training by
     shifting the test window to a future date, or by
     collecting data from previous halving cycles (2016, 2020).

  5. REFRAME AS REGIME-CONDITIONAL (research contribution)
     Detect regimes with an HMM, train a model per regime,
     route predictions through the active regime model.
     Good paper contribution if Test 2 confirms feature drift.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regime shift diagnostic")
    parser.add_argument("--features-dir", default="./data/features",
                        help="Path to .pt feature files")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to saved model checkpoint (.pt)")
    args = parser.parse_args()
    main(args)
