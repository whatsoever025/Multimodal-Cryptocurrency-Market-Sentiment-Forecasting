"""
Run all baseline models with the exact same walk-forward protocol as the
main MultimodalFusionNet training pipeline.

Mirrors:
    - Data loading:       src/training/train.py  (main)
    - Target engineering: src/training/dataset.py (apply_target_engineering)
    - Walk-forward split: src/training/dataset.py (walk_forward_split)
    - Valid start logic:  src/training/dataset.py (WalkForwardDataset)
    - Metrics:            src/training/train.py  (_compute_metrics)

Usage:
    python -m src.baseline.run_baselines
    python -m src.baseline.run_baselines --features-dir ./data/features --num-folds 5
    python -m src.baseline.run_baselines --features-dir ./data/features --seed 42 --out-dir ./src/baseline/results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .metrics import compute_metrics, TARGET_NAMES
from .models import get_all_models, HistoricalMeanModel, LinearRegressionModel, XGBoostModel

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ─── Constants (must match src/training/config.py & dataset.py) ──────────────
SEQ_LEN      = 24
N_TABULAR    = 7
BUFFER       = 8       # t+8 target offset for y_baseline
TEST_PCT     = 0.15    # last 15% = hold-out test set
INIT_WIN_PCT = 0.70    # initial training window fraction
VAL_FOLD_PCT = 0.15    # validation pool fraction


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_data(features_dir: Path):
    """Load tabular features and targets for BTC and ETH."""
    btc_dir = features_dir / "BTC"
    eth_dir = features_dir / "ETH"

    if not (btc_dir.exists() and eth_dir.exists()):
        raise FileNotFoundError(
            f"Expected BTC/ and ETH/ subdirs inside {features_dir}. "
            "Run src/training/extract_features.py first."
        )

    btc_tab = torch.load(btc_dir / "tabular_features.pt", map_location="cpu").numpy()
    btc_tgt = torch.load(btc_dir / "target_scores.pt",    map_location="cpu").numpy()
    eth_tab = torch.load(eth_dir / "tabular_features.pt", map_location="cpu").numpy()
    eth_tgt = torch.load(eth_dir / "target_scores.pt",    map_location="cpu").numpy()

    tabular = np.concatenate([btc_tab, eth_tab], axis=0).astype(np.float32)  # (N, 7)
    targets = np.concatenate([btc_tgt, eth_tgt], axis=0).astype(np.float32)  # (N, 3)

    logger.info(
        f"Loaded: BTC={btc_tab.shape[0]}, ETH={eth_tab.shape[0]}, "
        f"total={tabular.shape[0]} samples"
    )
    return tabular, targets, btc_tab.shape[0], eth_tab.shape[0]


def apply_target_engineering(targets: np.ndarray) -> np.ndarray:
    """
    Mirror dataset.py target engineering (applied once, before any scaling):
        col 0 (y_baseline):       × 1000   — prevents underflow
        col 1 (y_heuristic):      clip to [-5, 5]
        col 2 (y_vol_adj_return): unchanged
    """
    t = targets.copy()
    t[:, 0] *= 1000.0
    t[:, 1] = np.clip(t[:, 1], -5.0, 5.0)
    return t


def walk_forward_split(data_len: int, window_size: int, step_size: int):
    """Identical to dataset.py walk_forward_split."""
    for i in range(0, data_len - window_size - step_size, step_size):
        train_end = i + window_size
        val_end   = min(train_end + step_size, data_len)
        yield slice(0, train_end), slice(train_end, val_end)


def get_valid_starts(data_slice: slice, btc_len: int, eth_len: int,
                     seq_len: int = SEQ_LEN, buffer: int = BUFFER):
    """
    Identical boundary logic to WalkForwardDataset.__init__:
        - Never cross asset boundary
        - Leave buffer=8 at end to safely access t+7 target index
    """
    start = data_slice.start or 0
    stop  = data_slice.stop

    btc_valid = list(range(
        start,
        min(stop - seq_len - buffer + 1,
            btc_len - seq_len - buffer + 1)
    ))
    eth_valid = list(range(
        start + btc_len,
        min(stop - seq_len - buffer + 1 + btc_len,
            btc_len + eth_len - seq_len - buffer + 1)
    ))
    return btc_valid + eth_valid


def build_Xy(tabular_scaled: np.ndarray, targets_eng: np.ndarray,
             valid_starts: list, target_idx: int,
             seq_len: int = SEQ_LEN) -> tuple:
    """
    Build feature matrix X and target vector y.
        X: (n, seq_len * N_TABULAR)  — flattened 24-step window
        y: (n,)                      — engineered target (not scaled)
    """
    X, y = [], []
    for i in valid_starts:
        X.append(tabular_scaled[i: i + seq_len].flatten())
        if target_idx == 0:
            y.append(targets_eng[i + seq_len + 7, 0])   # y_baseline: t+8
        elif target_idx == 1:
            y.append(targets_eng[i + seq_len, 1])        # y_heuristic: t+1
        else:
            y.append(targets_eng[i + seq_len, 2])        # y_vol_adj_return: t+1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─── Per-asset StandardScaler (mirrors dataset.py) ───────────────────────────

def fit_scale_tabular(tabular_raw: np.ndarray, btc_len: int, eth_len: int,
                      train_slice: slice) -> np.ndarray:
    """
    Fit StandardScaler independently per asset on train_slice.
    Apply to all samples (train + val + test) to prevent leakage.
    Mirrors create_walk_forward_dataloaders() scaler logic.
    """
    btc_train = slice(train_slice.start or 0, train_slice.stop)
    eth_train = slice((train_slice.start or 0) + btc_len, train_slice.stop + btc_len)

    btc_scaler = StandardScaler().fit(tabular_raw[btc_train])
    eth_scaler = StandardScaler().fit(tabular_raw[eth_train])

    scaled = tabular_raw.copy()
    scaled[:btc_len]              = btc_scaler.transform(tabular_raw[:btc_len])
    scaled[btc_len: btc_len + eth_len] = eth_scaler.transform(tabular_raw[btc_len: btc_len + eth_len])
    return scaled


# ─── Main run loop ────────────────────────────────────────────────────────────

def run_baselines(features_dir: str = "./data/features",
                  num_folds: int = 5,
                  seed: int = 42,
                  out_dir: str = "./src/baseline/results") -> dict:

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    tabular_raw, targets_raw, btc_len, eth_len = load_data(Path(features_dir))
    targets_eng = apply_target_engineering(targets_raw)

    # ── Walk-forward split parameters (must match dataset.py) ──
    data_len    = int(btc_len * (1.0 - TEST_PCT))
    window_size = int(INIT_WIN_PCT * data_len)
    step_size   = int(VAL_FOLD_PCT * data_len) // num_folds

    test_start  = data_len   # per-asset index where test begins
    logger.info(
        f"Walk-forward config: data_len={data_len}, window={window_size}, "
        f"step={step_size}, folds={num_folds}, test=[{test_start}:{btc_len}]"
    )

    # ── Build test set valid starts (fixed; same for all models/folds) ──
    test_slice       = slice(test_start, btc_len)
    # Scale with the full train slice for test (use last fold's full train data)
    # This will be updated inside the fold loop
    test_tabular_scaled_cache = None
    test_starts_cache         = None

    all_results = {}  # { model_name: { target_name: { fold_results, test_metrics } } }

    # ─────────────────────────────────────────────────────────────────────────
    # Outer loop: models
    # ─────────────────────────────────────────────────────────────────────────
    for model_template in get_all_models(seed=seed):
        model_name = model_template.name
        logger.info("\n" + "=" * 70)
        logger.info(f"MODEL: {model_name}")
        logger.info("=" * 70)
        all_results[model_name] = {}

        # ── Per-target loop ──
        for target_idx, target_name in enumerate(TARGET_NAMES):
            logger.info(f"\n  TARGET {target_idx + 1}/3: {target_name}")
            logger.info("  " + "-" * 60)

            fold_results  = {}
            last_model    = None
            last_train_mean = None
            last_tab_scaled = None

            # ── Walk-forward folds ──
            for fold_num, (train_slice, val_slice) in enumerate(
                walk_forward_split(data_len, window_size, step_size), start=1
            ):
                if fold_num > num_folds:
                    break

                logger.info(
                    f"  Fold {fold_num}/{num_folds} | "
                    f"train=[0:{train_slice.stop}] val=[{val_slice.start}:{val_slice.stop}]"
                )

                # Scale tabular features per-asset on this fold's train slice
                tab_scaled = fit_scale_tabular(
                    tabular_raw, btc_len, eth_len, train_slice
                )

                # Build train & val arrays
                train_starts = get_valid_starts(train_slice, btc_len, eth_len)
                val_starts   = get_valid_starts(val_slice,   btc_len, eth_len)

                X_train, y_train = build_Xy(tab_scaled, targets_eng, train_starts, target_idx)
                X_val,   y_val   = build_Xy(tab_scaled, targets_eng, val_starts,   target_idx)

                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning(f"  Fold {fold_num}: empty train or val — skipping")
                    continue

                train_mean = float(np.mean(y_train))

                # Fresh model for each fold (mirrors per-fold weight reset in DL)
                if isinstance(model_template, HistoricalMeanModel):
                    model = HistoricalMeanModel()
                elif isinstance(model_template, LinearRegressionModel):
                    model = LinearRegressionModel()
                else:
                    model = XGBoostModel(seed=seed)

                model.fit(X_train, y_train)

                val_preds = model.predict(X_val)
                val_metrics = compute_metrics(
                    val_preds, y_val, target_name=target_name,
                    train_targets_mean=train_mean
                )

                fold_results[fold_num] = val_metrics
                logger.info(
                    f"    Val → MAE={val_metrics['mae']:.4f}  "
                    f"RMSE={val_metrics['rmse']:.4f}  "
                    f"R²={val_metrics['r2']:.4f}  "
                    f"R²_OOS={val_metrics['r2_oos']:.4f}"
                )

                # Cache the last fold's model & scaling for test evaluation
                last_model      = model
                last_train_mean = train_mean
                last_tab_scaled = tab_scaled

            # ── Test set evaluation (last fold's model) ──
            if last_model is None:
                logger.warning(f"  No folds completed for {target_name} — skipping test")
                test_metrics = None
            else:
                test_starts = get_valid_starts(test_slice, btc_len, eth_len)
                X_test, y_test = build_Xy(
                    last_tab_scaled, targets_eng, test_starts, target_idx
                )
                test_preds   = last_model.predict(X_test)
                test_metrics = compute_metrics(
                    test_preds, y_test, target_name=target_name,
                    train_targets_mean=last_train_mean
                )
                logger.info(
                    f"\n  [TEST] {target_name} → "
                    f"MAE={test_metrics['mae']:.4f}  "
                    f"RMSE={test_metrics['rmse']:.4f}  "
                    f"R²={test_metrics['r2']:.4f}  "
                    f"R²_OOS={test_metrics['r2_oos']:.4f}  "
                    f"n={test_metrics['n_samples']}"
                )

            all_results[model_name][target_name] = {
                "fold_results": fold_results,
                "test_metrics": test_metrics,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────────
    out_file = out_path / "baseline_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ Results saved to {out_file}")

    # ── Summary table ──
    _print_summary(all_results)

    return all_results


def _print_summary(results: dict):
    """Print a compact cross-model, cross-target summary table."""
    header = f"\n{'='*90}"
    print(header)
    print("BASELINE RESULTS SUMMARY — Test Set Performance")
    print("=" * 90)
    fmt = "{:<22} {:<22} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format("Model", "Target", "MAE", "RMSE", "R²", "R²_OOS"))
    print("-" * 90)
    for model_name, targets in results.items():
        for target_name, res in targets.items():
            tm = res.get("test_metrics")
            if tm:
                print(fmt.format(
                    model_name, target_name,
                    f"{tm['mae']:.4f}",
                    f"{tm['rmse']:.4f}",
                    f"{tm['r2']:.4f}",
                    f"{tm['r2_oos']:.4f}",
                ))
    print("=" * 90)
    print("R²_OOS benchmark: Historical Mean for y_baseline; Zero-predictor for others (GKX 2020)")


# ─── CV summary helper (used in thesis reporting) ────────────────────────────

def print_cv_summary(results: dict):
    """Print fold-averaged CV metrics (mean ± std) per model per target."""
    print(f"\n{'='*90}")
    print("CROSS-VALIDATION SUMMARY (mean ± std across folds)")
    print("=" * 90)
    fmt = "{:<22} {:<22} {:>14} {:>14} {:>14}"
    print(fmt.format("Model", "Target", "MAE (CV)", "RMSE (CV)", "R²_OOS (CV)"))
    print("-" * 90)
    for model_name, targets in results.items():
        for target_name, res in targets.items():
            folds = res.get("fold_results", {})
            if not folds:
                continue
            maes    = [v["mae"]     for v in folds.values()]
            rmses   = [v["rmse"]    for v in folds.values()]
            r2ooss  = [v["r2_oos"]  for v in folds.values()]
            print(fmt.format(
                model_name, target_name,
                f"{np.mean(maes):.4f}±{np.std(maes):.4f}",
                f"{np.mean(rmses):.4f}±{np.std(rmses):.4f}",
                f"{np.mean(r2ooss):.4f}±{np.std(r2ooss):.4f}",
            ))
    print("=" * 90)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline models for crypto sentiment forecasting")
    parser.add_argument("--features-dir", type=str, default="./data/features",
                        help="Path to pre-extracted features directory (default: ./data/features)")
    parser.add_argument("--num-folds", type=int, default=5,
                        help="Number of walk-forward folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out-dir", type=str, default="./src/baseline/results",
                        help="Output directory for results JSON (default: ./src/baseline/results)")
    args = parser.parse_args()

    _setup_logging()
    results = run_baselines(
        features_dir=args.features_dir,
        num_folds=args.num_folds,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print_cv_summary(results)
