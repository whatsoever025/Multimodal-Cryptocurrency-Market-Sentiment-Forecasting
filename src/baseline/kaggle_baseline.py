"""
Baseline Models — Single-file Kaggle version (no package imports required).

Upload this single file to your Kaggle notebook and run:

    !pip install xgboost -q
    %run kaggle_baseline.py

Or from a code cell:

    exec(open("/kaggle/working/kaggle_baseline.py").read())
    results = run_baselines(features_dir="/kaggle/working/data/features")
    print_cv_summary(results)

Baselines:
    1. HistoricalMean    — predicts training-set mean (naive benchmark)
    2. LinearRegression  — sklearn on flattened 24-step tabular window (168 features)
    3. XGBoost           — gradient boosting on flattened 24-step tabular window

Mirrors EXACTLY:
    - Walk-forward splits  : src/training/dataset.py
    - Target engineering   : src/training/dataset.py
    - Valid-start logic    : src/training/dataset.py (WalkForwardDataset)
    - Metrics / R²_OOS     : src/training/train.py (_compute_metrics)
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_XGBOOST = False
    print("[WARNING] xgboost not found — using sklearn GradientBoostingRegressor as fallback")

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Constants (must match src/training/config.py & dataset.py) ──────────────

TARGET_NAMES  = ["y_baseline", "y_heuristic", "y_vol_adj_return"]
SEQ_LEN       = 24      # 24-hour sliding window
N_TABULAR     = 7       # number of tabular features
BUFFER        = 8       # default funding_horizon; prevents t+funding_horizon index overflow
TEST_PCT      = 0.15    # last 15% = hold-out test set (per asset)
INIT_WIN_PCT  = 0.70    # initial training window fraction
VAL_FOLD_PCT  = 0.15    # validation pool fraction

# Column order per src/preprocessing/data_aligner.py final_columns (tabular section):
# ["return_1h", "volume", "funding_rate", "gdelt_econ_volume", "gdelt_econ_tone",
#  "gdelt_conflict_volume", "is_post_ETF"]
FUNDING_RATE_COL_IDX = 2

# =============================================================================
# SECTION 1: METRICS
# =============================================================================

def compute_metrics(predictions, targets, target_name=None, train_targets_mean=None):
    """
    Compute MAE, RMSE, R², R²_OOS — mirrors train.py _compute_metrics().

    R²_OOS benchmark per target:
        y_baseline       → Historical Mean  (funding rate has structural positive bias)
        y_heuristic      → Zero-predictor   (GKX 2020; Z-score mean ≈ 0)
        y_vol_adj_return → Zero-predictor   (GKX 2020; log-return mean ≈ 0)
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets     = np.asarray(targets,     dtype=np.float64)

    mae  = float(np.mean(np.abs(predictions - targets)))
    mse  = float(np.mean((predictions - targets) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((predictions - targets) ** 2))

    # Standard R²
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # R²_OOS — GKX 2020 zero-predictor
    ss_tot_gkx  = float(np.sum(targets ** 2))
    r2_oos_gkx  = 1.0 - ss_res / ss_tot_gkx if ss_tot_gkx > 0 else 0.0

    # R²_OOS — Historical Mean benchmark
    if train_targets_mean is not None:
        ss_tot_hist = float(np.sum((targets - train_targets_mean) ** 2))
        r2_oos_hist = 1.0 - ss_res / ss_tot_hist if ss_tot_hist > 0 else 0.0
    else:
        r2_oos_hist = 0.0

    # Select primary R²_OOS per-target (mirrors train.py)
    if target_name == "y_baseline":
        r2_oos = r2_oos_hist
    else:
        r2_oos = r2_oos_gkx

    corr = (
        float(np.corrcoef(predictions, targets)[0, 1])
        if len(predictions) > 1 and predictions.std() > 0 and targets.std() > 0
        else 0.0
    )

    return {
        "mae":              mae,
        "mse":              mse,
        "rmse":             rmse,
        "r2":               r2,
        "r2_oos":           r2_oos,
        "r2_oos_gkx":       r2_oos_gkx,
        "r2_oos_hist":      r2_oos_hist,
        "correlation":      corr,
        "prediction_bias":  float(np.mean(predictions - targets)),
        "n_samples":        int(len(targets)),
    }

from scipy.stats import norm

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1):
    """Diebold-Mariano test for non-nested models (e.g., XGBoost vs MFN)."""
    e1, e2 = y_true - y_pred1, y_true - y_pred2
    d = e1**2 - e2**2
    T = len(d)
    mean_d = np.mean(d)
    
    lag = h - 1
    gamma = np.zeros(lag + 1)
    for i in range(lag + 1):
        gamma[i] = np.sum((d - mean_d)**2)/T if i == 0 else np.sum((d[i:] - mean_d) * (d[:-i] - mean_d))/T
            
    var_d = gamma[0] + sum(2 * (1.0 - i/(lag + 1)) * gamma[i] for i in range(1, lag + 1))
    if var_d == 0: return 0.0, 1.0
    
    dm_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)

def clark_west_test(y_true, y_pred1, y_pred2, h=1):
    """Clark-West test for nested models (e.g., Historical Mean vs XGBoost)."""
    e1, e2 = y_true - y_pred1, y_true - y_pred2
    f = e1**2 - e2**2 + (y_pred1 - y_pred2)**2
    T = len(f)
    mean_f = np.mean(f)
    
    lag = h - 1
    gamma = np.zeros(lag + 1)
    for i in range(lag + 1):
        gamma[i] = np.sum((f - mean_f)**2)/T if i == 0 else np.sum((f[i:] - mean_f) * (f[:-i] - mean_f))/T
            
    var_f = gamma[0] + sum(2 * (1.0 - i/(lag + 1)) * gamma[i] for i in range(1, lag + 1))
    if var_f == 0: return 0.0, 1.0
    
    cw_stat = mean_f / np.sqrt(var_f / T)
    p_value = 1 - norm.cdf(cw_stat)
    return float(cw_stat), float(p_value)

# =============================================================================
# SECTION 2: MODELS
# =============================================================================

class HistoricalMeanModel:
    """Predicts training-set mean for every sample — naive benchmark."""
    name = "HistoricalMean"

    def __init__(self):
        self.train_mean = 0.0

    def fit(self, X, y):
        self.train_mean = float(np.mean(y))
        logger.info(f"  [{self.name}] train_mean = {self.train_mean:.6f}")
        return self

    def predict(self, X):
        return np.full(len(X), self.train_mean, dtype=np.float32)


class PersistenceModel:
    """
    Persistence baseline: ŷ = funding_rate_t (raw, ×1000 to match target engineering).

    Only meaningful for y_baseline (funding) — no natural analogue exists for
    y_heuristic / y_vol_adj_return, which lack a directly-persistable raw input
    feature (thesis Section 4.1). run_baselines() only evaluates this model for
    target_idx == 0 and skips it for the other two targets.

    Rule-based: no fitting required. Unlike the other models, prediction is
    computed from the RAW (unscaled) tabular window via predict_from_raw() —
    NOT from the StandardScaler-transformed, flattened X used by the other
    baselines — because z-scoring funding_rate would destroy the raw value
    this baseline is defined on.
    """
    name = "Persistence"

    def fit(self, X, y):
        return self

    @staticmethod
    def predict_from_raw(tabular_raw_windows):
        """
        Args:
            tabular_raw_windows: (n, seq_len, n_features) — RAW, unscaled tabular windows.
        Returns:
            (n,) — funding_rate at the last input timestep of each window, ×1000.
        """
        last_step_funding_rate = tabular_raw_windows[:, -1, FUNDING_RATE_COL_IDX]
        return (last_step_funding_rate * 1000.0).astype(np.float32)


class LinearRegressionModel:
    """sklearn LinearRegression on flattened 24-step tabular window (168 features)."""
    name = "LinearRegression"

    def __init__(self):
        self._model = LinearRegression(fit_intercept=True, n_jobs=-1)

    def fit(self, X, y):
        self._model.fit(X, y)
        logger.info(f"  [{self.name}] fitted | n={X.shape[0]}, features={X.shape[1]}")
        return self

    def predict(self, X):
        return self._model.predict(X).astype(np.float32)


class XGBoostModel:
    """XGBRegressor (fallback: GradientBoostingRegressor) on flattened 24-step window."""
    name = "XGBoost" if _HAS_XGBOOST else "GradientBoosting"

    def __init__(self, n_estimators=500, max_depth=6, learning_rate=0.05, seed=42):
        if _HAS_XGBOOST:
            self._model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
            )
        else:
            self._model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                random_state=seed,
            )

    def fit(self, X, y):
        self._model.fit(X, y)
        logger.info(f"  [{self.name}] fitted | n={X.shape[0]}")
        return self

    def predict(self, X):
        return self._model.predict(X).astype(np.float32)


def _fresh_model(template, seed=42):
    """Return a new instance of the same model class (mirrors per-fold weight reset)."""
    if isinstance(template, HistoricalMeanModel):
        return HistoricalMeanModel()
    elif isinstance(template, PersistenceModel):
        return PersistenceModel()
    elif isinstance(template, LinearRegressionModel):
        return LinearRegressionModel()
    else:
        return XGBoostModel(seed=seed)

# =============================================================================
# SECTION 3: DATA HELPERS
# =============================================================================

def load_data(features_dir):
    """Load tabular features and target scores for BTC and ETH."""
    features_dir = Path(features_dir)
    btc_dir = features_dir / "BTC"
    eth_dir = features_dir / "ETH"

    if not (btc_dir.exists() and eth_dir.exists()):
        raise FileNotFoundError(
            f"Expected BTC/ and ETH/ subdirs inside {features_dir}.\n"
            f"Found: {list(features_dir.iterdir())}"
        )

    btc_tab = torch.load(btc_dir / "tabular_features.pt", map_location="cpu").numpy().astype(np.float32)
    btc_tgt = torch.load(btc_dir / "target_scores.pt",    map_location="cpu").numpy().astype(np.float32)
    eth_tab = torch.load(eth_dir / "tabular_features.pt", map_location="cpu").numpy().astype(np.float32)
    eth_tgt = torch.load(eth_dir / "target_scores.pt",    map_location="cpu").numpy().astype(np.float32)

    tabular = np.concatenate([btc_tab, eth_tab], axis=0)
    targets = np.concatenate([btc_tgt, eth_tgt], axis=0)

    logger.info(f"Data loaded — BTC: {btc_tab.shape[0]} | ETH: {eth_tab.shape[0]} | Total: {tabular.shape[0]}")
    return tabular, targets, btc_tab.shape[0], eth_tab.shape[0]


def apply_target_engineering(targets):
    """
    Mirror dataset.py target engineering:
        col 0 (y_baseline):       × 1000   (prevent gradient underflow)
        col 1 (y_heuristic):      clip to [-5, 5]
        col 2 (y_vol_adj_return): unchanged
    """
    t = targets.copy()
    t[:, 0] *= 1000.0
    t[:, 1] = np.clip(t[:, 1], -5.0, 5.0)
    return t


def walk_forward_split(data_len, window_size, step_size):
    """Identical to dataset.py walk_forward_split()."""
    for i in range(0, data_len - window_size - step_size, step_size):
        train_end = i + window_size
        val_end   = min(train_end + step_size, data_len)
        yield slice(0, train_end), slice(train_end, val_end)


def get_valid_starts(data_slice, btc_len, eth_len, seq_len=SEQ_LEN, buffer=BUFFER):
    """
    Mirrors WalkForwardDataset.__init__ valid_starts logic:
        - Never cross asset boundary (BTC / ETH are separate timelines)
        - Leave buffer=funding_horizon steps at end to safely access the
          funding target index (t + seq_len + funding_horizon - 1).
          Pass buffer=funding_horizon explicitly when funding_horizon != 8
          (the diagnostic horizon-sensitivity check).
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


def fit_scale_tabular(tabular_raw, btc_len, eth_len, train_slice):
    """
    Fit StandardScaler per-asset on train_slice; apply to full array.
    Mirrors create_walk_forward_dataloaders() scaler logic in dataset.py.
    """
    start = train_slice.start or 0
    stop  = train_slice.stop

    btc_scaler = StandardScaler().fit(tabular_raw[start:stop])
    eth_scaler = StandardScaler().fit(tabular_raw[start + btc_len: stop + btc_len])

    scaled = tabular_raw.copy()
    scaled[:btc_len]               = btc_scaler.transform(tabular_raw[:btc_len])
    scaled[btc_len: btc_len + eth_len] = eth_scaler.transform(tabular_raw[btc_len: btc_len + eth_len])
    return scaled


def build_Xy(tabular_scaled, targets_eng, valid_starts, target_idx, seq_len=SEQ_LEN, funding_horizon=BUFFER):
    """
    Build (X, y) arrays from valid start indices.
        X shape: (n, seq_len * N_TABULAR) — flattened 24-step window
        y shape: (n,)                      — raw engineered target
    Target horizons (mirror WalkForwardDataset.__getitem__):
        y_baseline (idx=0): t + seq_len + (funding_horizon - 1)  (funding_horizon hours ahead;
                            default 8. 16/24 are diagnostic-only robustness checks — see
                            thesis Section 4.x, "Funding-Rate Horizon Sensitivity".)
        y_heuristic (idx=1): t + seq_len     (1 hour ahead, unaffected by funding_horizon)
        y_vol_adj_return (idx=2): t + seq_len (1 hour ahead, unaffected by funding_horizon)
    """
    funding_offset = funding_horizon - 1
    X, y = [], []
    for i in valid_starts:
        X.append(tabular_scaled[i: i + seq_len].flatten())
        if target_idx == 0:
            y.append(targets_eng[i + seq_len + funding_offset, 0])
        elif target_idx == 1:
            y.append(targets_eng[i + seq_len, 1])
        else:
            y.append(targets_eng[i + seq_len, 2])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_raw_windows(tabular_raw, valid_starts, seq_len=SEQ_LEN):
    """
    (n, seq_len, n_features) RAW (unscaled) tabular windows — used only by
    PersistenceModel, which needs the true funding_rate value, not its
    per-fold StandardScaler-transformed z-score.
    """
    return np.stack([tabular_raw[i: i + seq_len] for i in valid_starts], axis=0)

# =============================================================================
# SECTION 4: MAIN RUNNER
# =============================================================================

def run_baselines(features_dir="./data/features", num_folds=5, seed=42,
                  out_dir="./baseline_results", funding_horizon=BUFFER):
    """
    Run all baseline models with the exact walk-forward protocol as MultimodalFusionNet.

    Args:
        funding_horizon: Prediction horizon in hours for y_baseline (funding). Default 8
            = thesis's primary scope. 16/24 are diagnostic-only robustness checks (2/3
            settlement cycles) — see thesis Section 4.x, "Funding-Rate Horizon
            Sensitivity" — NOT a proposed change of scope. Does not affect
            y_heuristic / y_vol_adj_return (fixed at t+1h).

    Returns:
        dict: { model_name: { target_name: { "fold_results": {...}, "test_metrics": {...} } } }
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if funding_horizon != BUFFER:
        logger.warning(
            f"⚠ DIAGNOSTIC RUN: funding_horizon={funding_horizon}h (default={BUFFER}h). "
            f"This is a robustness check on the funding target's autocorrelation decay, "
            f"not a proposed operating horizon — the deployed system remains fixed at t+8h."
        )

    # ── Load & engineer data ──────────────────────────────────────────────────
    tabular_raw, targets_raw, btc_len, eth_len = load_data(features_dir)
    targets_eng = apply_target_engineering(targets_raw)

    # ── Walk-forward parameters (must match dataset.py) ───────────────────────
    data_len    = int(btc_len * (1.0 - TEST_PCT))   # per-asset train+val length
    window_size = int(INIT_WIN_PCT * data_len)
    step_size   = int(VAL_FOLD_PCT * data_len) // num_folds
    test_start  = data_len                           # per-asset test start index
    test_slice  = slice(test_start, btc_len)

    logger.info(
        f"\nWalk-forward config: data_len={data_len} | window={window_size} | "
        f"step={step_size} | folds={num_folds} | "
        f"test=[{test_start}:{btc_len}] per asset | funding_horizon={funding_horizon}h"
    )

    model_templates = [HistoricalMeanModel(), PersistenceModel(), LinearRegressionModel(), XGBoostModel(seed=seed)]
    all_results = {}
    
    # Dictionary to store raw test predictions for significance testing
    # Format: test_preds_cache[target_name][model_name] = predictions_array
    test_preds_cache = {t: {} for t in TARGET_NAMES}
    test_targets_cache = {}

    # ── Model loop ────────────────────────────────────────────────────────────
    for template in model_templates:
        model_name = template.name
        logger.info("\n" + "=" * 70)
        logger.info(f"  MODEL: {model_name}")
        logger.info("=" * 70)
        all_results[model_name] = {}

        # ── Target loop ───────────────────────────────────────────────────────
        for target_idx, target_name in enumerate(TARGET_NAMES):
            if isinstance(template, PersistenceModel) and target_idx != 0:
                logger.info(
                    f"\n  TARGET {target_idx + 1}/3: {target_name} — skipping Persistence "
                    f"(undefined: no raw input feature analogue; see thesis Section 4.1)"
                )
                continue

            logger.info(f"\n  TARGET {target_idx + 1}/3: {target_name}")

            fold_results    = {}
            last_model      = None
            last_train_mean = None
            last_tab_scaled = None
            fold_count      = 0

            # ── Walk-forward fold loop ────────────────────────────────────────
            for train_slice, val_slice in walk_forward_split(data_len, window_size, step_size):
                fold_count += 1
                if fold_count > num_folds:
                    break

                logger.info(
                    f"  Fold {fold_count}/{num_folds} | "
                    f"train=[0:{train_slice.stop}] | val=[{val_slice.start}:{val_slice.stop}]"
                )

                # Scale tabular per-asset on this fold's training slice
                tab_scaled = fit_scale_tabular(tabular_raw, btc_len, eth_len, train_slice)

                # Build arrays
                train_starts = get_valid_starts(train_slice, btc_len, eth_len, buffer=funding_horizon)
                val_starts   = get_valid_starts(val_slice,   btc_len, eth_len, buffer=funding_horizon)

                X_train, y_train = build_Xy(tab_scaled, targets_eng, train_starts, target_idx, funding_horizon=funding_horizon)
                X_val,   y_val   = build_Xy(tab_scaled, targets_eng, val_starts,   target_idx, funding_horizon=funding_horizon)

                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning(f"  Fold {fold_count}: empty split — skipping")
                    continue

                train_mean = float(np.mean(y_train))

                # Fresh model per fold (mirrors per-fold weight reset)
                model = _fresh_model(template, seed=seed)
                model.fit(X_train, y_train)

                if isinstance(model, PersistenceModel):
                    # Persistence needs the RAW funding_rate, not the z-scored X_val
                    raw_val_windows = build_raw_windows(tabular_raw, val_starts)
                    val_preds = PersistenceModel.predict_from_raw(raw_val_windows)
                else:
                    val_preds = model.predict(X_val)
                val_metrics = compute_metrics(
                    val_preds, y_val,
                    target_name=target_name,
                    train_targets_mean=train_mean,
                )

                fold_results[fold_count] = val_metrics
                logger.info(
                    f"    → MAE={val_metrics['mae']:.5f}  RMSE={val_metrics['rmse']:.5f}"
                    f"  R²={val_metrics['r2']:.5f}  R²_OOS={val_metrics['r2_oos']:.5f}"
                )

                # Cache last fold state for test evaluation
                last_model      = model
                last_train_mean = train_mean
                last_tab_scaled = tab_scaled

            # ── Test evaluation (uses last fold's model & scaler) ─────────────
            if last_model is None:
                test_metrics = None
                logger.warning(f"  No folds ran for {target_name}")
            else:
                test_starts          = get_valid_starts(test_slice, btc_len, eth_len, buffer=funding_horizon)
                X_test, y_test       = build_Xy(last_tab_scaled, targets_eng, test_starts, target_idx, funding_horizon=funding_horizon)
                if isinstance(last_model, PersistenceModel):
                    raw_test_windows = build_raw_windows(tabular_raw, test_starts)
                    test_preds = PersistenceModel.predict_from_raw(raw_test_windows)
                else:
                    test_preds = last_model.predict(X_test)

                # Cache predictions and targets for significance testing
                test_preds_cache[target_name][model_name] = test_preds
                if target_name not in test_targets_cache:
                    test_targets_cache[target_name] = y_test
                    
                test_metrics         = compute_metrics(
                    test_preds, y_test,
                    target_name=target_name,
                    train_targets_mean=last_train_mean,
                )
                logger.info(
                    f"\n  [TEST] {target_name} → "
                    f"MAE={test_metrics['mae']:.5f}  RMSE={test_metrics['rmse']:.5f}  "
                    f"R²={test_metrics['r2']:.5f}  R²_OOS={test_metrics['r2_oos']:.5f}  "
                    f"n={test_metrics['n_samples']}"
                )
                
                # Automatic Significance Testing vs Historical Mean
                if model_name != "HistoricalMean" and "HistoricalMean" in test_preds_cache[target_name]:
                    h = funding_horizon if target_name == "y_baseline" else 1
                    y_pred_hist = test_preds_cache[target_name]["HistoricalMean"]
                    cw_stat, cw_pval = clark_west_test(y_test, y_pred_hist, test_preds, h=h)
                    logger.info(f"         Clark-West test vs HistoricalMean: stat={cw_stat:.4f}, p={cw_pval:.4f}")
                    test_metrics["cw_pval_vs_histmean"] = cw_pval

            all_results[model_name][target_name] = {
                "fold_results": fold_results,
                "test_metrics": test_metrics,
            }

    # ── Save predictions to disk for DM testing with MFN ──────────────────────
    preds_file = out_path / "baseline_predictions.npz"
    save_dict = {}
    for tgt, models_dict in test_preds_cache.items():
        save_dict[f"{tgt}_y_true"] = test_targets_cache[tgt]
        for mod, preds in models_dict.items():
            save_dict[f"{tgt}_{mod}"] = preds
    np.savez(preds_file, **save_dict)
    logger.info(f"\n✓ Test predictions saved → {preds_file} (Use this for DM tests vs MFN)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_file = out_path / "baseline_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ Results saved → {out_file}")

    _print_test_summary(all_results)
    return all_results


# =============================================================================
# SECTION 5: REPORTING
# =============================================================================

def _print_test_summary(results):
    """Compact test-set summary table."""
    print(f"\n{'='*92}")
    print("BASELINE TEST-SET RESULTS")
    print("=" * 92)
    fmt = "{:<22} {:<22} {:>9} {:>9} {:>9} {:>9} {:>8}"
    print(fmt.format("Model", "Target", "MAE", "RMSE", "R²", "R²_OOS", "n"))
    print("-" * 92)
    for model_name, targets in results.items():
        for target_name, res in targets.items():
            tm = res.get("test_metrics")
            if tm:
                print(fmt.format(
                    model_name, target_name,
                    f"{tm['mae']:.4f}", f"{tm['rmse']:.4f}",
                    f"{tm['r2']:.4f}",  f"{tm['r2_oos']:.4f}",
                    str(tm["n_samples"]),
                ))
    print("=" * 92)
    print("R²_OOS: Historical Mean benchmark for y_baseline; Zero-predictor for others (GKX 2020)\n")


def print_cv_summary(results):
    """Cross-validation summary: mean ± std across folds."""
    print(f"\n{'='*92}")
    print("BASELINE CROSS-VALIDATION SUMMARY (mean ± std across folds)")
    print("=" * 92)
    fmt = "{:<22} {:<22} {:>16} {:>16} {:>16}"
    print(fmt.format("Model", "Target", "MAE (CV)", "RMSE (CV)", "R²_OOS (CV)"))
    print("-" * 92)
    for model_name, targets in results.items():
        for target_name, res in targets.items():
            folds = res.get("fold_results", {})
            if not folds:
                continue
            maes   = [v["mae"]    for v in folds.values()]
            rmses  = [v["rmse"]   for v in folds.values()]
            r2ooss = [v["r2_oos"] for v in folds.values()]
            print(fmt.format(
                model_name, target_name,
                f"{np.mean(maes):.4f}±{np.std(maes):.4f}",
                f"{np.mean(rmses):.4f}±{np.std(rmses):.4f}",
                f"{np.mean(r2ooss):.4f}±{np.std(r2ooss):.4f}",
            ))
    print("=" * 92)


# =============================================================================
# SECTION 6: ENTRY POINT (when run as script)
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--features-dir", default="./data/features",
                        help="Path to pre-extracted features (BTC/ and ETH/ subdirs)")
    parser.add_argument("--num-folds",    type=int, default=5)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--out-dir",      default="./baseline_results")
    # parse_known_args ignores unknown args (e.g. Jupyter/Colab kernel flags)
    args, _ = parser.parse_known_args()

    results = run_baselines(
        features_dir=args.features_dir,
        num_folds=args.num_folds,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print_cv_summary(results)
