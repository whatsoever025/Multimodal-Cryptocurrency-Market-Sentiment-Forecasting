"""
Verify the two foundational Persistence-related numbers currently stated in the
thesis (chapter3.tex "Structural autocorrelation caveat" + Table 4.2), on the
EXACT scope the report implies: BTC+ETH combined, held-out TEST SET ONLY
(final 15% chronologically), at funding_horizon=8.

Numbers to check against the report:
    r  ~ 0.974  (Pearson correlation, funding_rate_t vs y_funding_{t+8})
    R2_OOS(Persistence vs Historical Mean) ~ 0.742

Neither number has a traceable, saved computation anywhere in this repo (confirmed
by exhaustive search of all .py files, all .ipynb notebooks, git history, and
baseline_results.json). This script is the first reproducible computation of both,
on the scope (BTC+ETH, test-set only) that Chapter 4's methodology describes.

Usage:
    python -m src.baseline.verify_persistence_baseline --features-dir /kaggle/input/<dataset>/features
"""

import argparse
import numpy as np

from .kaggle_baseline import (
    load_data, apply_target_engineering, get_valid_starts, build_Xy,
    build_raw_windows, PersistenceModel, compute_metrics, TEST_PCT, FUNDING_RATE_COL_IDX,
)


def verify(features_dir, funding_horizon=8):
    tabular_raw, targets_raw, btc_len, eth_len = load_data(features_dir)
    targets_eng = apply_target_engineering(targets_raw)

    data_len = int(btc_len * (1.0 - TEST_PCT))
    train_slice = slice(0, data_len)
    test_slice = slice(data_len, btc_len)

    # ── Train-set mean (for Historical Mean benchmark, mirrors train.py convention) ──
    train_starts = get_valid_starts(train_slice, btc_len, eth_len, buffer=funding_horizon)
    _, y_train = build_Xy(tabular_raw, targets_eng, train_starts, target_idx=0,
                           funding_horizon=funding_horizon)
    train_mean = float(np.mean(y_train))

    # ── Test set only (BTC + ETH combined, matches Chapter 4's held-out 15%) ────────
    test_starts = get_valid_starts(test_slice, btc_len, eth_len, buffer=funding_horizon)
    _, y_test = build_Xy(tabular_raw, targets_eng, test_starts, target_idx=0,
                          funding_horizon=funding_horizon)

    # ── (1) Pearson correlation: funding_rate_t vs y_funding_{t+horizon}, TEST SET ONLY ──
    raw_windows = build_raw_windows(tabular_raw, test_starts)
    funding_rate_t = raw_windows[:, -1, FUNDING_RATE_COL_IDX].astype(np.float64)
    # y_test is already x1000-scaled (target engineering); undo that to compare against
    # the raw funding_rate_t on the same scale.
    y_test_raw_scale = y_test.astype(np.float64) / 1000.0
    r = float(np.corrcoef(funding_rate_t, y_test_raw_scale)[0, 1])

    # ── (2) Persistence R²_OOS vs Historical Mean, TEST SET ONLY ────────────────────
    persistence_preds = PersistenceModel.predict_from_raw(raw_windows)
    metrics = compute_metrics(persistence_preds, y_test, target_name="y_baseline",
                               train_targets_mean=train_mean)

    print(f"Scope: BTC+ETH combined, held-out test set only (final {TEST_PCT:.0%}), "
          f"funding_horizon={funding_horizon}h, n={len(y_test)}")
    print(f"  Pearson r(funding_rate_t, y_funding_t+{funding_horizon}) = {r:.4f}   "
          f"(report currently states ~0.974)")
    print(f"  R2_OOS(Persistence vs Historical Mean)                  = {metrics['r2_oos']:.4f}   "
          f"(report currently states ~0.742)")

    return {"r": r, "persistence_r2_oos_vs_histmean": metrics["r2_oos"], "n_samples": len(y_test)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--funding-horizon", type=int, default=8, choices=[8, 16, 24])
    args = parser.parse_args()
    verify(args.features_dir, args.funding_horizon)
