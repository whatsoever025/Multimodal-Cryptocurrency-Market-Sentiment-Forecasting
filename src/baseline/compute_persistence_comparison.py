"""
Post-hoc comparison: MFN vs Persistence, at a given funding_horizon.

train.py only computes R²_OOS vs Historical Mean / Zero-predictor for MFN — it has
no notion of the Persistence benchmark (ŷ = funding_rate_t). This script recomputes
Persistence-relative R²_OOS + a Diebold-Mariano test against MFN's saved predictions,
for any funding_horizon, without needing to touch the training pipeline again.

Usage (Kaggle, after running train.py with a given --funding-horizon):

    from src.baseline.compute_persistence_comparison import compare_to_persistence

    compare_to_persistence(
        mfn_npz_path="./checkpoints/mfn_test_predictions_tabular_only_h16.npz",
        features_dir="/kaggle/input/<dataset>/features",
        funding_horizon=16,
    )

Requires: the SAME features_dir used for the training run (so the independently
reconstructed y_true matches MFN's saved y_true exactly — this is checked
automatically and will raise if they don't match).
"""

import numpy as np

from .kaggle_baseline import (
    load_data,
    get_valid_starts,
    build_raw_windows,
    apply_target_engineering,
    PersistenceModel,
    compute_metrics,
    diebold_mariano_test,
    TEST_PCT,
)


def compare_to_persistence(mfn_npz_path, features_dir, funding_horizon=8,
                            target_key="y_baseline", verbose=True):
    """
    Args:
        mfn_npz_path: Path to the .npz saved by train.py
                      (mfn_test_predictions_{ablation}_h{funding_horizon}.npz),
                      must contain f"{target_key}_MFN" and f"{target_key}_y_true".
        features_dir: Same features_dir used for the training run (BTC/ETH subdirs).
        funding_horizon: Must match the horizon the MFN run was trained/evaluated at.
        target_key: Which target's predictions to compare (only "y_baseline" is
                    meaningful for Persistence — see thesis Section 4.1).

    Returns:
        dict with mfn_r2_oos_vs_persistence, persistence_r2_oos_vs_histmean,
        dm_stat, dm_pval, n_samples.
    """
    npz = np.load(mfn_npz_path)
    mfn_key = f"{target_key}_MFN"
    ytrue_key = f"{target_key}_y_true"
    if mfn_key not in npz or ytrue_key not in npz:
        raise KeyError(
            f"{mfn_npz_path} is missing {mfn_key!r} or {ytrue_key!r}. "
            f"Make sure train.py was re-run with the targets-saving fix "
            f"(saves both predictions and y_true, not just predictions)."
        )
    mfn_preds = npz[mfn_key].astype(np.float64)
    mfn_y_true = npz[ytrue_key].astype(np.float64)
    n = len(mfn_y_true)

    # ── Independently reconstruct the same test slice, at the same horizon ──────
    tabular_raw, targets_raw, btc_len, eth_len = load_data(features_dir)
    targets_eng = apply_target_engineering(targets_raw)

    data_len = int(btc_len * (1.0 - TEST_PCT))
    test_slice = slice(data_len, btc_len)
    test_starts = get_valid_starts(test_slice, btc_len, eth_len, buffer=funding_horizon)

    from .kaggle_baseline import build_Xy
    _, y_true_reconstructed = build_Xy(
        tabular_raw, targets_eng, test_starts, target_idx=0, funding_horizon=funding_horizon
    )

    # ── Consistency check: MFN's saved y_true must match what we reconstruct here ──
    if len(y_true_reconstructed) != n:
        raise ValueError(
            f"Sample count mismatch: MFN saved n={n}, reconstructed n={len(y_true_reconstructed)}. "
            f"Did you use the same features_dir / funding_horizon / num_folds as the training run?"
        )
    max_abs_diff = float(np.max(np.abs(y_true_reconstructed - mfn_y_true)))
    if max_abs_diff > 1e-2:
        raise ValueError(
            f"MFN's saved y_true does NOT match the independently reconstructed y_true "
            f"(max abs diff={max_abs_diff:.4f}). Do not trust a cross-comparison built on "
            f"mismatched sample order — check features_dir/funding_horizon/num_folds match "
            f"exactly what was used for the train.py run that produced {mfn_npz_path}."
        )
    if verbose:
        print(f"✓ Consistency check passed: MFN saved y_true matches reconstruction "
              f"(max abs diff={max_abs_diff:.6f}, n={n})")

    # ── Persistence predictions for the exact same samples ──────────────────────
    raw_windows = build_raw_windows(tabular_raw, test_starts)
    persistence_preds = PersistenceModel.predict_from_raw(raw_windows).astype(np.float64)

    # Historical mean for R²_OOS benchmarking (train-set mean, mirrors train.py convention)
    train_slice = slice(0, data_len)
    train_starts = get_valid_starts(train_slice, btc_len, eth_len, buffer=funding_horizon)
    _, y_train = build_Xy(tabular_raw, targets_eng, train_starts, target_idx=0,
                           funding_horizon=funding_horizon)
    train_mean = float(np.mean(y_train))

    # ── Metrics ──────────────────────────────────────────────────────────────────
    persistence_metrics = compute_metrics(
        persistence_preds, mfn_y_true, target_name="y_baseline", train_targets_mean=train_mean
    )

    # R²_OOS of MFN specifically against Persistence (not historical mean / zero):
    ss_res_mfn = float(np.sum((mfn_preds - mfn_y_true) ** 2))
    ss_tot_vs_persistence = float(np.sum((mfn_y_true - persistence_preds) ** 2))
    mfn_r2_oos_vs_persistence = (
        1.0 - ss_res_mfn / ss_tot_vs_persistence if ss_tot_vs_persistence > 0 else float("nan")
    )

    dm_stat, dm_pval = diebold_mariano_test(mfn_y_true, persistence_preds, mfn_preds, h=funding_horizon)

    result = {
        "funding_horizon": funding_horizon,
        "n_samples": n,
        "persistence_r2_oos_vs_histmean": persistence_metrics["r2_oos"],
        "mfn_r2_oos_vs_persistence": mfn_r2_oos_vs_persistence,
        "dm_stat_mfn_vs_persistence": dm_stat,
        "dm_pval_mfn_vs_persistence": dm_pval,
    }

    if verbose:
        print(f"\n=== Horizon = {funding_horizon}h (n={n}) ===")
        print(f"  Persistence R²_OOS (vs Historical Mean): {result['persistence_r2_oos_vs_histmean']:.4f}")
        print(f"  MFN        R²_OOS (vs Persistence):      {result['mfn_r2_oos_vs_persistence']:.4f}")
        print(f"  DM test (MFN vs Persistence): stat={dm_stat:.4f}, p={dm_pval:.4f}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mfn-npz", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--funding-horizon", type=int, default=8, choices=[8, 16, 24])
    args = parser.parse_args()
    compare_to_persistence(args.mfn_npz, args.features_dir, args.funding_horizon)
